#!/usr/bin/env python3
"""
Multi-Source Corpus Builder
Comprehensive academic paper discovery using OpenAlex as spine + arXiv, OpenReview, Crossref, Semantic Scholar
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
import requests
import feedparser
import openreview
from datasketch import MinHash, MinHashLSH
import pandas as pd

from directory_config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = load_config()


class MultiSourceCorpusBuilder:
    def __init__(self, corpus_dir: str = str(config.corpus_dir),
                 output_dir: str = str(config.output_dir),
                 temp_dir: str = str(config.temp_dir)):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)

        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize databases
        self.manifest_db = self.output_dir / "multi_source_manifest.db"
        self.init_database()
        
        # Alignment keywords for filtering
        self.alignment_keywords = [
            "alignment", "interpretability", "safety", "robustness",
            "reward modeling", "rlhf", "rlaif", "constitutional ai",
            "oversight", "amplification", "debate", "scalable oversight",
            "mesa-optimizer", "inner alignment", "outer alignment", 
            "deception", "reward hacking", "goal misgeneralization",
            "ai safety", "existential risk", "x-risk", "catastrophic risk",
            "mechanistic interpretability", "activation patching", "sae",
            "elk", "truthfulness", "honesty", "calibration"
        ]
        
        # Rate limiting
        self.last_request_time = {}
        self.rate_limits = {
            'openalex': 0.1,    # 10 req/sec
            'arxiv': 3.0,       # Conservative for arXiv
            'openreview': 1.0,  # 1 req/sec
            'crossref': 0.1,    # 10 req/sec
            'semantic_scholar': 0.1  # 10 req/sec
        }
        
    def init_database(self):
        """Initialize SQLite database for manifest"""
        conn = sqlite3.connect(self.manifest_db)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                openalex_id TEXT,
                arxiv_id TEXT,
                openreview_id TEXT,
                doi TEXT,
                title TEXT NOT NULL,
                authors TEXT,
                publication_year INTEGER,
                publication_date TEXT,
                venue TEXT,
                abstract TEXT,
                cited_by_count INTEGER,
                concepts TEXT,
                keywords TEXT,
                pdf_url TEXT,
                primary_source TEXT,
                sources TEXT,
                referenced_works TEXT,
                citing_works TEXT,
                is_open_access BOOLEAN,
                relevance_score REAL,
                raw_metadata TEXT,
                etag TEXT,
                last_modified TEXT,
                retrieved_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(openalex_id),
                UNIQUE(arxiv_id),
                UNIQUE(doi)
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id INTEGER,
                target_paper_id INTEGER,
                citation_type TEXT,
                FOREIGN KEY (source_paper_id) REFERENCES papers (id),
                FOREIGN KEY (target_paper_id) REFERENCES papers (id),
                UNIQUE(source_paper_id, target_paper_id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"📁 Database initialized: {self.manifest_db}")

    def rate_limit(self, service: str):
        """Apply rate limiting for API calls"""
        if service in self.last_request_time:
            elapsed = time.time() - self.last_request_time[service]
            required_wait = self.rate_limits.get(service, 1.0)
            if elapsed < required_wait:
                sleep_time = required_wait - elapsed
                time.sleep(sleep_time)
        self.last_request_time[service] = time.time()
        
    def calculate_alignment_relevance(self, title: str, abstract: str = "", concepts: List[str] = None) -> float:
        """Calculate alignment relevance score based on content"""
        text = f"{title} {abstract}".lower()
        
        # Keyword matching with weights
        critical_keywords = ["alignment", "ai safety", "existential risk", "mesa-optimizer"]
        high_keywords = ["interpretability", "robustness", "reward modeling", "rlhf"]
        medium_keywords = ["oversight", "amplification", "debate", "deception"]
        
        score = 0.0
        word_count = len(text.split())
        
        for keyword in critical_keywords:
            if keyword in text:
                score += 2.0 * text.count(keyword) / word_count * 1000
                
        for keyword in high_keywords:
            if keyword in text:
                score += 1.5 * text.count(keyword) / word_count * 1000
                
        for keyword in medium_keywords:
            if keyword in text:
                score += 1.0 * text.count(keyword) / word_count * 1000
        
        # Concept-based scoring if available
        if concepts:
            ai_concepts = ["artificial intelligence", "machine learning", "natural language processing"]
            safety_concepts = ["safety", "security", "robustness", "reliability"]
            
            for concept in concepts:
                concept_lower = concept.lower()
                if any(ai_term in concept_lower for ai_term in ai_concepts):
                    score += 0.3
                if any(safety_term in concept_lower for safety_term in safety_concepts):
                    score += 0.5
        
        return min(score, 1.0)  # Cap at 1.0

    def get_cached_headers(self, id_field: str, identifier: str) -> Dict[str, str]:
        """Retrieve cached ETag and Last-Modified headers for a record"""
        conn = sqlite3.connect(self.manifest_db)
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT etag, last_modified FROM papers WHERE {id_field} = ?", (identifier,))
            row = cur.fetchone()
        finally:
            conn.close()

        if row:
            return {'etag': row[0], 'last_modified': row[1]}
        return {}

class OpenAlexConnector:
    def __init__(self, parent: MultiSourceCorpusBuilder):
        self.parent = parent
        self.base_url = "https://api.openalex.org/works"

    def fetch_alignment_papers(self, per_page: int = 200, max_pages: int = 10) -> List[Dict]:
        """Fetch alignment papers from OpenAlex as the primary spine"""
        papers = []
        
        # Multiple search strategies
        search_queries = [
            "alignment artificial intelligence",
            "AI safety interpretability", 
            "reward modeling RLHF",
            "mesa-optimizer inner alignment",
            "AI oversight amplification"
        ]
        
        for query in search_queries:
            logger.info(f"🔍 OpenAlex search: {query}")
            
            for page in range(1, max_pages + 1):
                self.parent.rate_limit('openalex')
                
                params = {
                    'search': query,
                    'filter': 'concepts.level:0|1|2,publication_year:>2018',
                    'per_page': per_page,
                    'page': page,
                    'select': 'id,title,doi,publication_year,publication_date,primary_location,open_access,authorships,concepts,abstract_inverted_index,cited_by_count,referenced_works,related_works'
                }
                
                try:
                    response = requests.get(self.base_url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get('results'):
                        break
                        
                    for work in data['results']:
                        paper = self.fetch_work(work.get('id'))
                        if paper and paper.get('relevance_score', 0) > 0.1:  # Filter by relevance
                            papers.append(paper)

                    logger.info(
                        f"📄 Page {page}: {len(data['results'])} papers, "
                        f"{len([w for w in data['results'] if self.parse_openalex_work(w) and self.parse_openalex_work(w).get('relevance_score', 0) > 0.1])} relevant"
                    )
                    
                except Exception as e:
                    logger.error(f"OpenAlex API error: {e}")
                    break
                    
        logger.info(f"✅ OpenAlex: {len(papers)} alignment papers collected")
        return papers

    def fetch_work(self, work_id: str) -> Optional[Dict]:
        """Fetch a single OpenAlex work with caching headers"""
        if not work_id:
            return None
        cached = self.parent.get_cached_headers('openalex_id', work_id)
        headers = {}
        if cached.get('etag'):
            headers['If-None-Match'] = cached['etag']
        if cached.get('last_modified'):
            headers['If-Modified-Since'] = cached['last_modified']

        self.parent.rate_limit('openalex')
        url = f"{self.base_url}/{work_id.split('/')[-1]}" if '/' in work_id else f"{self.base_url}/{work_id}"
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 304:
            logger.info(f"🔄 OpenAlex {work_id} not modified")
            return None
        response.raise_for_status()

        data = response.json()
        paper = self.parse_openalex_work(data)
        if paper:
            paper['raw_metadata'] = json.dumps(data)
        paper['etag'] = response.headers.get('ETag')
        paper['last_modified'] = response.headers.get('Last-Modified')
        paper['retrieved_at'] = datetime.utcnow().isoformat()
        return paper
    
    def parse_openalex_work(self, work: Dict) -> Optional[Dict]:
        """Parse OpenAlex work into standardized format"""
        try:
            # Extract authors
            authors = []
            authorships = work.get('authorships', [])
            for authorship in authorships[:5]:  # Limit to first 5 authors
                author = authorship.get('author', {})
                if author.get('display_name'):
                    authors.append(author['display_name'])
            
            # Reconstruct abstract from inverted index
            abstract = ""
            abstract_idx = work.get('abstract_inverted_index')
            if abstract_idx:
                words = {}
                for word, positions in abstract_idx.items():
                    for pos in positions:
                        words[pos] = word
                if words:
                    abstract = " ".join([words[i] for i in sorted(words.keys())])
            
            # Extract concepts
            concepts = [concept.get('display_name', '') for concept in work.get('concepts', [])]
            
            # Calculate relevance
            title = work.get('title', '')
            relevance_score = self.parent.calculate_alignment_relevance(title, abstract, concepts)
            
            # Extract PDF URL
            pdf_url = None
            primary_location = work.get('primary_location', {})
            if work.get('open_access', {}).get('is_oa'):
                pdf_url = primary_location.get('pdf_url')
            
            paper = {
                'openalex_id': work.get('id'),
                'title': title,
                'doi': work.get('doi'),
                'authors': '; '.join(authors),
                'publication_year': work.get('publication_year'),
                'publication_date': work.get('publication_date'),
                'venue': primary_location.get('source', {}).get('display_name'),
                'abstract': abstract[:2000] if abstract else "",  # Truncate long abstracts
                'cited_by_count': work.get('cited_by_count', 0),
                'concepts': json.dumps(concepts),
                'keywords': ', '.join(self.parent.alignment_keywords[:10]),  # Add matched keywords
                'pdf_url': pdf_url,
                'primary_source': 'openalex',
                'sources': json.dumps(['openalex']),
                'referenced_works': json.dumps(work.get('referenced_works', [])),
                'citing_works': json.dumps([]),  # Will be populated later
                'is_open_access': work.get('open_access', {}).get('is_oa', False),
                'relevance_score': relevance_score
            }
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error parsing OpenAlex work: {e}")
            return None

class ArxivConnector:
    def __init__(self, parent: MultiSourceCorpusBuilder):
        self.parent = parent
        self.base_url = "http://export.arxiv.org/api/query"
        
    def fetch_alignment_papers(self, max_results: int = 1000) -> List[Dict]:
        """Fetch alignment papers from arXiv"""
        papers = []
        
        categories = ['cs.AI', 'cs.LG', 'cs.CL', 'stat.ML']
        keyword_query = ' OR '.join([f'"{kw}"' for kw in self.parent.alignment_keywords[:10]])
        
        for category in categories:
            logger.info(f"🔍 arXiv search: {category}")
            
            search_query = f"cat:{category} AND ({keyword_query})"
            
            self.parent.rate_limit('arxiv')
            
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': max_results // len(categories),
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }
            
            headers = {'User-Agent': 'multi-source-corpus-builder/1.0'}
            
            try:
                response = requests.get(self.base_url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                feed = feedparser.parse(response.text)

                for entry in feed.entries:
                    arxiv_id = entry.id.split('/abs/')[-1]
                    paper = self.fetch_entry(arxiv_id)
                    if paper and paper.get('relevance_score', 0) > 0.1:
                        papers.append(paper)

                logger.info(
                    f"📄 {category}: {len(feed.entries)} papers, "
                    f"{len([e for e in feed.entries if self.parse_arxiv_entry(e) and self.parse_arxiv_entry(e).get('relevance_score', 0) > 0.1])} relevant"
                )

            except Exception as e:
                logger.error(f"arXiv API error for {category}: {e}")
                
        logger.info(f"✅ arXiv: {len(papers)} alignment papers collected")
        return papers

    def fetch_entry(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch a single arXiv entry with caching headers"""
        cached = self.parent.get_cached_headers('arxiv_id', arxiv_id)
        headers = {'User-Agent': 'multi-source-corpus-builder/1.0'}
        if cached.get('etag'):
            headers['If-None-Match'] = cached['etag']
        if cached.get('last_modified'):
            headers['If-Modified-Since'] = cached['last_modified']

        params = {'id_list': arxiv_id}
        self.parent.rate_limit('arxiv')
        response = requests.get(self.base_url, params=params, headers=headers, timeout=30)
        if response.status_code == 304:
            logger.info(f"🔄 arXiv {arxiv_id} not modified")
            return None
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        if not feed.entries:
            return None
        entry = feed.entries[0]
        paper = self.parse_arxiv_entry(entry)
        if paper:
            paper['raw_metadata'] = json.dumps(dict(entry))
        paper['etag'] = response.headers.get('ETag')
        paper['last_modified'] = response.headers.get('Last-Modified')
        paper['retrieved_at'] = datetime.utcnow().isoformat()
        return paper
    
    def parse_arxiv_entry(self, entry) -> Optional[Dict]:
        """Parse arXiv entry into standardized format"""
        try:
            # Extract arXiv ID
            arxiv_id = entry.id.split('/abs/')[-1]
            
            # Extract authors
            authors = [author.name for author in entry.get('authors', [])]
            
            # Extract date
            pub_date = None
            pub_year = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_year = entry.published_parsed.tm_year
                pub_date = f"{entry.published_parsed.tm_year}-{entry.published_parsed.tm_mon:02d}-{entry.published_parsed.tm_mday:02d}"
            
            # Calculate relevance
            title = entry.get('title', '')
            abstract = entry.get('summary', '')
            relevance_score = self.parent.calculate_alignment_relevance(title, abstract)
            
            paper = {
                'arxiv_id': arxiv_id,
                'title': title,
                'doi': getattr(entry, 'arxiv_doi', None),
                'authors': '; '.join(authors),
                'publication_year': pub_year,
                'publication_date': pub_date,
                'venue': 'arXiv preprint',
                'abstract': abstract[:2000] if abstract else "",
                'cited_by_count': 0,  # arXiv doesn't provide citation counts
                'concepts': json.dumps([]),
                'keywords': ', '.join([kw for kw in self.parent.alignment_keywords if kw.lower() in f"{title} {abstract}".lower()]),
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                'primary_source': 'arxiv',
                'sources': json.dumps(['arxiv']),
                'referenced_works': json.dumps([]),
                'citing_works': json.dumps([]),
                'is_open_access': True,  # arXiv papers are open access
                'relevance_score': relevance_score
            }
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None

class OpenReviewConnector:
    def __init__(self, parent: MultiSourceCorpusBuilder):
        self.parent = parent
        self.client = openreview.Client(baseurl='https://api.openreview.net')
        
        # Major AI conferences
        self.venues = {
            'ICLR': ['ICLR.cc/2024', 'ICLR.cc/2023', 'ICLR.cc/2022'],
            'NeurIPS': ['NeurIPS.cc/2023', 'NeurIPS.cc/2022', 'NeurIPS.cc/2021'],
            'ICML': ['ICML.cc/2023', 'ICML.cc/2022', 'ICML.cc/2021']
        }
        
    def fetch_alignment_papers(self, limit_per_venue: int = 200) -> List[Dict]:
        """Fetch alignment papers from OpenReview"""
        papers = []
        
        for conf_name, venues in self.venues.items():
            for venue in venues:
                logger.info(f"🔍 OpenReview search: {venue}")
                
                try:
                    self.parent.rate_limit('openreview')
                    
                    # Get accepted papers from venue
                    notes = self.client.get_notes(
                        content={'venue': venue},
                        details='all',
                        limit=limit_per_venue
                    )
                    
                    for note in notes:
                        paper = self.fetch_note(note.id, conf_name)
                        if paper and paper.get('relevance_score', 0) > 0.1:
                            papers.append(paper)

                    logger.info(
                        f"📄 {venue}: {len(notes)} papers, "
                        f"{len([n for n in notes if self.parse_openreview_note(n.to_json(), conf_name) and self.parse_openreview_note(n.to_json(), conf_name).get('relevance_score', 0) > 0.1])} relevant"
                    )
                    
                except Exception as e:
                    logger.error(f"OpenReview API error for {venue}: {e}")
                    
        logger.info(f"✅ OpenReview: {len(papers)} alignment papers collected")
        return papers

    def fetch_note(self, note_id: str, conf_name: str) -> Optional[Dict]:
        """Fetch a single OpenReview note with caching headers"""
        cached = self.parent.get_cached_headers('openreview_id', note_id)
        headers = {}
        if cached.get('etag'):
            headers['If-None-Match'] = cached['etag']
        if cached.get('last_modified'):
            headers['If-Modified-Since'] = cached['last_modified']

        params = {'id': note_id, 'details': 'all'}
        self.parent.rate_limit('openreview')
        response = requests.get('https://api.openreview.net/notes', params=params, headers=headers, timeout=30)
        if response.status_code == 304:
            logger.info(f"🔄 OpenReview {note_id} not modified")
            return None
        response.raise_for_status()

        data = response.json()
        notes = data.get('notes', [])
        if not notes:
            return None
        note = notes[0]
        paper = self.parse_openreview_note(note, conf_name)
        if paper:
            paper['raw_metadata'] = json.dumps(note)
        paper['etag'] = response.headers.get('ETag')
        paper['last_modified'] = response.headers.get('Last-Modified')
        paper['retrieved_at'] = datetime.utcnow().isoformat()
        return paper
    
    def parse_openreview_note(self, note: Dict, conf_name: str) -> Optional[Dict]:
        """Parse OpenReview note into standardized format"""
        try:
            content = note.get('content', {})
            
            title = content.get('title', '')
            abstract = content.get('abstract', '')
            
            # Skip if no alignment relevance
            relevance_score = self.parent.calculate_alignment_relevance(title, abstract)
            if relevance_score < 0.1:
                return None
            
            # Extract authors
            authors = content.get('authors', [])
            if isinstance(authors, str):
                authors = [authors]
                
            # Extract year from venue
            year_match = re.search(r'(20\d{2})', content.get('venue', ''))
            pub_year = int(year_match.group()) if year_match else None
            
            paper = {
                'openreview_id': note.get('id'),
                'title': title,
                'doi': None,  # OpenReview doesn't typically have DOIs
                'authors': '; '.join(authors[:5]) if authors else "",
                'publication_year': pub_year,
                'publication_date': None,
                'venue': f"{conf_name} {pub_year}" if pub_year else conf_name,
                'abstract': abstract[:2000] if abstract else "",
                'cited_by_count': 0,  # OpenReview doesn't provide citation counts
                'concepts': json.dumps([]),
                'keywords': content.get('keywords', []),
                'pdf_url': f"https://openreview.net/pdf?id={note.get('id')}",
                'primary_source': 'openreview',
                'sources': json.dumps(['openreview']),
                'referenced_works': json.dumps([]),
                'citing_works': json.dumps([]),
                'is_open_access': True,  # OpenReview papers are typically open access
                'relevance_score': relevance_score
            }
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error parsing OpenReview note: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Multi-source corpus builder for AI alignment papers")
    parser.add_argument("--corpus-dir", default=str(config.corpus_dir))
    parser.add_argument("--output-dir", default=str(config.output_dir))
    parser.add_argument("--temp-dir", default=str(config.temp_dir))
    parser.add_argument("--openalex-pages", type=int, default=5, help="Max pages per OpenAlex query")
    parser.add_argument("--arxiv-results", type=int, default=1000, help="Max results from arXiv")
    parser.add_argument("--openreview-limit", type=int, default=200, help="Max papers per venue from OpenReview")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use cached data")

    args = parser.parse_args()

    builder = MultiSourceCorpusBuilder(args.corpus_dir, args.output_dir, args.temp_dir)
    
    logger.info("🚀 Starting multi-source corpus building...")
    
    all_papers = []
    
    if not args.dry_run:
        # 1. OpenAlex (primary spine)
        openalex_connector = OpenAlexConnector(builder)
        openalex_papers = openalex_connector.fetch_alignment_papers(max_pages=args.openalex_pages)
        all_papers.extend(openalex_papers)
        
        # 2. arXiv
        arxiv_connector = ArxivConnector(builder)
        arxiv_papers = arxiv_connector.fetch_alignment_papers(max_results=args.arxiv_results)
        all_papers.extend(arxiv_papers)
        
        # 3. OpenReview
        openreview_connector = OpenReviewConnector(builder)
        openreview_papers = openreview_connector.fetch_alignment_papers(limit_per_venue=args.openreview_limit)
        all_papers.extend(openreview_papers)
    
    # Save collected papers
    if all_papers:
        papers_file = builder.output_dir / "raw_papers_collected.json"
        with open(papers_file, 'w') as f:
            json.dump(all_papers, f, indent=2, default=str)
            
        logger.info(f"💾 Saved {len(all_papers)} raw papers to {papers_file}")
    
    logger.info("="*60)
    logger.info("🎯 MULTI-SOURCE COLLECTION COMPLETE")
    logger.info("="*60)
    logger.info(f"📊 Total papers collected: {len(all_papers)}")
    logger.info(f"📁 Output directory: {builder.output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()