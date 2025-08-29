#!/usr/bin/env python3
"""
Corpus Deduplicator
Uses OpenAlex as spine for deduplication across multiple sources
Implements DOI → arXiv ID → MinHash title similarity matching
"""

import json
import sqlite3
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from datasketch import MinHash, MinHashLSH
import pandas as pd
import re
from urllib.parse import urlparse
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusDeduplicator:
    def __init__(self, corpus_dir: str = "/home/ubuntu/LW_scrape/multi_source_corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.manifest_db = self.corpus_dir / "multi_source_manifest.db"
        
        # MinHash LSH for near-duplicate detection
        self.lsh = MinHashLSH(threshold=0.8, num_perm=128)
        self.minhash_to_paper = {}
        
        # Tracking
        self.dedup_stats = {
            'total_papers': 0,
            'doi_duplicates': 0,
            'arxiv_duplicates': 0,
            'title_duplicates': 0,
            'unique_papers': 0,
            'merged_records': 0
        }
        
    def normalize_identifier(self, identifier: str, id_type: str) -> str:
        """Normalize identifiers for consistent matching"""
        if not identifier:
            return ""
            
        if id_type == 'doi':
            # Remove doi: prefix and normalize
            clean_doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', identifier)
            clean_doi = re.sub(r'^doi:', '', clean_doi)
            return clean_doi.lower().strip()
        
        elif id_type == 'arxiv':
            # Extract arXiv ID from various formats
            arxiv_match = re.search(r'(\d{4}\.\d{4,5}v?\d*)', identifier)
            if arxiv_match:
                return arxiv_match.group(1)
            return identifier.strip()
        
        elif id_type == 'openalex':
            # Extract OpenAlex ID
            if identifier.startswith('https://openalex.org/'):
                return identifier.split('/')[-1]
            return identifier.strip()
        
        return identifier.strip()
    
    def create_minhash(self, title: str, authors: str = "", year: int = None) -> MinHash:
        """Create MinHash for near-duplicate detection"""
        # Normalize text for comparison
        text_parts = []
        
        # Title normalization
        if title:
            # Remove common prefixes/suffixes and normalize
            clean_title = re.sub(r'^(a|an|the)\s+', '', title.lower())
            clean_title = re.sub(r'[^\w\s]', '', clean_title)
            text_parts.append(clean_title)
        
        # Authors normalization
        if authors:
            # Take first few authors for comparison
            author_parts = [name.strip().lower() for name in authors.split(';')[:3]]
            text_parts.extend(author_parts)
        
        # Year
        if year:
            text_parts.append(str(year))
        
        # Create MinHash
        minhash = MinHash(num_perm=128)
        combined_text = ' '.join(text_parts)
        
        for token in set(combined_text.split()):
            if len(token) > 2:  # Skip very short tokens
                minhash.update(token.encode('utf8'))
        
        return minhash
    
    def load_papers_from_json(self, json_file: str) -> List[Dict]:
        """Load papers from JSON file"""
        json_path = Path(json_file)
        if not json_path.exists():
            logger.error(f"Papers file not found: {json_file}")
            return []
            
        with open(json_path, 'r') as f:
            papers = json.load(f)
            
        logger.info(f"📚 Loaded {len(papers)} papers from {json_file}")
        return papers
    
    def merge_paper_records(self, primary: Dict, secondary: Dict) -> Dict:
        """Merge two paper records, keeping best information"""
        merged = primary.copy()
        
        # Update sources list
        primary_sources = json.loads(primary.get('sources', '[]'))
        secondary_sources = json.loads(secondary.get('sources', '[]'))
        combined_sources = list(set(primary_sources + secondary_sources))
        merged['sources'] = json.dumps(combined_sources)
        
        # Take best available information
        for field in ['abstract', 'authors', 'venue', 'publication_date', 'keywords']:
            if not merged.get(field) and secondary.get(field):
                merged[field] = secondary[field]
        
        # Take higher citation count
        if secondary.get('cited_by_count', 0) > primary.get('cited_by_count', 0):
            merged['cited_by_count'] = secondary['cited_by_count']
        
        # Merge PDF URLs - prefer open access
        if not merged.get('pdf_url') and secondary.get('pdf_url'):
            merged['pdf_url'] = secondary['pdf_url']
        elif secondary.get('is_open_access') and not primary.get('is_open_access'):
            merged['pdf_url'] = secondary.get('pdf_url', primary.get('pdf_url'))
        
        # Take higher relevance score
        if secondary.get('relevance_score', 0) > primary.get('relevance_score', 0):
            merged['relevance_score'] = secondary['relevance_score']
        
        # Merge identifiers
        for id_field in ['openalex_id', 'arxiv_id', 'openreview_id', 'doi']:
            if not merged.get(id_field) and secondary.get(id_field):
                merged[id_field] = secondary[id_field]
        
        # Merge referenced works
        primary_refs = json.loads(primary.get('referenced_works', '[]'))
        secondary_refs = json.loads(secondary.get('referenced_works', '[]'))
        combined_refs = list(set(primary_refs + secondary_refs))
        merged['referenced_works'] = json.dumps(combined_refs)
        
        self.dedup_stats['merged_records'] += 1
        return merged
    
    def deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Deduplicate papers using multi-stage approach"""
        logger.info("🔍 Starting deduplication process...")
        
        self.dedup_stats['total_papers'] = len(papers)
        
        # Stage 1: DOI-based deduplication
        doi_map = {}
        doi_deduplicated = []
        
        for paper in papers:
            doi = self.normalize_identifier(paper.get('doi', ''), 'doi')
            if doi:
                if doi in doi_map:
                    # Merge with existing
                    doi_map[doi] = self.merge_paper_records(doi_map[doi], paper)
                    self.dedup_stats['doi_duplicates'] += 1
                else:
                    doi_map[doi] = paper
                    doi_deduplicated.append(paper)
            else:
                doi_deduplicated.append(paper)
        
        logger.info(f"📊 Stage 1 - DOI dedup: {len(papers)} → {len(doi_deduplicated)} ({self.dedup_stats['doi_duplicates']} duplicates)")
        
        # Stage 2: arXiv ID-based deduplication
        arxiv_map = {}
        arxiv_deduplicated = []
        
        for paper in doi_deduplicated:
            arxiv_id = self.normalize_identifier(paper.get('arxiv_id', ''), 'arxiv')
            if arxiv_id:
                if arxiv_id in arxiv_map:
                    # Merge with existing
                    arxiv_map[arxiv_id] = self.merge_paper_records(arxiv_map[arxiv_id], paper)
                    self.dedup_stats['arxiv_duplicates'] += 1
                else:
                    arxiv_map[arxiv_id] = paper
                    arxiv_deduplicated.append(paper)
            else:
                arxiv_deduplicated.append(paper)
        
        logger.info(f"📊 Stage 2 - arXiv dedup: {len(doi_deduplicated)} → {len(arxiv_deduplicated)} ({self.dedup_stats['arxiv_duplicates']} duplicates)")
        
        # Stage 3: MinHash-based title similarity
        final_papers = []
        
        for paper in arxiv_deduplicated:
            title = paper.get('title', '')
            authors = paper.get('authors', '')
            year = paper.get('publication_year')
            
            if not title:
                final_papers.append(paper)
                continue
            
            minhash = self.create_minhash(title, authors, year)
            
            # Check for similar papers
            candidates = self.lsh.query(minhash)
            
            if candidates:
                # Merge with most similar existing paper
                existing_paper = self.minhash_to_paper[candidates[0]]
                merged_paper = self.merge_paper_records(existing_paper, paper)
                
                # Update the existing paper in the final list
                for i, final_paper in enumerate(final_papers):
                    if final_paper is existing_paper:
                        final_papers[i] = merged_paper
                        break
                
                self.dedup_stats['title_duplicates'] += 1
                
            else:
                # Add as new paper
                paper_id = f"paper_{len(self.minhash_to_paper)}"
                self.lsh.insert(paper_id, minhash)
                self.minhash_to_paper[paper_id] = paper
                final_papers.append(paper)
        
        logger.info(f"📊 Stage 3 - Title dedup: {len(arxiv_deduplicated)} → {len(final_papers)} ({self.dedup_stats['title_duplicates']} duplicates)")
        
        self.dedup_stats['unique_papers'] = len(final_papers)
        
        return final_papers
    
    def assign_paper_ids(self, papers: List[Dict]) -> List[Dict]:
        """Assign consistent paper IDs based on available identifiers"""
        for paper in papers:
            # Priority: OpenAlex ID → DOI → arXiv ID → generated hash
            paper_id = None
            
            if paper.get('openalex_id'):
                paper_id = self.normalize_identifier(paper['openalex_id'], 'openalex')
            elif paper.get('doi'):
                paper_id = f"doi_{self.normalize_identifier(paper['doi'], 'doi').replace('/', '_')}"
            elif paper.get('arxiv_id'):
                paper_id = f"arxiv_{self.normalize_identifier(paper['arxiv_id'], 'arxiv')}"
            else:
                # Generate hash-based ID
                title = paper.get('title', '')
                authors = paper.get('authors', '')
                content = f"{title}_{authors}".encode('utf-8')
                paper_id = f"hash_{hashlib.md5(content).hexdigest()[:16]}"
            
            paper['paper_id'] = paper_id
        
        return papers
    
    def save_deduplicated_papers(self, papers: List[Dict]):
        """Save deduplicated papers to database and JSON"""
        
        # Save to JSON for backup
        output_file = self.corpus_dir / "deduplicated_papers.json"
        with open(output_file, 'w') as f:
            json.dump(papers, f, indent=2, default=str)
        
        # Save to database
        conn = sqlite3.connect(self.manifest_db)
        cur = conn.cursor()
        
        # Clear existing data
        cur.execute("DELETE FROM papers")
        
        # Insert deduplicated papers
        for paper in papers:
            cur.execute("""
                INSERT INTO papers (
                    openalex_id, arxiv_id, openreview_id, doi, title,
                    authors, publication_year, publication_date, venue, abstract,
                    cited_by_count, concepts, keywords, pdf_url, primary_source,
                    sources, referenced_works, citing_works, is_open_access, relevance_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.get('openalex_id'),
                paper.get('arxiv_id'),
                paper.get('openreview_id'),
                paper.get('doi'),
                paper.get('title'),
                paper.get('authors'),
                paper.get('publication_year'),
                paper.get('publication_date'),
                paper.get('venue'),
                paper.get('abstract'),
                paper.get('cited_by_count', 0),
                paper.get('concepts'),
                paper.get('keywords'),
                paper.get('pdf_url'),
                paper.get('primary_source'),
                paper.get('sources'),
                paper.get('referenced_works'),
                paper.get('citing_works'),
                paper.get('is_open_access', False),
                paper.get('relevance_score', 0.0)
            ))
        
        conn.commit()
        conn.close()
        
        # Save statistics
        stats_file = self.corpus_dir / "deduplication_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.dedup_stats, f, indent=2)
        
        logger.info(f"💾 Saved {len(papers)} deduplicated papers to database and {output_file}")
        logger.info(f"📊 Deduplication stats saved to {stats_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate multi-source corpus")
    parser.add_argument("--corpus-dir", default="/home/ubuntu/LW_scrape/multi_source_corpus")
    parser.add_argument("--input-json", default="raw_papers_collected.json")
    
    args = parser.parse_args()
    
    deduplicator = CorpusDeduplicator(args.corpus_dir)
    
    # Load papers
    input_file = Path(args.corpus_dir) / args.input_json
    papers = deduplicator.load_papers_from_json(input_file)
    
    if not papers:
        logger.error("No papers to deduplicate")
        return
    
    # Deduplicate
    deduplicated_papers = deduplicator.deduplicate_papers(papers)
    
    # Assign consistent IDs
    deduplicated_papers = deduplicator.assign_paper_ids(deduplicated_papers)
    
    # Save results
    deduplicator.save_deduplicated_papers(deduplicated_papers)
    
    logger.info("="*60)
    logger.info("🎯 DEDUPLICATION COMPLETE")
    logger.info("="*60)
    logger.info(f"📊 Total papers processed: {deduplicator.dedup_stats['total_papers']}")
    logger.info(f"🔄 DOI duplicates merged: {deduplicator.dedup_stats['doi_duplicates']}")
    logger.info(f"🔄 arXiv duplicates merged: {deduplicator.dedup_stats['arxiv_duplicates']}")
    logger.info(f"🔄 Title duplicates merged: {deduplicator.dedup_stats['title_duplicates']}")
    logger.info(f"✅ Unique papers retained: {deduplicator.dedup_stats['unique_papers']}")
    logger.info(f"🔗 Records merged: {deduplicator.dedup_stats['merged_records']}")
    logger.info("="*60)

if __name__ == "__main__":
    main()