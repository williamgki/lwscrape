#!/usr/bin/env python3
"""
Reference Normalizer
Extract references from GROBID TEI, normalize via Crossref, map to OpenAlex canonical IDs
"""

import json
import xml.etree.ElementTree as ET
import requests
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re
from datetime import datetime
from fuzzywuzzy import fuzz
from dataclasses import dataclass
import hashlib
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NormalizedReference:
    """Normalized reference with canonical identifiers"""
    ref_id: str
    raw_text: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: str
    doi: Optional[str]
    crossref_doi: Optional[str]
    openalex_id: Optional[str]
    crossref_score: float
    normalization_method: str  # 'direct_doi', 'crossref_lookup', 'failed'
    canonical_title: str
    canonical_authors: List[str]
    canonical_venue: str
    citation_count: int
    referenced_works: List[str]
    citing_works: List[str]
    concepts: List[str]

class CrossrefClient:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'alignment-corpus-normalizer/1.0 (mailto:research@example.com)'
        })
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
    
    def rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def lookup_by_doi(self, doi: str) -> Optional[Dict]:
        """Lookup reference by DOI"""
        if not doi:
            return None
        
        self.rate_limit()
        
        try:
            # Clean DOI
            clean_doi = doi.strip().lower()
            if clean_doi.startswith('http'):
                clean_doi = clean_doi.split('/')[-2] + '/' + clean_doi.split('/')[-1]
            clean_doi = re.sub(r'^doi:', '', clean_doi)
            
            url = f"{self.base_url}/{clean_doi}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {})
            else:
                logger.debug(f"DOI lookup failed {response.status_code}: {doi}")
                return None
                
        except Exception as e:
            logger.debug(f"DOI lookup error for {doi}: {e}")
            return None
    
    def search_by_metadata(self, title: str, authors: List[str], year: Optional[int]) -> Optional[Tuple[Dict, float]]:
        """Search Crossref by metadata with fuzzy matching"""
        if not title or len(title.strip()) < 10:
            return None
        
        self.rate_limit()
        
        try:
            # Build query
            query_parts = [title.strip()]
            
            # Add first author if available
            if authors:
                first_author = authors[0].split()[-1] if authors[0] else ""  # Last name
                if first_author:
                    query_parts.append(first_author)
            
            # Add year if available
            if year and 1900 <= year <= 2030:
                query_parts.append(str(year))
            
            query = " ".join(query_parts)
            
            params = {
                'query': query,
                'rows': 5,  # Get top 5 matches
                'sort': 'score',
                'order': 'desc'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                # Find best match with fuzzy scoring
                best_match = None
                best_score = 0.0
                
                for item in items:
                    score = self.calculate_match_score(title, authors, year, item)
                    if score > best_score and score > 0.7:  # Minimum threshold
                        best_match = item
                        best_score = score
                
                if best_match:
                    return best_match, best_score
            
            return None
            
        except Exception as e:
            logger.debug(f"Metadata search error for '{title[:50]}': {e}")
            return None
    
    def calculate_match_score(self, query_title: str, query_authors: List[str], 
                            query_year: Optional[int], crossref_item: Dict) -> float:
        """Calculate fuzzy match score between query and Crossref item"""
        
        scores = []
        
        # Title similarity (most important)
        cr_title = crossref_item.get('title', [''])[0] if crossref_item.get('title') else ""
        if cr_title and query_title:
            title_score = fuzz.token_sort_ratio(query_title.lower(), cr_title.lower()) / 100.0
            scores.append(title_score * 0.6)  # 60% weight
        
        # Author similarity
        cr_authors = []
        for author in crossref_item.get('author', []):
            given = author.get('given', '')
            family = author.get('family', '')
            if family:
                cr_authors.append(f"{given} {family}".strip())
        
        if query_authors and cr_authors:
            # Compare first authors
            if query_authors[0] and cr_authors:
                author_score = max([
                    fuzz.token_sort_ratio(query_authors[0].lower(), cr_author.lower()) / 100.0
                    for cr_author in cr_authors[:2]  # Check first 2 authors
                ])
                scores.append(author_score * 0.3)  # 30% weight
        
        # Year similarity
        cr_year = None
        published = crossref_item.get('published-print') or crossref_item.get('published-online')
        if published and 'date-parts' in published:
            try:
                cr_year = published['date-parts'][0][0]
            except:
                pass
        
        if query_year and cr_year:
            year_diff = abs(query_year - cr_year)
            if year_diff <= 1:
                scores.append(0.1)  # 10% weight for exact year match
            elif year_diff <= 2:
                scores.append(0.05)  # 5% weight for close year match
        
        return sum(scores) if scores else 0.0

class OpenAlexClient:
    def __init__(self):
        self.base_url = "https://api.openalex.org/works"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 10 requests per second
        self.last_request_time = 0
    
    def rate_limit(self):
        """Apply rate limiting for OpenAlex"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def lookup_by_doi(self, doi: str) -> Optional[Dict]:
        """Get OpenAlex work by DOI"""
        if not doi:
            return None
        
        self.rate_limit()
        
        try:
            # Clean DOI for OpenAlex format
            clean_doi = doi.strip().lower()
            if clean_doi.startswith('http'):
                clean_doi = clean_doi.split('/')[-2] + '/' + clean_doi.split('/')[-1]
            clean_doi = re.sub(r'^doi:', '', clean_doi)
            
            url = f"{self.base_url}/doi:{clean_doi}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.debug(f"OpenAlex DOI lookup failed {response.status_code}: {doi}")
                return None
                
        except Exception as e:
            logger.debug(f"OpenAlex lookup error for {doi}: {e}")
            return None

class ReferenceNormalizer:
    def __init__(self, corpus_dir: str = "/home/ubuntu/LW_scrape/multi_source_corpus"):
        self.corpus_dir = Path(corpus_dir)
        
        # Initialize directories
        self.tei_dir = self.corpus_dir / "grobid_tei"
        self.normalized_refs_dir = self.corpus_dir / "normalized_references"
        self.normalized_refs_dir.mkdir(exist_ok=True)
        
        # Initialize clients
        self.crossref = CrossrefClient()
        self.openalex = OpenAlexClient()
        
        # TEI namespace
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        # Cache for lookups
        self.doi_cache = {}
        self.crossref_cache = {}
        
        # Statistics
        self.stats = {
            'total_references': 0,
            'direct_doi_matches': 0,
            'crossref_lookups': 0,
            'openalex_mappings': 0,
            'failed_normalizations': 0,
            'cached_lookups': 0
        }
    
    def extract_references_from_tei(self, tei_file: Path) -> List[Dict]:
        """Extract references from GROBID TEI XML"""
        
        try:
            tree = ET.parse(tei_file)
            root = tree.getroot()
            
            references = []
            
            # Find bibliography section
            back = root.find('.//tei:text/tei:back', self.ns)
            if back is not None:
                listbibl = back.find('.//tei:listBibl', self.ns)
                if listbibl is not None:
                    for i, biblstruct in enumerate(listbibl.findall('./tei:biblStruct', self.ns)):
                        ref = self.parse_biblstruct(biblstruct, i, tei_file.stem)
                        if ref:
                            references.append(ref)
            
            logger.debug(f"Extracted {len(references)} references from {tei_file.name}")
            return references
            
        except Exception as e:
            logger.error(f"Error extracting references from {tei_file}: {e}")
            return []
    
    def parse_biblstruct(self, biblstruct: ET.Element, index: int, doc_id: str) -> Optional[Dict]:
        """Parse individual biblStruct element"""
        
        try:
            # Generate reference ID
            ref_id = f"{doc_id}_ref_{index:03d}"
            
            # Extract title
            title_elem = biblstruct.find('.//tei:title[@level="a"]', self.ns)
            if title_elem is None:
                title_elem = biblstruct.find('.//tei:title', self.ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            
            # Extract authors
            authors = []
            for author in biblstruct.findall('.//tei:author', self.ns):
                name_parts = []
                
                # Given names
                for given in author.findall('.//tei:forename', self.ns):
                    if given.text and given.text.strip():
                        name_parts.append(given.text.strip())
                
                # Surname
                surname = author.find('.//tei:surname', self.ns)
                if surname is not None and surname.text:
                    name_parts.append(surname.text.strip())
                
                if name_parts:
                    authors.append(' '.join(name_parts))
            
            # Extract venue (journal/conference)
            venue_elem = biblstruct.find('.//tei:title[@level="j"]', self.ns)
            if venue_elem is None:
                venue_elem = biblstruct.find('.//tei:title[@level="m"]', self.ns)
            venue = venue_elem.text.strip() if venue_elem is not None and venue_elem.text else ""
            
            # Extract year
            year = None
            date_elem = biblstruct.find('.//tei:date[@type="published"]', self.ns)
            if date_elem is not None:
                when_attr = date_elem.get('when', '')
                if when_attr and len(when_attr) >= 4:
                    try:
                        year = int(when_attr[:4])
                    except ValueError:
                        pass
            
            # Extract DOI
            doi = None
            idno_elem = biblstruct.find('.//tei:idno[@type="DOI"]', self.ns)
            if idno_elem is not None and idno_elem.text:
                doi = idno_elem.text.strip()
            
            # Get raw text representation
            raw_text = ET.tostring(biblstruct, encoding='unicode', method='text').strip()
            raw_text = ' '.join(raw_text.split())  # Normalize whitespace
            
            # Only keep references with meaningful content
            if title or (authors and venue) or doi:
                return {
                    'ref_id': ref_id,
                    'doc_id': doc_id,
                    'title': title,
                    'authors': authors,
                    'venue': venue,
                    'year': year,
                    'doi': doi,
                    'raw_text': raw_text
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing biblStruct: {e}")
            return None
    
    def normalize_reference(self, ref: Dict) -> NormalizedReference:
        """Normalize a single reference via Crossref and OpenAlex"""
        
        ref_id = ref['ref_id']
        title = ref['title']
        authors = ref['authors']
        year = ref['year']
        original_doi = ref['doi']
        
        logger.debug(f"Normalizing reference: {ref_id}")
        
        # Initialize result
        normalized = NormalizedReference(
            ref_id=ref_id,
            raw_text=ref['raw_text'],
            title=title,
            authors=authors,
            year=year,
            venue=ref['venue'],
            doi=original_doi,
            crossref_doi=None,
            openalex_id=None,
            crossref_score=0.0,
            normalization_method='failed',
            canonical_title='',
            canonical_authors=[],
            canonical_venue='',
            citation_count=0,
            referenced_works=[],
            citing_works=[],
            concepts=[]
        )
        
        # Step 1: Try direct DOI lookup
        if original_doi:
            # Check cache first
            cache_key = f"doi_{original_doi}"
            if cache_key in self.doi_cache:
                crossref_data = self.doi_cache[cache_key]
                self.stats['cached_lookups'] += 1
            else:
                crossref_data = self.crossref.lookup_by_doi(original_doi)
                self.doi_cache[cache_key] = crossref_data
            
            if crossref_data:
                normalized.crossref_doi = crossref_data.get('DOI')
                normalized.crossref_score = 1.0
                normalized.normalization_method = 'direct_doi'
                self.stats['direct_doi_matches'] += 1
                
                # Update with Crossref canonical data
                self.update_with_crossref_data(normalized, crossref_data)
        
        # Step 2: Try Crossref metadata search if no DOI success
        if normalized.normalization_method == 'failed' and title:
            cache_key = f"search_{hashlib.md5(f'{title}_{authors}_{year}'.encode()).hexdigest()}"
            
            if cache_key in self.crossref_cache:
                search_result = self.crossref_cache[cache_key]
                self.stats['cached_lookups'] += 1
            else:
                search_result = self.crossref.search_by_metadata(title, authors, year)
                self.crossref_cache[cache_key] = search_result
            
            if search_result:
                crossref_data, score = search_result
                normalized.crossref_doi = crossref_data.get('DOI')
                normalized.crossref_score = score
                normalized.normalization_method = 'crossref_lookup'
                self.stats['crossref_lookups'] += 1
                
                # Update with Crossref canonical data
                self.update_with_crossref_data(normalized, crossref_data)
        
        # Step 3: Map to OpenAlex if we have a DOI
        final_doi = normalized.crossref_doi or normalized.doi
        if final_doi:
            openalex_data = self.openalex.lookup_by_doi(final_doi)
            if openalex_data:
                self.update_with_openalex_data(normalized, openalex_data)
                self.stats['openalex_mappings'] += 1
        
        # Update final statistics
        if normalized.normalization_method == 'failed':
            self.stats['failed_normalizations'] += 1
        
        return normalized
    
    def update_with_crossref_data(self, normalized: NormalizedReference, crossref_data: Dict):
        """Update normalized reference with Crossref canonical data"""
        
        # Canonical title
        titles = crossref_data.get('title', [])
        if titles:
            normalized.canonical_title = titles[0]
        
        # Canonical authors
        authors = crossref_data.get('author', [])
        normalized.canonical_authors = []
        for author in authors:
            given = author.get('given', '')
            family = author.get('family', '')
            if family:
                normalized.canonical_authors.append(f"{given} {family}".strip())
        
        # Canonical venue
        container_title = crossref_data.get('container-title', [])
        if container_title:
            normalized.canonical_venue = container_title[0]
        
        # Citation count (if available)
        normalized.citation_count = crossref_data.get('is-referenced-by-count', 0)
    
    def update_with_openalex_data(self, normalized: NormalizedReference, openalex_data: Dict):
        """Update normalized reference with OpenAlex data"""
        
        # OpenAlex ID
        normalized.openalex_id = openalex_data.get('id', '')
        
        # Enhanced citation count
        normalized.citation_count = openalex_data.get('cited_by_count', normalized.citation_count)
        
        # Referenced works (outgoing citations)
        normalized.referenced_works = openalex_data.get('referenced_works', [])
        
        # Citing works (incoming citations) - from cited_by_api_url if needed
        # Note: This would require additional API calls, so we'll leave empty for now
        
        # Concepts
        concepts = openalex_data.get('concepts', [])
        normalized.concepts = [concept.get('display_name', '') for concept in concepts if concept.get('score', 0) > 0.3]
        
        # Use OpenAlex canonical data if better than Crossref
        if not normalized.canonical_title and openalex_data.get('title'):
            normalized.canonical_title = openalex_data['title']
        
        if not normalized.canonical_venue:
            primary_location = openalex_data.get('primary_location', {})
            source = primary_location.get('source', {})
            if source.get('display_name'):
                normalized.canonical_venue = source['display_name']
    
    def process_all_tei_files(self) -> str:
        """Process all TEI files and normalize references"""
        
        tei_files = list(self.tei_dir.glob("*.tei.xml"))
        logger.info(f"📚 Processing {len(tei_files)} TEI files for reference extraction...")
        
        if not tei_files:
            logger.warning("No TEI files found for processing")
            return ""
        
        all_normalized_refs = []
        
        for i, tei_file in enumerate(tei_files):
            logger.info(f"📄 Processing {i+1}/{len(tei_files)}: {tei_file.name}")
            
            # Extract references from TEI
            references = self.extract_references_from_tei(tei_file)
            self.stats['total_references'] += len(references)
            
            # Normalize each reference
            for ref in references:
                try:
                    normalized_ref = self.normalize_reference(ref)
                    all_normalized_refs.append(normalized_ref)
                    
                    # Log progress every 10 references
                    if len(all_normalized_refs) % 10 == 0:
                        logger.debug(f"   Normalized {len(all_normalized_refs)} references...")
                        
                except Exception as e:
                    logger.error(f"Failed to normalize reference {ref.get('ref_id', '')}: {e}")
                    continue
        
        # Convert to DataFrame for saving
        ref_dicts = []
        for ref in all_normalized_refs:
            ref_dict = {
                'ref_id': ref.ref_id,
                'raw_text': ref.raw_text,
                'original_title': ref.title,
                'original_authors': json.dumps(ref.authors),
                'original_year': ref.year,
                'original_venue': ref.venue,
                'original_doi': ref.doi,
                'crossref_doi': ref.crossref_doi,
                'openalex_id': ref.openalex_id,
                'crossref_score': ref.crossref_score,
                'normalization_method': ref.normalization_method,
                'canonical_title': ref.canonical_title,
                'canonical_authors': json.dumps(ref.canonical_authors),
                'canonical_venue': ref.canonical_venue,
                'citation_count': ref.citation_count,
                'referenced_works': json.dumps(ref.referenced_works),
                'citing_works': json.dumps(ref.citing_works),
                'concepts': json.dumps(ref.concepts),
                'processing_timestamp': datetime.now().isoformat()
            }
            ref_dicts.append(ref_dict)
        
        # Save to Parquet
        output_file = self.normalized_refs_dir / "normalized_references.parquet"
        
        if ref_dicts:
            df = pd.DataFrame(ref_dicts)
            df.to_parquet(output_file, index=False)
            
            logger.info(f"💾 Saved {len(ref_dicts)} normalized references to {output_file}")
        else:
            logger.warning("No references were successfully normalized")
            return ""
        
        # Save statistics
        stats_file = self.normalized_refs_dir / "normalization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save caches for future runs
        cache_file = self.normalized_refs_dir / "lookup_cache.json"
        combined_cache = {
            'doi_cache': self.doi_cache,
            'crossref_cache': self.crossref_cache
        }
        with open(cache_file, 'w') as f:
            json.dump(combined_cache, f, indent=2, default=str)
        
        logger.info("="*60)
        logger.info("🎯 REFERENCE NORMALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Total references: {self.stats['total_references']}")
        logger.info(f"✅ Direct DOI matches: {self.stats['direct_doi_matches']}")
        logger.info(f"🔍 Crossref lookups: {self.stats['crossref_lookups']}")
        logger.info(f"🔗 OpenAlex mappings: {self.stats['openalex_mappings']}")
        logger.info(f"❌ Failed normalizations: {self.stats['failed_normalizations']}")
        logger.info(f"⚡ Cached lookups: {self.stats['cached_lookups']}")
        logger.info(f"💾 Output: {output_file}")
        logger.info("="*60)
        
        return str(output_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize references via Crossref and OpenAlex")
    parser.add_argument("--corpus-dir", default="/home/ubuntu/LW_scrape/multi_source_corpus")
    parser.add_argument("--load-cache", action="store_true", help="Load existing lookup cache")
    
    args = parser.parse_args()
    
    normalizer = ReferenceNormalizer(args.corpus_dir)
    
    # Load cache if requested
    if args.load_cache:
        cache_file = normalizer.normalized_refs_dir / "lookup_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                normalizer.doi_cache = cache_data.get('doi_cache', {})
                normalizer.crossref_cache = cache_data.get('crossref_cache', {})
            logger.info(f"📋 Loaded cache with {len(normalizer.doi_cache)} DOI and {len(normalizer.crossref_cache)} Crossref entries")
    
    # Process TEI files
    output_file = normalizer.process_all_tei_files()
    
    if output_file:
        logger.info(f"✅ Reference normalization completed: {output_file}")
    else:
        logger.error("❌ Reference normalization failed")

if __name__ == "__main__":
    main()