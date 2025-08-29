#!/usr/bin/env python3
"""
Corpus Normalizer - Bulletproof normalization and deduplication
Processes 1.1GB corpus into clean document/chunk stores
"""

import hashlib
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import xxhash
from datasketch import MinHashLSH, MinHash
import nltk
from langdetect import detect
from collections import defaultdict
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class URLCanonicalizer:
    """Robust URL canonicalization"""
    
    def __init__(self):
        self.domain_mappings = {
            # ArXiv variants
            'browse.arxiv.org': 'arxiv.org',
            'export.arxiv.org': 'arxiv.org',
            # Wikipedia variants  
            'en.m.wikipedia.org': 'en.wikipedia.org',
            # Common redirects
            'www.': '',
        }
    
    def canonicalize(self, url: str) -> str:
        """Canonicalize URL to standard form"""
        if not url or not isinstance(url, str):
            return ""
        
        # Remove fragments and normalize
        url = urldefrag(url)[0]
        
        # Parse URL components
        try:
            parsed = urlparse(url.lower().strip())
        except:
            return url
        
        # Normalize scheme
        scheme = parsed.scheme or 'https'
        if scheme not in ['http', 'https']:
            return url
        
        # Normalize domain
        domain = parsed.netloc
        for old, new in self.domain_mappings.items():
            if domain.startswith(old):
                domain = domain.replace(old, new, 1)
        
        # Remove common tracking parameters
        query_parts = []
        if parsed.query:
            for param in parsed.query.split('&'):
                if '=' not in param:
                    continue
                key = param.split('=')[0].lower()
                if key not in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 
                              'utm_content', 'fbclid', 'gclid', 'ref', 'source']:
                    query_parts.append(param)
        
        query = '&'.join(query_parts) if query_parts else ''
        
        # Rebuild canonical URL
        canonical = f"{scheme}://{domain}{parsed.path}"
        if query:
            canonical += f"?{query}"
            
        return canonical
    
    def compute_doc_id(self, canonical_url: str) -> str:
        """Compute SHA256 hash of canonical URL"""
        return hashlib.sha256(canonical_url.encode('utf-8')).hexdigest()

class DocumentMetadataExtractor:
    """Extract metadata from documents"""
    
    def __init__(self):
        self.domain_classes = {
            'en.wikipedia.org': 'encyclopedia',
            'arxiv.org': 'paper',
            'github.com': 'code',
            'medium.com': 'blog',
            'substack.com': 'blog',
            'openai.com': 'paper',
            'anthropic.com': 'paper',
            'intelligence.org': 'paper',
            'fhi.ox.ac.uk': 'paper',
            'bbc.com': 'news',
            'nytimes.com': 'news',
            'lesswrong.com': 'blog',
            'greaterwrong.com': 'blog'
        }
        
        self.domain_reputation = {
            'encyclopedia': 1.0,
            'paper': 0.95,
            'code': 0.8,
            'blog': 0.6,
            'news': 0.5,
            'social': 0.3
        }
    
    def extract_metadata(self, doc_data: dict) -> dict:
        """Extract comprehensive metadata from document"""
        metadata = {
            'original_url': doc_data.get('url', ''),
            'title': self._extract_title(doc_data),
            'authors': self._extract_authors(doc_data),
            'pub_date': self._extract_pub_date(doc_data),
            'language': self._detect_language(doc_data.get('content', '')),
            'content_length': len(doc_data.get('content', '')),
            'domain': urlparse(doc_data.get('url', '')).netloc.lower(),
            'fetch_time': self._parse_timestamp(doc_data.get('scraped_at', '')),
            'content_type': 'pdf' if doc_data.get('url', '').endswith('.pdf') else 'html'
        }
        
        # Add domain classification
        metadata['domain_class'] = self._classify_domain(metadata['domain'])
        metadata['domain_reputation'] = self.domain_reputation.get(
            metadata['domain_class'], 0.4
        )
        
        return metadata
    
    def _extract_title(self, doc_data: dict) -> str:
        """Extract document title"""
        title = doc_data.get('title', '').strip()
        if title and title != 'No title':
            return title[:500]  # Limit length
        
        # Fallback to URL-based title
        url = doc_data.get('url', '')
        if url:
            path = urlparse(url).path
            return path.split('/')[-1][:100]
        
        return 'Untitled'
    
    def _extract_authors(self, doc_data: dict) -> str:
        """Extract authors (mainly for PDFs)"""
        # For PDFs, check metadata
        if 'metadata' in doc_data:
            metadata = doc_data['metadata']
            if isinstance(metadata, dict):
                # Look for author fields
                for field in ['authors', 'author', 'creator']:
                    if field in metadata:
                        return str(metadata[field])[:200]
        
        return ''
    
    def _extract_pub_date(self, doc_data: dict) -> Optional[str]:
        """Extract publication date when available"""
        # For PDFs with metadata
        if 'metadata' in doc_data:
            metadata = doc_data['metadata']
            if isinstance(metadata, dict):
                for field in ['pub_date', 'creation_date', 'date']:
                    if field in metadata:
                        return str(metadata[field])[:10]  # YYYY-MM-DD
        
        # Try to extract from ArXiv URL
        url = doc_data.get('url', '')
        if 'arxiv.org' in url:
            # ArXiv URLs often contain date info: 2309.01933 = Sep 2023
            match = re.search(r'/(\d{4})\.(\d{5})', url)
            if match:
                year_month = match.group(1)
                if len(year_month) == 4 and year_month[:2] in ['23', '24', '25']:
                    year = f"20{year_month[:2]}"
                    month = year_month[2:4]
                    if 1 <= int(month) <= 12:
                        return f"{year}-{month:0>2}-01"
        
        return None
    
    def _detect_language(self, content: str) -> str:
        """Detect content language"""
        if not content or len(content) < 50:
            return 'en'  # Default
        
        try:
            # Sample first 1000 chars for speed
            sample = content[:1000]
            lang = detect(sample)
            return lang if lang else 'en'
        except:
            return 'en'
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """Parse scraped timestamp"""
        if not timestamp_str:
            return None
        try:
            # Handle ISO format
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.isoformat()
        except:
            return None
    
    def _classify_domain(self, domain: str) -> str:
        """Classify domain into content type"""
        for domain_pattern, class_name in self.domain_classes.items():
            if domain_pattern in domain:
                return class_name
        
        # Heuristic classification
        if any(x in domain for x in ['wiki', 'encyclopedia']):
            return 'encyclopedia'
        elif any(x in domain for x in ['arxiv', 'acm', 'ieee', 'springer', 'nature', 'science']):
            return 'paper'
        elif any(x in domain for x in ['github', 'gitlab', 'stackoverflow']):
            return 'code'
        elif any(x in domain for x in ['substack', 'medium', 'blog']):
            return 'blog'
        elif any(x in domain for x in ['news', 'times', 'bbc', 'cnn', 'guardian']):
            return 'news'
        elif any(x in domain for x in ['twitter', 'facebook', 'linkedin', 'reddit']):
            return 'social'
        else:
            return 'other'

class NearDuplicateDetector:
    """MinHash-based near-duplicate detection"""
    
    def __init__(self, threshold: float = 0.9, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.signatures = {}
    
    def compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash signature for text"""
        if not text:
            return MinHash(num_perm=self.num_perm)
        
        # Create shingles (3-grams of words)
        words = text.lower().split()
        shingles = []
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            shingles.append(shingle)
        
        if not shingles:
            shingles = [text.lower()]  # Fallback for short text
        
        # Compute MinHash
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf8'))
        
        return minhash
    
    def find_duplicates(self, documents: List[Tuple[str, str, dict]]) -> Dict[str, List[str]]:
        """Find near-duplicates in document collection"""
        duplicate_groups = defaultdict(list)
        processed_docs = {}
        
        for doc_id, content, metadata in documents:
            minhash = self.compute_minhash(content)
            
            # Check for existing duplicates
            duplicates = self.lsh.query(minhash)
            
            if duplicates:
                # Find the best representative
                existing_doc = duplicates[0]  # First match
                duplicate_groups[existing_doc].append(doc_id)
            else:
                # New unique document
                self.lsh.insert(doc_id, minhash)
                processed_docs[doc_id] = {
                    'minhash': minhash,
                    'metadata': metadata,
                    'content_length': len(content)
                }
        
        return duplicate_groups, processed_docs
    
    def choose_best_representative(self, doc_group: List[str], all_docs: dict) -> str:
        """Choose best document from duplicate group"""
        if len(doc_group) == 1:
            return doc_group[0]
        
        # Scoring criteria:
        # 1. Highest reference frequency
        # 2. Longest content
        # 3. Best domain reputation
        
        best_doc = doc_group[0]
        best_score = 0
        
        for doc_id in doc_group:
            if doc_id not in all_docs:
                continue
                
            doc = all_docs[doc_id]
            metadata = doc['metadata']
            
            score = (
                metadata.get('ref_frequency', 0) * 10 +  # Reference frequency (high weight)
                metadata.get('content_length', 0) / 1000 +           # Content length
                metadata.get('domain_reputation', 0) * 5  # Domain reputation
            )
            
            if score > best_score:
                best_score = score
                best_doc = doc_id
        
        return best_doc

class CorpusNormalizer:
    """Main corpus normalization pipeline"""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.canonicalizer = URLCanonicalizer()
        self.metadata_extractor = DocumentMetadataExtractor()
        self.duplicate_detector = NearDuplicateDetector()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'normalization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_corpus(self):
        """Main processing pipeline"""
        self.logger.info("Starting corpus normalization...")
        
        # Step 1: Load and canonicalize all documents
        self.logger.info("Step 1: Loading and canonicalizing documents...")
        documents = self.load_all_documents()
        self.logger.info(f"Loaded {len(documents)} documents")
        
        # Step 2: Extract metadata and compute signatures
        self.logger.info("Step 2: Extracting metadata and computing signatures...")
        processed_docs = self.process_documents(documents)
        self.logger.info(f"Processed {len(processed_docs)} documents")
        
        # Step 3: Near-duplicate detection
        self.logger.info("Step 3: Near-duplicate detection...")
        deduplicated_docs = self.deduplicate_documents(processed_docs)
        self.logger.info(f"After deduplication: {len(deduplicated_docs)} unique documents")
        
        # Step 4: Save document store
        self.logger.info("Step 4: Saving document store...")
        self.save_document_store(deduplicated_docs)
        
        # Step 5: Text chunking and chunk store
        self.logger.info("Step 5: Creating chunk store...")
        self.create_chunk_store(deduplicated_docs)
        
        self.logger.info("Corpus normalization completed!")
        
        return deduplicated_docs
    
    def load_all_documents(self) -> List[dict]:
        """Load all JSONL documents from crawler output"""
        documents = []
        
        # Load web content
        crawled_dir = self.input_dir / 'crawled'
        if crawled_dir.exists():
            for jsonl_file in crawled_dir.glob('worker_*.jsonl'):
                self.logger.info(f"Loading {jsonl_file}")
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            doc = json.loads(line)
                            if doc.get('status') == 'success' and doc.get('content'):
                                documents.append(doc)
                        except Exception as e:
                            self.logger.warning(f"Error parsing line {line_num} in {jsonl_file}: {e}")
        
        # Load PDF content
        pdf_dir = self.input_dir / 'pdfs'
        if pdf_dir.exists():
            for jsonl_file in pdf_dir.glob('*.jsonl'):
                self.logger.info(f"Loading {jsonl_file}")
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            doc = json.loads(line)
                            if doc.get('status') == 'success' and doc.get('content'):
                                documents.append(doc)
                        except Exception as e:
                            self.logger.warning(f"Error parsing line {line_num} in {jsonl_file}: {e}")
        
        return documents
    
    def process_documents(self, documents: List[dict]) -> dict:
        """Process documents: canonicalize, extract metadata, compute signatures"""
        processed = {}
        
        for doc in documents:
            try:
                # Canonicalize URL
                canonical_url = self.canonicalizer.canonicalize(doc.get('url', ''))
                if not canonical_url:
                    continue
                
                doc_id = self.canonicalizer.compute_doc_id(canonical_url)
                
                # Extract metadata
                metadata = self.metadata_extractor.extract_metadata(doc)
                metadata['canonical_url'] = canonical_url
                metadata['doc_id'] = doc_id
                
                # Compute MinHash signature
                content = doc.get('content', '')
                minhash = self.duplicate_detector.compute_minhash(content)
                metadata['minhash_signature'] = minhash.digest()
                
                # Store processed document
                processed[doc_id] = {
                    'content': content,
                    'metadata': metadata,
                    'minhash': minhash
                }
                
            except Exception as e:
                self.logger.warning(f"Error processing document {doc.get('url', 'unknown')}: {e}")
        
        return processed
    
    def deduplicate_documents(self, processed_docs: dict) -> dict:
        """Remove near-duplicates"""
        # Prepare data for duplicate detection
        documents_for_dedup = []
        for doc_id, doc_data in processed_docs.items():
            documents_for_dedup.append((
                doc_id,
                doc_data['content'],
                doc_data['metadata']
            ))
        
        # Find duplicates
        duplicate_groups, unique_docs = self.duplicate_detector.find_duplicates(documents_for_dedup)
        
        # Choose best representatives
        deduplicated = {}
        all_processed = {doc_id: data for doc_id, data in processed_docs.items()}
        
        for representative, duplicates in duplicate_groups.items():
            if representative not in all_processed:
                continue
                
            best_doc = self.duplicate_detector.choose_best_representative(
                [representative] + duplicates, all_processed
            )
            
            # Mark status
            doc_data = all_processed[best_doc]
            doc_data['metadata']['status'] = 'processed'
            doc_data['metadata']['duplicate_count'] = len(duplicates)
            
            deduplicated[best_doc] = doc_data
            
            # Log duplicate info
            if duplicates:
                self.logger.info(f"Kept {best_doc}, removed {len(duplicates)} duplicates")
        
        # Add unique documents (no duplicates found)
        for doc_id, doc_data in all_processed.items():
            if doc_id not in deduplicated:
                if doc_id in unique_docs:
                    doc_data['metadata']['status'] = 'processed'
                    doc_data['metadata']['duplicate_count'] = 0
                    deduplicated[doc_id] = doc_data
        
        return deduplicated
    
    def save_document_store(self, documents: dict):
        """Save document metadata to Parquet"""
        doc_records = []
        
        for doc_id, doc_data in documents.items():
            metadata = doc_data['metadata']
            
            record = {
                'doc_id': doc_id,
                'content': doc_data.get('content', ''),  # Include content for chunking
                'original_url': metadata.get('original_url', ''),
                'canonical_url': metadata.get('canonical_url', ''),
                'fetch_time': metadata.get('fetch_time'),
                'content_type': metadata.get('content_type', 'html'),
                'language': metadata.get('language', 'en'),
                'title': metadata.get('title', ''),
                'authors': metadata.get('authors', ''),
                'pub_date': metadata.get('pub_date'),
                'domain': metadata.get('domain', ''),
                'domain_class': metadata.get('domain_class', 'other'),
                'ref_frequency': metadata.get('ref_frequency', 1),
                'content_length': metadata.get('content_length', 0),
                'minhash_signature': metadata.get('minhash_signature', b''),
                'status': metadata.get('status', 'processed'),
                'domain_reputation': metadata.get('domain_reputation', 0.5),
                'duplicate_count': metadata.get('duplicate_count', 0)
            }
            
            doc_records.append(record)
        
        # Save to Parquet
        df = pd.DataFrame(doc_records)
        output_path = self.output_dir / 'document_store.parquet'
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        self.logger.info(f"Saved {len(doc_records)} documents to {output_path}")
        
        # Save summary stats
        stats = {
            'total_documents': len(doc_records),
            'content_types': df['content_type'].value_counts().to_dict(),
            'domain_classes': df['domain_class'].value_counts().to_dict(),
            'languages': df['language'].value_counts().to_dict(),
            'avg_content_length': df['content_length'].mean(),
            'total_content_size': df['content_length'].sum()
        }
        
        with open(self.output_dir / 'corpus_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def create_chunk_store(self, documents: dict):
        """Create chunk store with detailed metadata (placeholder)"""
        # This will be implemented in the next phase
        # For now, just log the plan
        self.logger.info("Chunk store creation planned for next phase")
        self.logger.info(f"Will process {len(documents)} documents into chunks")

def main():
    """Main execution function"""
    input_dir = Path("/home/ubuntu/LW_scrape/data")
    output_dir = Path("/home/ubuntu/LW_scrape/normalized_corpus")
    
    normalizer = CorpusNormalizer(input_dir, output_dir)
    result = normalizer.process_corpus()
    
    print(f"Normalization complete. Processed {len(result)} unique documents.")

if __name__ == '__main__':
    main()