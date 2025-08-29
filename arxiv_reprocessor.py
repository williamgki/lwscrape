#!/usr/bin/env python3
"""
ArXiv Reprocessor - Upgrade Main Corpus ArXiv Papers
Reprocess main corpus arXiv papers with improved PDF extraction and Wilson Lin chunking
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import re
import json
from datetime import datetime
import requests
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# Import PDF processing libraries
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# Import existing processing components  
import sys
sys.path.append('/home/ubuntu/LW_scrape')

import tiktoken
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArXivReprocessor:
    """Reprocess main corpus arXiv papers with improved extraction"""
    
    def __init__(self, 
                 unified_corpus_path: str,
                 output_dir: str,
                 max_workers: int = 12):
        
        self.unified_corpus_path = Path(unified_corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Create subdirectories
        self.pdfs_dir = self.output_dir / 'pdfs'
        self.pdfs_dir.mkdir(exist_ok=True)
        self.chunks_dir = self.output_dir / 'reprocessed_chunks'
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / 'arxiv_reprocessing.log'
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        # Stats tracking
        self._lock = threading.Lock()
        self.stats = {
            'papers_identified': 0,
            'papers_downloaded': 0,
            'papers_processed': 0,
            'papers_chunked': 0,
            'chunks_created': 0,
            'download_failures': 0,
            'processing_failures': 0
        }
        
    def extract_main_arxiv_papers(self) -> pd.DataFrame:
        """Extract arXiv papers from main corpus for reprocessing"""
        logger.info("Extracting main corpus arXiv papers...")
        
        # Load unified corpus
        df = pd.read_parquet(self.unified_corpus_path)
        
        # Filter for main corpus arXiv papers
        main_arxiv = df[
            (df['corpus_source'] == 'main') & 
            (df['domain'] == 'arxiv.org')
        ].copy()
        
        logger.info(f"Found {len(main_arxiv)} arXiv chunks from main corpus")
        
        # Group by document to get unique papers
        papers = main_arxiv.groupby('doc_id').agg({
            'original_url': 'first',
            'doc_title': 'first', 
            'authors': 'first',
            'pub_date': 'first',
            'chunk_id': 'count'  # Count chunks per paper
        }).rename(columns={'chunk_id': 'chunk_count'})
        
        papers = papers.reset_index()
        
        # Extract arXiv ID from URLs and titles
        papers['arxiv_id'] = papers['original_url'].str.extract(r'arxiv\.org/(?:abs/|pdf/)?([0-9]{4}\.[0-9]{4,5}v?[0-9]*)')
        
        # Try to extract from title if URL extraction failed
        no_id_mask = papers['arxiv_id'].isna()
        papers.loc[no_id_mask, 'arxiv_id'] = papers.loc[no_id_mask, 'doc_title'].str.extract(r'([0-9]{4}\.[0-9]{4,5})')
        
        # Prioritize by recency (parse pub_date)
        papers['pub_year'] = pd.to_numeric(papers['pub_date'].str[:4], errors='coerce')
        papers['priority'] = papers['pub_year'].fillna(1999)  # Old papers get low priority
        
        # Sort by priority (recent first)
        papers = papers.sort_values(['priority', 'chunk_count'], ascending=[False, False])
        
        # Create processing phases
        papers['phase'] = 1  # Default
        papers.loc[papers['priority'] >= 2020, 'phase'] = 1  # High priority
        papers.loc[(papers['priority'] >= 2010) & (papers['priority'] < 2020), 'phase'] = 2  # Medium
        papers.loc[papers['priority'] < 2010, 'phase'] = 3  # Lower priority
        
        self.stats['papers_identified'] = len(papers)
        
        logger.info(f"Identified {len(papers)} unique arXiv papers for reprocessing:")
        logger.info(f"  Phase 1 (2020+): {len(papers[papers['phase'] == 1])} papers")
        logger.info(f"  Phase 2 (2010-2019): {len(papers[papers['phase'] == 2])} papers") 
        logger.info(f"  Phase 3 (pre-2010): {len(papers[papers['phase'] == 3])} papers")
        
        # Save extraction results
        papers.to_json(self.output_dir / 'main_arxiv_papers.json', orient='records', indent=2)
        
        return papers
    
    def build_download_queue(self, papers: pd.DataFrame) -> List[Dict]:
        """Build prioritized download queue"""
        logger.info("Building download queue...")
        
        download_queue = []
        
        for _, paper in papers.iterrows():
            arxiv_id = paper['arxiv_id']
            if pd.isna(arxiv_id):
                continue
                
            # Build download entry
            entry = {
                'doc_id': paper['doc_id'],
                'arxiv_id': arxiv_id,
                'original_url': paper['original_url'],
                'title': paper['doc_title'],
                'authors': paper['authors'],
                'pub_date': paper['pub_date'],
                'priority': paper['priority'],
                'phase': paper['phase'],
                'chunk_count': paper['chunk_count'],
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            }
            
            download_queue.append(entry)
        
        logger.info(f"Built download queue with {len(download_queue)} papers")
        
        # Save download queue
        with open(self.output_dir / 'download_queue.json', 'w') as f:
            json.dump(download_queue, f, indent=2, default=str)
            
        return download_queue
    
    def download_single_paper(self, entry: Dict) -> Optional[Dict]:
        """Download and extract single arXiv paper"""
        arxiv_id = entry['arxiv_id']
        pdf_url = entry['pdf_url']
        
        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Save PDF
            pdf_path = self.pdfs_dir / f"arxiv_{arxiv_id}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            # Extract text using improved PDF extraction
            text = self._extract_pdf_text(pdf_path)
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Poor extraction for {arxiv_id}: {len(text)} characters")
                return None
            
            # Build result
            result = {
                'doc_id': entry['doc_id'],
                'arxiv_id': arxiv_id,
                'pdf_path': str(pdf_path),
                'content': text,
                'title': self._extract_title(text, entry['title']),
                'authors': self._extract_authors(text, entry['authors']),
                'pub_date': entry['pub_date'],
                'original_url': pdf_url,
                'domain': 'arxiv.org',
                'content_type': 'pdf',
                'extraction_tokens': len(text.split())
            }
            
            with self._lock:
                self.stats['papers_downloaded'] += 1
            
            logger.info(f"✅ Downloaded {arxiv_id}: {len(text):,} chars, {result['extraction_tokens']:,} tokens")
            return result
            
        except Exception as e:
            with self._lock:
                self.stats['download_failures'] += 1
            logger.error(f"❌ Download failed for {arxiv_id}: {e}")
            return None
    
    def _extract_title(self, content: str, fallback_title: str) -> str:
        """Extract proper title from PDF content"""
        lines = content.split('\n')[:20]  # First 20 lines
        
        for line in lines:
            line = line.strip()
            # Look for title-like lines (not too short, not too long, contains letters)
            if (10 < len(line) < 200 and 
                not line.lower().startswith('arxiv:') and
                not re.match(r'^[0-9\.\s]+$', line) and
                any(c.isalpha() for c in line)):
                return line
        
        # Fallback to original title if available and not a filename
        if fallback_title and not fallback_title.endswith('.pdf'):
            return fallback_title
            
        return f"ArXiv Paper (ID not extracted)"
    
    def _extract_authors(self, content: str, fallback_authors: str) -> str:
        """Extract authors from PDF content"""
        lines = content.split('\n')[:30]  # First 30 lines
        
        # Look for author patterns
        for i, line in enumerate(lines):
            line = line.strip()
            # Common author indicators
            if any(indicator in line.lower() for indicator in ['author', 'by ']):
                # Check next few lines for author names
                for j in range(i, min(i+5, len(lines))):
                    potential_authors = lines[j].strip()
                    if (5 < len(potential_authors) < 100 and
                        re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', potential_authors)):
                        return potential_authors
        
        # Fallback
        if fallback_authors and fallback_authors.strip():
            return fallback_authors
        
        return "Authors not extracted"
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using available libraries"""
        text = ""
        
        # Method 1: Try PyMuPDF (fitz) first - better quality
        if HAS_FITZ:
            try:
                doc = fitz.open(str(pdf_path))
                text_parts = []
                for page_num in range(min(len(doc), 50)):  # Limit to 50 pages
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                doc.close()
                if text_parts:
                    text = '\n\n'.join(text_parts)
                    if len(text.strip()) > 100:
                        return text
            except Exception as e:
                logger.debug(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
        
        # Method 2: Fall back to PyPDF2
        if HAS_PYPDF2 and not text:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    for page_num in range(min(len(pdf_reader.pages), 50)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                    if text_parts:
                        text = '\n\n'.join(text_parts)
            except Exception as e:
                logger.debug(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
        
        return text.strip() if text else ""
    
    def download_papers_parallel(self, download_queue: List[Dict], phase: int = 1) -> List[Dict]:
        """Download papers in parallel for specified phase"""
        phase_papers = [entry for entry in download_queue if entry['phase'] == phase]
        
        logger.info(f"Starting Phase {phase} downloads: {len(phase_papers)} papers")
        
        successful_extractions = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all downloads
            future_to_paper = {
                executor.submit(self.download_single_paper, entry): entry 
                for entry in phase_papers
            }
            
            # Collect results
            for future in future_to_paper:
                try:
                    result = future.result()
                    if result:
                        successful_extractions.append(result)
                        
                        # Progress logging
                        if len(successful_extractions) % 25 == 0:
                            logger.info(f"Phase {phase} progress: {len(successful_extractions)}/{len(phase_papers)} completed")
                            
                except Exception as e:
                    paper = future_to_paper[future]
                    logger.error(f"Download error for {paper.get('arxiv_id', 'unknown')}: {e}")
        
        logger.info(f"Phase {phase} completed: {len(successful_extractions)}/{len(phase_papers)} successful downloads")
        return successful_extractions
    
    def chunk_extracted_papers(self, extractions: List[Dict]) -> List[Dict]:
        """Apply simplified chunking to extracted papers"""
        logger.info(f"Starting simplified chunking for {len(extractions)} papers...")
        
        all_chunks = []
        
        for i, extraction in enumerate(extractions):
            try:
                # Create chunks using simple paragraph-based splitting
                doc_chunks = self._create_simple_chunks(extraction)
                all_chunks.extend(doc_chunks)
                
                with self._lock:
                    self.stats['papers_chunked'] += 1
                    self.stats['chunks_created'] += len(doc_chunks)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Chunked {i+1}/{len(extractions)} papers, {len(all_chunks)} total chunks")
                    
            except Exception as e:
                with self._lock:
                    self.stats['processing_failures'] += 1
                logger.error(f"Chunking failed for {extraction['doc_id']}: {e}")
        
        logger.info(f"Chunking completed: {len(all_chunks)} chunks from {len(extractions)} papers")
        return all_chunks
    
    def _create_simple_chunks(self, extraction: Dict) -> List[Dict]:
        """Create simple chunks from extracted paper"""
        content = extraction['content']
        doc_id = extraction['doc_id']
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        prev_chunk_text = ""
        
        for paragraph in paragraphs:
            para_tokens = len(paragraph.split())
            
            # Check if adding this paragraph exceeds max tokens (1200)
            if current_tokens + para_tokens > 1200 and current_chunk:
                # Save current chunk if it meets minimum (700)
                if current_tokens >= 700:
                    chunk_dict = self._format_chunk(
                        doc_id, chunk_index, current_chunk, 
                        prev_chunk_text, current_tokens, extraction
                    )
                    chunks.append(chunk_dict)
                    
                    # Update previous context for next chunk
                    sentences = re.split(r'[.!?]+', current_chunk)
                    good_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                    prev_chunk_text = '. '.join(good_sentences[-2:]).strip()
                    
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = paragraph
                current_tokens = para_tokens
            else:
                # Add to current chunk
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
                current_tokens += para_tokens
        
        # Don't forget the last chunk
        if current_chunk and current_tokens >= 700:
            chunk_dict = self._format_chunk(
                doc_id, chunk_index, current_chunk, 
                prev_chunk_text, current_tokens, extraction
            )
            chunks.append(chunk_dict)
        
        return chunks
    
    def _format_chunk(self, doc_id: str, chunk_index: int, content: str, 
                     prev_context: str, token_count: int, extraction: Dict) -> Dict:
        """Format chunk in compatible schema"""
        chunk_id = f"{doc_id}_reprocessed_chunk_{chunk_index:03d}"
        
        # Simple section path
        first_line = content.split('\n')[0].strip()
        section_path = first_line[:50] if len(first_line) > 10 else "Document"
        
        # Simple summary header (first sentence)
        sentences = re.split(r'[.!?]+', content)
        summary_header = sentences[0].strip()[:150] if sentences else "Content section"
        
        return {
            'chunk_id': chunk_id,
            'doc_id': doc_id,
            'content': content,
            'prev_context': prev_context[:300],
            'section_path': section_path,
            'summary_header': summary_header,
            'token_count': token_count,
            'chunk_index': chunk_index,
            'page_refs': "",  # Simple version doesn't track pages
            'content_type': 'pdf',
            'structure_type': 'paragraph',
            'heading_level': None,
            'doc_title': extraction['title'],
            'domain': extraction['domain'],
            'original_url': extraction['original_url'],
            'authors': extraction['authors'],
            'pub_date': extraction['pub_date']
        }
    
    def save_reprocessed_chunks(self, chunks: List[Dict]) -> str:
        """Save reprocessed chunks to parquet files"""
        logger.info(f"Saving {len(chunks)} reprocessed chunks...")
        
        # Convert to DataFrame
        df = pd.DataFrame(chunks)
        
        # Add corpus source tag
        df['corpus_source'] = 'main_reprocessed'
        
        # Save in batches
        batch_size = 1000
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_file = self.chunks_dir / f"main_arxiv_reprocessed_batch_{batch_idx:04d}.parquet"
            batch_df.to_parquet(batch_file, index=False)
        
        # Save complete file
        complete_file = self.chunks_dir / "main_arxiv_reprocessed_complete.parquet"
        df.to_parquet(complete_file, index=False)
        
        # Save statistics
        stats = {
            'reprocessing_stats': self.stats,
            'total_chunks': len(chunks),
            'total_batches': num_batches,
            'avg_tokens_per_chunk': df['token_count'].mean(),
            'content_types': df['content_type'].value_counts().to_dict(),
            'domains': df['domain'].value_counts().to_dict(),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(self.chunks_dir / 'reprocessing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"✅ Saved reprocessed chunks: {num_batches} batches + complete file")
        return str(complete_file)

def main():
    """Main reprocessing execution"""
    
    reprocessor = ArXivReprocessor(
        unified_corpus_path="/home/ubuntu/LW_scrape/unified_corpus/unified_contextual_chunks_complete.parquet",
        output_dir="/home/ubuntu/LW_scrape/arxiv_reprocessing",
        max_workers=16
    )
    
    logger.info("🚀 Starting ArXiv reprocessing pipeline...")
    
    try:
        # Phase 1: Extract papers from main corpus
        papers = reprocessor.extract_main_arxiv_papers()
        
        # Phase 2: Build download queue
        download_queue = reprocessor.build_download_queue(papers)
        
        # Phase 3: Download papers (start with high priority)
        logger.info("Starting Phase 1 downloads (2020+ papers)...")
        phase1_extractions = reprocessor.download_papers_parallel(download_queue, phase=1)
        
        # Phase 4: Chunk extracted papers
        if phase1_extractions:
            chunks = reprocessor.chunk_extracted_papers(phase1_extractions)
            
            # Phase 5: Save results
            complete_file = reprocessor.save_reprocessed_chunks(chunks)
            
            # Final report
            logger.info("\n" + "="*60)
            logger.info("🎉 PHASE 1 ARXIV REPROCESSING COMPLETED")
            logger.info("="*60)
            logger.info(f"Papers processed: {reprocessor.stats['papers_chunked']:,}")
            logger.info(f"Chunks created: {reprocessor.stats['chunks_created']:,}")
            logger.info(f"Download success rate: {reprocessor.stats['papers_downloaded']/(reprocessor.stats['papers_downloaded']+reprocessor.stats['download_failures'])*100:.1f}%")
            logger.info(f"Saved to: {complete_file}")
            
            # Continue with Phase 2 if successful
            if reprocessor.stats['papers_chunked'] > 50:  # If Phase 1 was successful
                logger.info("\n🚀 Starting Phase 2 downloads (2010-2019 papers)...")
                phase2_extractions = reprocessor.download_papers_parallel(download_queue, phase=2)
                
                if phase2_extractions:
                    phase2_chunks = reprocessor.chunk_extracted_papers(phase2_extractions)
                    chunks.extend(phase2_chunks)
                    reprocessor.save_reprocessed_chunks(chunks)
                    
                    logger.info(f"✅ Phase 2 completed: +{len(phase2_chunks)} chunks")
        
    except Exception as e:
        logger.error(f"❌ Reprocessing failed: {e}")
        raise

if __name__ == '__main__':
    main()