#!/usr/bin/env python3
"""
Continue ArXiv Phases 2+3 - Extension of simple reprocessor
Process remaining papers from Phase 2 (2010-2019) and Phase 3 (pre-2010)
"""

import pandas as pd
import requests
import json
import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from datetime import datetime

# PDF processing
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinueArXivPhases:
    def __init__(self):
        self.output_dir = Path("/home/ubuntu/LW_scrape/arxiv_reprocessing")
        self.pdfs_dir = self.output_dir / 'pdfs'
        
        self.stats = {'phase2_downloaded': 0, 'phase3_downloaded': 0, 'failed': 0}
        self._lock = threading.Lock()
        
        # Setup file logging
        handler = logging.FileHandler(self.output_dir / 'phases23_processing.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
    def extract_remaining_papers(self):
        """Extract Phase 2 and 3 papers that weren't processed"""
        logger.info("Loading unified corpus for remaining phases...")
        df = pd.read_parquet('/home/ubuntu/LW_scrape/unified_corpus/unified_contextual_chunks_complete.parquet')
        
        # Get main corpus arXiv papers
        main_arxiv = df[(df['corpus_source'] == 'main') & (df['domain'] == 'arxiv.org')]
        
        # Group by document
        papers = main_arxiv.groupby('doc_id').agg({
            'original_url': 'first',
            'doc_title': 'first', 
            'authors': 'first',
            'pub_date': 'first'
        }).reset_index()
        
        # Extract arXiv IDs
        papers['arxiv_id'] = papers['original_url'].str.extract(r'([0-9]{4}\.[0-9]{4,5}v?[0-9]*)')
        
        # Try alternative extraction from title if URL failed
        no_id_mask = papers['arxiv_id'].isna()
        papers.loc[no_id_mask, 'arxiv_id'] = papers.loc[no_id_mask, 'doc_title'].str.extract(r'([0-9]{4}\.[0-9]{4,5})')
        
        papers = papers.dropna(subset=['arxiv_id'])
        
        # Add publication year for phase assignment
        papers['pub_year'] = pd.to_numeric(papers['pub_date'].str[:4], errors='coerce').fillna(1999)
        
        # Assign phases (Phase 1 was already processed)
        papers['phase'] = 3  # Default
        papers.loc[(papers['pub_year'] >= 2010) & (papers['pub_year'] < 2020), 'phase'] = 2
        # pub_year < 2010 stays as phase 3
        
        # Filter for Phase 2 and 3 only
        remaining_papers = papers[papers['phase'].isin([2, 3])].copy()
        remaining_papers = remaining_papers.sort_values('pub_year', ascending=False)  # Recent first
        
        logger.info(f"Found remaining papers to process:")
        logger.info(f"  Phase 2 (2010-19): {len(remaining_papers[remaining_papers['phase']==2])}")
        logger.info(f"  Phase 3 (pre-2010): {len(remaining_papers[remaining_papers['phase']==3])}")
        
        return remaining_papers
        
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF using available libraries"""
        text = ""
        
        if HAS_FITZ:
            try:
                doc = fitz.open(str(pdf_path))
                text_parts = []
                for page in doc[:50]:  # Max 50 pages
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                doc.close()
                text = '\n\n'.join(text_parts)
                return text
            except Exception as e:
                logger.debug(f"PyMuPDF failed: {e}")
        
        if HAS_PYPDF2 and not text:
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    for page in reader.pages[:50]:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                    text = '\n\n'.join(text_parts)
            except Exception as e:
                logger.debug(f"PyPDF2 failed: {e}")
                
        return text.strip()
    
    def download_paper(self, row):
        """Download single paper"""
        arxiv_id = row['arxiv_id']
        doc_id = row['doc_id']
        phase = row['phase']
        
        try:
            # Download PDF
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save PDF
            pdf_path = self.pdfs_dir / f"arxiv_{arxiv_id}.pdf"
            
            # Skip if already downloaded
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                logger.debug(f"Skipping {arxiv_id} - already exists")
                # Still extract text for processing
                text = self.extract_pdf_text(pdf_path)
                if len(text.strip()) < 100:
                    return None
            else:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                    
                # Extract text
                text = self.extract_pdf_text(pdf_path)
                
                if len(text.strip()) < 100:
                    logger.warning(f"Poor extraction for {arxiv_id}")
                    return None
            
            # Extract better title
            lines = text.split('\n')[:20]
            title = row['doc_title']
            for line in lines:
                line = line.strip()
                if (10 < len(line) < 200 and 
                    not line.lower().startswith('arxiv:') and
                    not re.match(r'^[0-9\.\s]+$', line) and
                    any(c.isalpha() for c in line)):
                    title = line
                    break
            
            # Extract authors
            authors = self._extract_authors(text, row['authors'])
            
            result = {
                'doc_id': doc_id,
                'arxiv_id': arxiv_id,
                'content': text,
                'title': title,
                'authors': authors,
                'pub_date': row['pub_date'],
                'original_url': url,
                'domain': 'arxiv.org',
                'extraction_chars': len(text),
                'phase': phase
            }
            
            with self._lock:
                if phase == 2:
                    self.stats['phase2_downloaded'] += 1
                else:
                    self.stats['phase3_downloaded'] += 1
                
            logger.info(f"✅ Phase {phase} - {arxiv_id}: {len(text):,} chars")
            return result
            
        except Exception as e:
            with self._lock:
                self.stats['failed'] += 1
            logger.error(f"❌ Phase {phase} - {arxiv_id}: {e}")
            return None
    
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
    
    def create_chunks(self, extraction):
        """Create simple chunks from text"""
        content = extraction['content']
        doc_id = extraction['doc_id']
        
        # Simple paragraph-based chunking
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        prev_context = ""
        
        for para in paragraphs:
            para_tokens = len(para.split())
            
            if current_tokens + para_tokens > 1200 and current_chunk:
                if current_tokens >= 700:
                    # Create chunk
                    chunk = self.format_chunk(
                        doc_id, chunk_index, current_chunk, 
                        prev_context, current_tokens, extraction
                    )
                    chunks.append(chunk)
                    
                    # Update prev context
                    sentences = re.split(r'[.!?]+', current_chunk)
                    good_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                    prev_context = '. '.join(good_sentences[-2:]).strip()[:300]
                    
                    chunk_index += 1
                
                current_chunk = para
                current_tokens = para_tokens
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_tokens += para_tokens
        
        # Last chunk
        if current_chunk and current_tokens >= 700:
            chunk = self.format_chunk(
                doc_id, chunk_index, current_chunk, 
                prev_context, current_tokens, extraction
            )
            chunks.append(chunk)
            
        return chunks
    
    def format_chunk(self, doc_id, chunk_index, content, prev_context, token_count, extraction):
        """Format chunk in schema-compatible way"""
        first_line = content.split('\n')[0].strip()
        section_path = first_line[:50] if len(first_line) > 10 else "Document Section"
        
        sentences = re.split(r'[.!?]+', content)
        summary_header = sentences[0].strip()[:150] if sentences else "Research content"
        
        return {
            'chunk_id': f"{doc_id}_reprocessed_{chunk_index:03d}",
            'doc_id': doc_id,
            'content': content,
            'prev_context': prev_context,
            'section_path': section_path,
            'summary_header': summary_header,
            'token_count': token_count,
            'chunk_index': chunk_index,
            'page_refs': "",
            'content_type': 'pdf',
            'structure_type': 'paragraph',
            'heading_level': None,
            'doc_title': extraction['title'],
            'domain': extraction['domain'],
            'original_url': extraction['original_url'],
            'authors': extraction['authors'],
            'pub_date': extraction['pub_date']
        }
    
    def process_phase(self, papers, phase_num):
        """Process papers for given phase"""
        phase_papers = papers[papers.phase == phase_num].copy()
        
        if len(phase_papers) == 0:
            logger.info(f"No papers to process for Phase {phase_num}")
            return []
            
        logger.info(f"Starting Phase {phase_num}: {len(phase_papers)} papers")
        
        successful_extractions = []
        all_chunks = []
        
        # Download papers in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_paper = {
                executor.submit(self.download_paper, row): row 
                for _, row in phase_papers.iterrows()
            }
            
            for future in future_to_paper:
                try:
                    result = future.result()
                    if result:
                        successful_extractions.append(result)
                        
                        # Create chunks
                        chunks = self.create_chunks(result)
                        all_chunks.extend(chunks)
                        
                        if len(successful_extractions) % 25 == 0:
                            logger.info(f"Phase {phase_num} progress: {len(successful_extractions)}/{len(phase_papers)}")
                            
                except Exception as e:
                    logger.error(f"Processing error: {e}")
        
        logger.info(f"Phase {phase_num} complete: {len(successful_extractions)} papers, {len(all_chunks)} chunks")
        return all_chunks
        
    def save_chunks(self, chunks, phase_num):
        """Save chunks to parquet"""
        if not chunks:
            return
            
        df = pd.DataFrame(chunks)
        df['corpus_source'] = 'main_reprocessed'
        
        # Save phase results
        filename = f"arxiv_reprocessed_phase{phase_num}.parquet"
        df.to_parquet(self.output_dir / filename, index=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {filename}")
        
        # Save stats
        stats = {
            'phase': phase_num,
            'chunks': len(chunks),
            'papers': df['doc_id'].nunique(),
            'avg_tokens': df['token_count'].mean(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / f'phase{phase_num}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    processor = ContinueArXivPhases()
    
    logger.info("🚀 Starting Phase 2 & 3 ArXiv Processing...")
    
    # Extract remaining papers
    papers = processor.extract_remaining_papers()
    
    if len(papers) == 0:
        logger.info("No remaining papers to process")
        return
    
    all_chunks = []
    
    # Process Phase 2 (2010-2019)
    phase2_chunks = processor.process_phase(papers, 2)
    if phase2_chunks:
        processor.save_chunks(phase2_chunks, 2)
        all_chunks.extend(phase2_chunks)
    
    # Brief pause
    time.sleep(5)
    
    # Process Phase 3 (pre-2010)  
    phase3_chunks = processor.process_phase(papers, 3)
    if phase3_chunks:
        processor.save_chunks(phase3_chunks, 3)
        all_chunks.extend(phase3_chunks)
    
    # Final summary
    logger.info("✅ Phase 2 & 3 Processing Complete!")
    logger.info(f"Phase 2 papers: {processor.stats['phase2_downloaded']}")
    logger.info(f"Phase 3 papers: {processor.stats['phase3_downloaded']}") 
    logger.info(f"Total new chunks: {len(all_chunks)}")
    logger.info(f"Failed downloads: {processor.stats['failed']}")

if __name__ == '__main__':
    main()