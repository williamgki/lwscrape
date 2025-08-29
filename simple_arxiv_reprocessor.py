#!/usr/bin/env python3
"""
Simple ArXiv Reprocessor - Clean implementation without complex dependencies
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

class SimpleArXivReprocessor:
    def __init__(self):
        self.output_dir = Path("/home/ubuntu/LW_scrape/arxiv_reprocessing")
        self.output_dir.mkdir(exist_ok=True)
        self.pdfs_dir = self.output_dir / 'pdfs'
        self.pdfs_dir.mkdir(exist_ok=True)
        
        self.stats = {'papers_identified': 0, 'downloaded': 0, 'failed': 0}
        self._lock = threading.Lock()
        
        # Setup file logging
        handler = logging.FileHandler(self.output_dir / 'processing.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
    def extract_papers(self):
        """Extract main corpus arXiv papers"""
        logger.info("Loading unified corpus...")
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
        papers['arxiv_id'] = papers['original_url'].str.extract(r'([0-9]{4}\.[0-9]{4,5})')
        papers = papers.dropna(subset=['arxiv_id'])
        
        # Add publication year for prioritization
        papers['pub_year'] = pd.to_numeric(papers['pub_date'].str[:4], errors='coerce').fillna(1999)
        papers = papers.sort_values('pub_year', ascending=False)  # Recent first
        
        # Priority phases
        papers['phase'] = 1
        papers.loc[papers['pub_year'] >= 2020, 'phase'] = 1  # High priority
        papers.loc[(papers['pub_year'] >= 2010) & (papers['pub_year'] < 2020), 'phase'] = 2
        papers.loc[papers['pub_year'] < 2010, 'phase'] = 3
        
        self.stats['papers_identified'] = len(papers)
        
        logger.info(f"Found {len(papers)} arXiv papers to reprocess:")
        logger.info(f"  Phase 1 (2020+): {len(papers[papers.phase==1])}")
        logger.info(f"  Phase 2 (2010-19): {len(papers[papers.phase==2])}")
        logger.info(f"  Phase 3 (pre-2010): {len(papers[papers.phase==3])}")
        
        return papers
        
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
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
        
        try:
            # Download PDF
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save PDF
            pdf_path = self.pdfs_dir / f"arxiv_{arxiv_id}.pdf"
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
                    any(c.isalpha() for c in line)):
                    title = line
                    break
            
            result = {
                'doc_id': doc_id,
                'arxiv_id': arxiv_id,
                'content': text,
                'title': title,
                'authors': row['authors'],
                'pub_date': row['pub_date'],
                'original_url': url,
                'domain': 'arxiv.org',
                'extraction_chars': len(text)
            }
            
            with self._lock:
                self.stats['downloaded'] += 1
                
            logger.info(f"✅ {arxiv_id}: {len(text):,} chars")
            return result
            
        except Exception as e:
            with self._lock:
                self.stats['failed'] += 1
            logger.error(f"❌ {arxiv_id}: {e}")
            return None
    
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
        logger.info(f"Starting Phase {phase_num}: {len(phase_papers)} papers")
        
        successful_extractions = []
        all_chunks = []
        
        # Download papers in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
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
                        
                        if len(successful_extractions) % 10 == 0:
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
    processor = SimpleArXivReprocessor()
    
    logger.info("🚀 Starting ArXiv reprocessing...")
    
    # Extract papers
    papers = processor.extract_papers()
    
    # Process Phase 1 (recent papers)
    phase1_chunks = processor.process_phase(papers, 1)
    processor.save_chunks(phase1_chunks, 1)
    
    # If successful, continue with Phase 2
    if len(phase1_chunks) > 50:
        logger.info("Phase 1 successful, starting Phase 2...")
        phase2_chunks = processor.process_phase(papers, 2)
        processor.save_chunks(phase2_chunks, 2)
    
    logger.info("✅ ArXiv reprocessing completed!")
    logger.info(f"Final stats: {processor.stats}")

if __name__ == '__main__':
    main()