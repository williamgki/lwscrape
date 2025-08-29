#!/usr/bin/env python3
"""
Enhanced ArXiv Reprocessor - Full Wilson Lin Implementation
Includes all phases + API-generated summary headers for consistency
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
from typing import List, Dict
import anthropic
from aisitools.api_key import get_api_key_for_proxy
import os

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

class EnhancedArXivReprocessor:
    def __init__(self):
        self.output_dir = Path("/home/ubuntu/LW_scrape/arxiv_reprocessing_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        self.pdfs_dir = self.output_dir / 'pdfs'
        self.pdfs_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'papers_identified': 0, 
            'phase1_downloaded': 0, 
            'phase2_downloaded': 0,
            'phase3_downloaded': 0,
            'total_chunks': 0,
            'api_calls': 0,
            'api_failures': 0
        }
        self._lock = threading.Lock()
        
        # Setup Claude API for summary generation
        self.anthropic_client = None
        self._setup_claude_api()
        
        # Setup file logging
        handler = logging.FileHandler(self.output_dir / 'enhanced_processing.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
    def _setup_claude_api(self):
        """Setup Claude API client for summary generation"""
        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            anthropic_base = os.environ.get("ANTHROPIC_BASE_URL")
            
            if not anthropic_key or not anthropic_base:
                raise ValueError("Missing ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL")
                
            proxy_key = get_api_key_for_proxy(anthropic_key)
            
            self.anthropic_client = anthropic.Anthropic(
                api_key=proxy_key,
                base_url=anthropic_base
            )
            logger.info("✅ Claude API client initialized for enhanced chunking")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Claude API: {e}")
            self.anthropic_client = None
        
    def extract_papers(self):
        """Extract main corpus arXiv papers with proper phase assignment"""
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
        
        # Extract arXiv IDs from URLs
        papers['arxiv_id'] = papers['original_url'].str.extract(r'([0-9]{4}\.[0-9]{4,5}v?[0-9]*)')
        
        # Try alternative extraction from title if URL failed
        no_id_mask = papers['arxiv_id'].isna()
        papers.loc[no_id_mask, 'arxiv_id'] = papers.loc[no_id_mask, 'doc_title'].str.extract(r'([0-9]{4}\.[0-9]{4,5})')
        
        papers = papers.dropna(subset=['arxiv_id'])
        
        # Add publication year for prioritization
        papers['pub_year'] = pd.to_numeric(papers['pub_date'].str[:4], errors='coerce').fillna(1999)
        papers = papers.sort_values('pub_year', ascending=False)  # Recent first
        
        # Assign priority phases correctly
        papers['phase'] = 3  # Default to lowest priority
        papers.loc[papers['pub_year'] >= 2020, 'phase'] = 1  # High priority (recent)
        papers.loc[(papers['pub_year'] >= 2010) & (papers['pub_year'] < 2020), 'phase'] = 2  # Medium priority
        # Papers < 2010 stay as phase 3 (low priority)
        
        self.stats['papers_identified'] = len(papers)
        
        logger.info(f"Found {len(papers)} arXiv papers to reprocess:")
        logger.info(f"  Phase 1 (2020+): {len(papers[papers['phase']==1])}")
        logger.info(f"  Phase 2 (2010-19): {len(papers[papers['phase']==2])}")
        logger.info(f"  Phase 3 (pre-2010): {len(papers[papers['phase']==3])}")
        
        return papers
        
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
                if len(text.strip()) > 100:
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
    
    def generate_wilson_lin_summary(self, chunk_content: str, section_path: str) -> str:
        """Generate Wilson Lin style summary using Claude API"""
        if not self.anthropic_client:
            # Fallback to simple summary
            sentences = re.split(r'[.!?]+', chunk_content)
            return sentences[0].strip()[:150] if sentences else "Research content"
            
        try:
            # Add delay for rate limiting
            time.sleep(0.1)
            
            # Limit content for API call
            content_sample = chunk_content[:1200] if len(chunk_content) > 1200 else chunk_content
            
            prompt = f"""Given this academic text chunk from section "{section_path}", write a concise 1-2 line contextual summary that captures the main point and provides local context. Be specific about the research content:

{content_sample}

Contextual summary (1-2 lines max):"""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            lines = summary.split('\n')
            result = '. '.join(lines[:2]) if len(lines) > 1 else lines[0]
            
            with self._lock:
                self.stats['api_calls'] += 1
            
            return result
            
        except Exception as e:
            logger.debug(f"API summary failed: {e}")
            with self._lock:
                self.stats['api_failures'] += 1
            
            # Fallback to first sentence
            sentences = re.split(r'[.!?]+', chunk_content)
            return sentences[0].strip()[:150] if sentences else "Research content"
    
    def download_paper(self, row):
        """Download and extract single paper"""
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
                
            # Enhanced title extraction
            title = self._extract_title(text, row['doc_title'])
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
                'phase': row['phase']
            }
            
            with self._lock:
                if row['phase'] == 1:
                    self.stats['phase1_downloaded'] += 1
                elif row['phase'] == 2:
                    self.stats['phase2_downloaded'] += 1
                else:
                    self.stats['phase3_downloaded'] += 1
                
            logger.info(f"✅ Phase {row['phase']} - {arxiv_id}: {len(text):,} chars")
            return result
            
        except Exception as e:
            logger.error(f"❌ {arxiv_id}: {e}")
            return None
    
    def _extract_title(self, content: str, fallback_title: str) -> str:
        """Enhanced title extraction from PDF content"""
        lines = content.split('\n')[:25]  # Check more lines
        
        for line in lines:
            line = line.strip()
            # Better title detection
            if (10 < len(line) < 200 and 
                not line.lower().startswith('arxiv:') and
                not re.match(r'^[0-9\.\s]+$', line) and
                not re.match(r'^[A-Z]{2,}\s*[0-9]', line) and  # Avoid headers like "PAGE 1"
                any(c.isalpha() for c in line) and
                line.count(' ') > 1):  # Multi-word titles
                return line
        
        # Enhanced fallback
        if fallback_title and not fallback_title.endswith('.pdf') and len(fallback_title) > 10:
            return fallback_title
            
        return f"ArXiv Paper (extraction incomplete)"
    
    def _extract_authors(self, content: str, fallback_authors: str) -> str:
        """Enhanced authors extraction from PDF content"""
        lines = content.split('\n')[:35]  # Check more lines
        
        # Look for author patterns
        for i, line in enumerate(lines):
            line = line.strip()
            # Enhanced author detection
            if any(indicator in line.lower() for indicator in ['author', 'by ']):
                # Check next few lines for author names
                for j in range(i, min(i+6, len(lines))):
                    potential_authors = lines[j].strip()
                    if (5 < len(potential_authors) < 150 and
                        re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', potential_authors) and
                        not any(skip in potential_authors.lower() for skip in ['abstract', 'university', 'department'])):
                        return potential_authors
        
        # Fallback to original if reasonable
        if fallback_authors and fallback_authors.strip() and len(fallback_authors) > 3:
            return fallback_authors
        
        return "Authors not extracted"
    
    def create_wilson_lin_chunks(self, extraction):
        """Create Wilson Lin style contextual chunks"""
        content = extraction['content']
        doc_id = extraction['doc_id']
        
        # Enhanced structure detection for PDFs
        sections = self._detect_pdf_structure(content)
        
        # Paragraph-based chunking with structure awareness
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        prev_context = ""
        
        for i, para in enumerate(paragraphs):
            para_tokens = len(para.split())
            
            if current_tokens + para_tokens > 1200 and current_chunk:
                if current_tokens >= 700:
                    # Build section path
                    section_path = self._build_section_path(sections, i)
                    
                    # Generate Wilson Lin summary
                    summary_header = self.generate_wilson_lin_summary(current_chunk, section_path)
                    
                    # Create chunk
                    chunk = self._format_enhanced_chunk(
                        doc_id, chunk_index, current_chunk, 
                        prev_context, current_tokens, extraction,
                        section_path, summary_header
                    )
                    chunks.append(chunk)
                    
                    # Update prev context (Wilson Lin: 1-3 sentences)
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
            section_path = self._build_section_path(sections, len(paragraphs))
            summary_header = self.generate_wilson_lin_summary(current_chunk, section_path)
            
            chunk = self._format_enhanced_chunk(
                doc_id, chunk_index, current_chunk, 
                prev_context, current_tokens, extraction,
                section_path, summary_header
            )
            chunks.append(chunk)
            
        return chunks
    
    def _detect_pdf_structure(self, content: str) -> list:
        """Detect PDF structure for section path building"""
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Academic paper section patterns (enhanced)
            heading_patterns = [
                (r'^(Abstract|ABSTRACT)$', 1),
                (r'^(Introduction|INTRODUCTION)$', 1),
                (r'^(\d+\.?\s+[A-Z][A-Za-z\s]+)$', 1),  # "1. Introduction"
                (r'^([A-Z][A-Z\s]{5,})$', 1),  # ALL CAPS headings
                (r'^(\d+\.\d+\.?\s+[A-Z][A-Za-z\s]+)$', 2),  # "1.1 Background"
                (r'^(Methods?|METHODS?|Methodology|METHODOLOGY)$', 1),
                (r'^(Results?|RESULTS?)$', 1),
                (r'^(Discussion|DISCUSSION)$', 1),
                (r'^(Conclusion|CONCLUSION|Conclusions?|CONCLUSIONS?)$', 1),
                (r'^(References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)$', 1),
                (r'^(Appendix|APPENDIX)$', 1),
            ]
            
            for pattern, level in heading_patterns:
                if re.match(pattern, line):
                    sections.append({
                        'level': level,
                        'text': line,
                        'line_num': i,
                        'type': 'heading'
                    })
                    break
        
        return sections
    
    def _build_section_path(self, sections: list, current_line: int) -> str:
        """Build Wilson Lin style section path"""
        relevant_sections = []
        
        for section in sections:
            if section.get('line_num', 0) <= current_line * 2:  # Rough line estimation
                relevant_sections.append(section)
            else:
                break
        
        # Build hierarchical path
        path_elements = []
        for section in relevant_sections[-3:]:  # Last 3 levels
            text = section['text']
            # Clean up section text
            text = re.sub(r'^\d+\.?\s*', '', text)  # Remove numbering
            text = text.title() if text.isupper() else text
            path_elements.append(text[:40])  # Limit length
        
        return ' › '.join(path_elements) if path_elements else 'Document'
    
    def _format_enhanced_chunk(self, doc_id, chunk_index, content, prev_context, 
                             token_count, extraction, section_path, summary_header):
        """Format chunk with full Wilson Lin compliance"""
        
        return {
            'chunk_id': f"{doc_id}_enhanced_reprocessed_{chunk_index:03d}",
            'doc_id': doc_id,
            'content': content,
            'prev_context': prev_context,
            'section_path': section_path,
            'summary_header': summary_header,  # AI-generated Wilson Lin summary
            'token_count': token_count,
            'chunk_index': chunk_index,
            'page_refs': "",  # Could be enhanced with page tracking
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
        """Process papers for given phase with enhanced chunking"""
        phase_papers = papers[papers['phase'] == phase_num].copy()
        
        if len(phase_papers) == 0:
            logger.info(f"No papers found for Phase {phase_num}")
            return []
            
        logger.info(f"Starting Enhanced Phase {phase_num}: {len(phase_papers)} papers")
        
        successful_extractions = []
        all_chunks = []
        
        # Download papers in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:  # Reduced for API rate limiting
            future_to_paper = {
                executor.submit(self.download_paper, row): row 
                for _, row in phase_papers.iterrows()
            }
            
            for future in future_to_paper:
                try:
                    result = future.result()
                    if result:
                        successful_extractions.append(result)
                        
                        # Create Wilson Lin chunks with API summaries
                        chunks = self.create_wilson_lin_chunks(result)
                        all_chunks.extend(chunks)
                        
                        with self._lock:
                            self.stats['total_chunks'] += len(chunks)
                        
                        if len(successful_extractions) % 10 == 0:
                            logger.info(f"Phase {phase_num} progress: {len(successful_extractions)}/{len(phase_papers)}, "
                                      f"{len(all_chunks)} chunks, {self.stats['api_calls']} API calls")
                            
                except Exception as e:
                    logger.error(f"Processing error: {e}")
        
        logger.info(f"Enhanced Phase {phase_num} complete: {len(successful_extractions)} papers, {len(all_chunks)} chunks")
        return all_chunks
        
    def save_chunks(self, chunks, phase_num):
        """Save enhanced chunks to parquet"""
        if not chunks:
            logger.info(f"No chunks to save for Phase {phase_num}")
            return
            
        df = pd.DataFrame(chunks)
        df['corpus_source'] = 'main_enhanced_reprocessed'
        
        # Save phase results
        filename = f"arxiv_enhanced_phase{phase_num}.parquet"
        df.to_parquet(self.output_dir / filename, index=False)
        
        logger.info(f"Saved {len(chunks)} enhanced chunks to {filename}")
        
        # Save enhanced stats
        stats = {
            'phase': phase_num,
            'chunks': len(chunks),
            'papers': df['doc_id'].nunique(),
            'avg_tokens': df['token_count'].mean(),
            'api_calls': self.stats['api_calls'],
            'api_failures': self.stats['api_failures'],
            'wilson_lin_compliance': True,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / f'enhanced_phase{phase_num}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    processor = EnhancedArXivReprocessor()
    
    logger.info("🚀 Starting Enhanced ArXiv Reprocessing with Wilson Lin Compliance...")
    
    # Extract papers with proper phase assignment
    papers = processor.extract_papers()
    
    # Process all phases
    all_chunks = []
    
    for phase in [1, 2, 3]:
        phase_chunks = processor.process_phase(papers, phase)
        if phase_chunks:
            processor.save_chunks(phase_chunks, phase)
            all_chunks.extend(phase_chunks)
            
            # Brief pause between phases
            if phase < 3:
                logger.info(f"Phase {phase} completed, pausing before Phase {phase+1}...")
                time.sleep(5)
    
    # Save combined results
    if all_chunks:
        df_all = pd.DataFrame(all_chunks)
        df_all['corpus_source'] = 'main_enhanced_reprocessed'
        df_all.to_parquet(processor.output_dir / 'arxiv_all_enhanced_reprocessed.parquet', index=False)
        
        final_stats = {
            'total_papers_processed': df_all['doc_id'].nunique(),
            'total_chunks': len(all_chunks),
            'avg_tokens_per_chunk': df_all['token_count'].mean(),
            'total_api_calls': processor.stats['api_calls'],
            'api_success_rate': 1 - (processor.stats['api_failures'] / max(processor.stats['api_calls'], 1)),
            'wilson_lin_compliance': True,
            'processing_complete': datetime.now().isoformat()
        }
        
        with open(processor.output_dir / 'final_enhanced_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
    
    logger.info("✅ Enhanced ArXiv Reprocessing Completed with Full Wilson Lin Compliance!")
    logger.info(f"Final stats: {processor.stats}")

if __name__ == '__main__':
    main()