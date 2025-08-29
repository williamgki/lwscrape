#!/usr/bin/env python3
"""
Wilson Lin Contextual Chunker - Proper Implementation
Follows the exact Wilson Lin methodology:
- 700-1200 tokens per chunk
- AI-generated summary headers (1-2 lines)
- Section path prefixes
- Previous context (1-3 sentences)
- PDF page references where applicable
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re
import logging
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import hashlib
from datetime import datetime
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WilsonLinChunk:
    """Wilson Lin contextual chunk with all specified fields"""
    chunk_id: str
    doc_id: str
    content: str
    prev_context: str  # 1-3 sentences of previous chunk
    section_path: str  # Title › Part II › Mesa-Optimizers › Threat Models
    summary_header: str  # 1-2 line AI-generated TL;DR
    token_count: int
    char_count: int
    chunk_index: int
    page_start: Optional[int] = None  # For PDFs
    page_end: Optional[int] = None    # For PDFs
    # Metadata
    title: str = ""
    url: str = ""
    domain: str = ""
    content_type: str = "html"
    language: str = "en"
    structure_type: str = "content"

class WilsonLinTokenizer:
    """Accurate token counting for Wilson Lin sizing"""
    
    def __init__(self):
        # Approximate tokens per word for different content types
        self.token_ratios = {
            'academic': 1.4,  # Academic text has more complex vocabulary
            'code': 1.2,      # Code has specific tokenization
            'web': 1.3,       # Web content average
            'default': 1.3
        }
    
    def estimate_tokens(self, text: str, content_type: str = 'default') -> int:
        """Estimate token count with content-type awareness"""
        if not text:
            return 0
        
        word_count = len(text.split())
        ratio = self.token_ratios.get(content_type, self.token_ratios['default'])
        return int(word_count * ratio)
    
    def is_within_target(self, text: str, content_type: str = 'default') -> bool:
        """Check if text is within Wilson Lin target range (700-1200 tokens)"""
        tokens = self.estimate_tokens(text, content_type)
        return 700 <= tokens <= 1200
    
    def exceeds_max(self, text: str, content_type: str = 'default') -> bool:
        """Check if text exceeds maximum tokens"""
        tokens = self.estimate_tokens(text, content_type)
        return tokens > 1200

class WilsonLinStructureDetector:
    """Detect document structure for proper segmentation"""
    
    def __init__(self):
        # HTML heading patterns
        self.html_heading_patterns = [
            r'<h([1-6])[^>]*>(.*?)</h\1>',  # HTML headings
            r'^#+\s+(.+)$',                 # Markdown headings
            r'^([A-Z][A-Z\s]{10,50})$',     # ALL CAPS headings
        ]
        
        # Academic paper section patterns
        self.academic_patterns = [
            r'^(Abstract|Introduction|Background|Related Work|Methodology|Methods|Results|Discussion|Conclusion|References|Appendix)\s*$',
            r'^\d+\.?\s+(Introduction|Background|Methods|Results|Discussion|Conclusion)',
            r'^[IVX]+\.?\s+[A-Z]',  # Roman numerals
        ]
    
    def detect_structure(self, content: str, content_type: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect document structure and return sections
        Returns: List of (section_name, section_content, start_pos, end_pos)
        """
        sections = []
        
        if content_type == 'html':
            sections = self._detect_html_structure(content)
        elif content_type == 'pdf' and self._looks_academic(content):
            sections = self._detect_academic_structure(content)
        else:
            sections = self._detect_paragraph_structure(content)
        
        if not sections:
            # Fallback: treat entire content as single section
            sections = [("Document", content, 0, len(content))]
        
        return sections
    
    def _detect_html_structure(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Detect HTML structure using headings"""
        sections = []
        lines = content.split('\n')
        current_section = "Introduction"
        current_content = []
        start_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for heading patterns
            heading = self._extract_heading(line_stripped)
            if heading:
                # Save previous section
                if current_content:
                    section_text = '\n'.join(current_content)
                    sections.append((current_section, section_text, start_pos, start_pos + len(section_text)))
                
                # Start new section
                current_section = heading
                current_content = []
                start_pos = start_pos + len('\n'.join(current_content)) if current_content else 0
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            section_text = '\n'.join(current_content)
            sections.append((current_section, section_text, start_pos, start_pos + len(section_text)))
        
        return sections
    
    def _detect_academic_structure(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Detect academic paper structure"""
        sections = []
        paragraphs = content.split('\n\n')
        current_section = "Abstract"
        current_content = []
        pos = 0
        
        for para in paragraphs:
            para_stripped = para.strip()
            
            # Check for academic section patterns
            for pattern in self.academic_patterns:
                match = re.match(pattern, para_stripped, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_content:
                        section_text = '\n\n'.join(current_content)
                        sections.append((current_section, section_text, pos - len(section_text), pos))
                    
                    # Start new section
                    current_section = match.group(1) if match.lastindex else para_stripped[:50]
                    current_content = []
                    break
            else:
                current_content.append(para)
            
            pos += len(para) + 2  # +2 for \n\n
        
        # Add final section
        if current_content:
            section_text = '\n\n'.join(current_content)
            sections.append((current_section, section_text, pos - len(section_text), pos))
        
        return sections
    
    def _detect_paragraph_structure(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Simple paragraph-based structure"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Group paragraphs into reasonable sections
        sections = []
        section_size = max(3, len(paragraphs) // 5)  # Aim for ~5 sections
        
        for i in range(0, len(paragraphs), section_size):
            section_paras = paragraphs[i:i + section_size]
            section_name = f"Section {i // section_size + 1}"
            section_content = '\n\n'.join(section_paras)
            
            start_pos = sum(len(p) + 2 for p in paragraphs[:i])
            end_pos = start_pos + len(section_content)
            
            sections.append((section_name, section_content, start_pos, end_pos))
        
        return sections
    
    def _extract_heading(self, line: str) -> Optional[str]:
        """Extract heading from line"""
        for pattern in self.html_heading_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    return match.group(2).strip()  # HTML heading content
                else:
                    return match.group(1).strip()  # Markdown heading
        return None
    
    def _looks_academic(self, content: str) -> bool:
        """Check if content looks like academic paper"""
        academic_keywords = ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references']
        content_lower = content.lower()
        matches = sum(1 for keyword in academic_keywords if keyword in content_lower)
        return matches >= 3

class WilsonLinChunker:
    """Complete Wilson Lin contextual chunker"""
    
    def __init__(self):
        self.survey_normalized_dir = Path("/home/ubuntu/LW_scrape/survey_normalized")
        self.output_dir = Path("/home/ubuntu/LW_scrape/wilson_lin_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Wilson Lin parameters
        self.target_min_tokens = 700   # Wilson Lin specification
        self.target_max_tokens = 1200  # Wilson Lin specification
        self.context_sentences = 2     # 1-3 sentences of previous context
        
        # Components
        self.tokenizer = WilsonLinTokenizer()
        self.structure_detector = WilsonLinStructureDetector()
        
        # Threading
        self._lock = threading.Lock()
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'total_tokens': 0,
            'avg_tokens_per_chunk': 0,
            'chunks_in_range': 0,
            'failed_processing': 0
        }
        
        # Setup logging
        handler = logging.FileHandler(self.output_dir / 'wilson_lin_chunking.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        logger.info("🔧 Wilson Lin Chunker initialized with 700-1200 token target")
    
    def build_section_path(self, doc_title: str, section_name: str, subsection: str = None) -> str:
        """Build hierarchical section path"""
        parts = [doc_title[:50] if doc_title else "Document"]
        
        if section_name and section_name != "Document":
            parts.append(section_name)
        
        if subsection:
            parts.append(subsection)
        
        return " › ".join(parts)
    
    def extract_previous_context(self, previous_content: str, sentence_count: int = 2) -> str:
        """Extract 1-3 sentences from previous content"""
        if not previous_content:
            return ""
        
        sentences = nltk.sent_tokenize(previous_content)
        if len(sentences) <= sentence_count:
            return previous_content
        
        # Take last N sentences
        context_sentences = sentences[-sentence_count:]
        return ' '.join(context_sentences)
    
    def chunk_section_with_wilson_lin(self, section_name: str, section_content: str, 
                                     doc_metadata: dict, section_path: str) -> List[WilsonLinChunk]:
        """Chunk a section following Wilson Lin methodology"""
        chunks = []
        
        if not section_content or len(section_content.strip()) < 100:
            return chunks
        
        content_type = self._determine_content_type(doc_metadata)
        
        # If section fits in target range, use as single chunk
        if self.tokenizer.is_within_target(section_content, content_type):
            chunk = self._create_wilson_lin_chunk(
                content=section_content,
                prev_context="",
                section_path=section_path,
                doc_metadata=doc_metadata,
                chunk_index=0
            )
            chunks.append(chunk)
            return chunks
        
        # Section too large - subdivide by paragraphs
        paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
        
        current_chunk_paras = []
        chunk_index = 0
        previous_chunk_content = ""
        
        for para in paragraphs:
            # Test if adding this paragraph exceeds max tokens
            test_content = '\n\n'.join(current_chunk_paras + [para])
            
            if (current_chunk_paras and 
                self.tokenizer.exceeds_max(test_content, content_type) and
                self.tokenizer.estimate_tokens('\n\n'.join(current_chunk_paras), content_type) >= self.target_min_tokens):
                
                # Create chunk with current paragraphs
                chunk_content = '\n\n'.join(current_chunk_paras)
                prev_context = self.extract_previous_context(previous_chunk_content, self.context_sentences)
                
                chunk = self._create_wilson_lin_chunk(
                    content=chunk_content,
                    prev_context=prev_context,
                    section_path=section_path,
                    doc_metadata=doc_metadata,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                
                # Setup for next chunk
                previous_chunk_content = chunk_content
                current_chunk_paras = [para]  # Start new chunk with current paragraph
                chunk_index += 1
            else:
                current_chunk_paras.append(para)
        
        # Handle remaining paragraphs
        if current_chunk_paras:
            chunk_content = '\n\n'.join(current_chunk_paras)
            prev_context = self.extract_previous_context(previous_chunk_content, self.context_sentences)
            
            chunk = self._create_wilson_lin_chunk(
                content=chunk_content,
                prev_context=prev_context,
                section_path=section_path,
                doc_metadata=doc_metadata,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_wilson_lin_chunk(self, content: str, prev_context: str, section_path: str,
                                doc_metadata: dict, chunk_index: int) -> WilsonLinChunk:
        """Create a Wilson Lin chunk with all required fields"""
        
        doc_id = doc_metadata['doc_id']
        chunk_id = self._generate_chunk_id(doc_id, chunk_index)
        content_type = self._determine_content_type(doc_metadata)
        
        token_count = self.tokenizer.estimate_tokens(content, content_type)
        
        return WilsonLinChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            prev_context=prev_context,
            section_path=section_path,
            summary_header="",  # Will be filled by AI summary generation
            token_count=token_count,
            char_count=len(content),
            chunk_index=chunk_index,
            page_start=None,  # TODO: Extract from PDF metadata if available
            page_end=None,
            title=doc_metadata.get('title', ''),
            url=doc_metadata.get('canonical_url', ''),
            domain=doc_metadata.get('domain', ''),
            content_type=doc_metadata.get('content_type', 'html'),
            language=doc_metadata.get('language', 'en'),
            structure_type=self._classify_structure_type(content)
        )
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique Wilson Lin chunk ID"""
        chunk_key = f"wl_{doc_id}_{chunk_index}"
        return hashlib.sha256(chunk_key.encode('utf-8')).hexdigest()[:16]
    
    def _determine_content_type(self, doc_metadata: dict) -> str:
        """Determine content type for tokenization"""
        content_type = doc_metadata.get('content_type', 'html')
        domain = doc_metadata.get('domain', 'web')
        
        if content_type == 'pdf' or domain == 'paper':
            return 'academic'
        elif content_type == 'code':
            return 'code'
        else:
            return 'web'
    
    def _classify_structure_type(self, content: str) -> str:
        """Classify chunk structure type"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['abstract', 'summary', 'tldr']):
            return 'summary'
        elif any(word in content_lower for word in ['introduction', 'overview', 'background']):
            return 'introduction'
        elif any(word in content_lower for word in ['method', 'approach', 'algorithm']):
            return 'methodology'
        elif any(word in content_lower for word in ['result', 'finding', 'evaluation']):
            return 'results'
        elif any(word in content_lower for word in ['conclusion', 'discussion']):
            return 'conclusion'
        else:
            return 'content'
    
    def process_single_document(self, doc_data: dict) -> List[WilsonLinChunk]:
        """Process single document with Wilson Lin methodology"""
        try:
            doc_id = doc_data['doc_id']
            content = doc_data['content']
            doc_title = doc_data.get('title', 'Untitled')
            content_type = doc_data.get('content_type', 'html')
            
            # Detect document structure
            sections = self.structure_detector.detect_structure(content, content_type)
            
            all_chunks = []
            
            # Process each section
            for section_name, section_content, start_pos, end_pos in sections:
                section_path = self.build_section_path(doc_title, section_name)
                
                section_chunks = self.chunk_section_with_wilson_lin(
                    section_name=section_name,
                    section_content=section_content,
                    doc_metadata=doc_data,
                    section_path=section_path
                )
                
                all_chunks.extend(section_chunks)
            
            # Update statistics
            with self._lock:
                self.stats['documents_processed'] += 1
                self.stats['chunks_created'] += len(all_chunks)
                total_tokens = sum(chunk.token_count for chunk in all_chunks)
                self.stats['total_tokens'] += total_tokens
                
                # Count chunks in target range
                in_range = sum(1 for chunk in all_chunks 
                             if self.target_min_tokens <= chunk.token_count <= self.target_max_tokens)
                self.stats['chunks_in_range'] += in_range
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {doc_data.get('doc_id', 'unknown')}: {e}")
            with self._lock:
                self.stats['failed_processing'] += 1
            return []
    
    def chunk_survey_corpus(self, max_workers: int = 6):
        """Process entire survey corpus with Wilson Lin methodology"""
        logger.info("🔧 Starting Wilson Lin contextual chunking (700-1200 tokens)...")
        
        # Load normalized documents
        doc_file = self.survey_normalized_dir / 'survey_normalized_documents.parquet'
        df = pd.read_parquet(doc_file)
        
        logger.info(f"📚 Loaded {len(df):,} normalized documents")
        
        # Convert to dict format
        documents = df.to_dict('records')
        
        # Process in parallel
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_doc = {executor.submit(self.process_single_document, doc): doc 
                           for doc in documents}
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_doc):
                completed += 1
                
                chunks = future.result()
                all_chunks.extend(chunks)
                
                if completed % 100 == 0:
                    progress = (completed / len(documents)) * 100
                    logger.info(f"📈 Progress: {progress:.1f}% ({completed:,}/{len(documents):,}) - "
                              f"WL Chunks: {len(all_chunks):,}")
        
        # Calculate final statistics
        if all_chunks:
            self.stats['avg_tokens_per_chunk'] = self.stats['total_tokens'] / len(all_chunks)
            
        # Save Wilson Lin chunks (without AI summaries yet)
        if all_chunks:
            logger.info(f"💾 Saving {len(all_chunks):,} Wilson Lin chunks...")
            self._save_wilson_lin_chunks(all_chunks)
        
        # Log completion
        self._log_wilson_lin_stats(len(all_chunks))
        
        return len(all_chunks)
    
    def _save_wilson_lin_chunks(self, chunks: List[WilsonLinChunk]):
        """Save Wilson Lin chunks to parquet"""
        chunk_data = []
        
        for chunk in chunks:
            chunk_data.append({
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'content': chunk.content,
                'prev_context': chunk.prev_context,
                'section_path': chunk.section_path,
                'summary_header': chunk.summary_header,  # Empty for now
                'token_count': chunk.token_count,
                'char_count': chunk.char_count,
                'chunk_index': chunk.chunk_index,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'title': chunk.title,
                'url': chunk.url,
                'domain': chunk.domain,
                'content_type': chunk.content_type,
                'language': chunk.language,
                'structure_type': chunk.structure_type
            })
        
        df = pd.DataFrame(chunk_data)
        
        # Save to parquet
        output_file = self.output_dir / 'wilson_lin_chunks.parquet'
        df.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Saved Wilson Lin chunks to {output_file}")
    
    def _log_wilson_lin_stats(self, chunk_count: int):
        """Log comprehensive Wilson Lin statistics"""
        logger.info("\n" + "="*60)
        logger.info("🔧 WILSON LIN CONTEXTUAL CHUNKING COMPLETE!")
        logger.info("="*60)
        logger.info(f"📊 WILSON LIN STATISTICS:")
        logger.info(f"  Documents processed: {self.stats['documents_processed']:,}")
        logger.info(f"  Wilson Lin chunks: {self.stats['chunks_created']:,}")
        logger.info(f"  Total tokens: {self.stats['total_tokens']:,}")
        logger.info(f"  Average tokens per chunk: {self.stats['avg_tokens_per_chunk']:.0f}")
        logger.info(f"  Chunks in 700-1200 range: {self.stats['chunks_in_range']:,} ({self.stats['chunks_in_range']/max(chunk_count,1)*100:.1f}%)")
        logger.info(f"  Failed processing: {self.stats['failed_processing']:,}")
        logger.info("="*60)
        
        # Save stats
        stats = {
            **self.stats,
            'wilson_lin_complete': datetime.now().isoformat(),
            'target_token_range': f"{self.target_min_tokens}-{self.target_max_tokens}",
            'context_sentences': self.context_sentences,
            'in_range_percentage': self.stats['chunks_in_range']/max(chunk_count,1)*100
        }
        
        with open(self.output_dir / 'wilson_lin_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    # System resources
    import psutil
    
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1024**3
    
    # Conservative workers for Wilson Lin processing
    optimal_workers = min(max(1, cpu_count // 3), 6)
    
    logger.info(f"🖥️  System resources: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    logger.info(f"⚡ Using {optimal_workers} workers for Wilson Lin chunking")
    
    chunker = WilsonLinChunker()
    chunk_count = chunker.chunk_survey_corpus(max_workers=optimal_workers)
    
    logger.info(f"🎯 Wilson Lin chunking complete: {chunk_count:,} chunks ready for AI summary generation")

if __name__ == '__main__':
    main()