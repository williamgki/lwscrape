#!/usr/bin/env python3
"""
Enhanced contextual chunker implementing Wilson Lin's approach with production-ready features.
Includes improved structure parsing, PDF page handling, and multi-worker capabilities.
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
import pandas as pd
import tiktoken
import anthropic
from aisitools.api_key import get_api_key_for_proxy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata for a contextual chunk"""
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_path: str
    token_count: int
    page_refs: Optional[str] = None
    content_type: str = "html"
    structure_type: str = "paragraph"  # heading, list_item, table_caption, paragraph
    heading_level: Optional[int] = None
    doc_title: str = ""
    domain: str = ""
    original_url: str = ""
    authors: str = ""
    pub_date: Optional[str] = None

@dataclass 
class ContextualChunk:
    """A contextually enriched chunk following Wilson Lin's approach"""
    content: str
    prev_context: str
    section_path: str
    summary_header: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'chunk_id': self.metadata.chunk_id,
            'doc_id': self.metadata.doc_id,
            'content': self.content,
            'prev_context': self.prev_context,
            'section_path': self.section_path,
            'summary_header': self.summary_header,
            'token_count': self.metadata.token_count,
            'chunk_index': self.metadata.chunk_index,
            'page_refs': self.metadata.page_refs,
            'content_type': self.metadata.content_type,
            'structure_type': self.metadata.structure_type,
            'heading_level': self.metadata.heading_level,
            'doc_title': self.metadata.doc_title,
            'domain': self.metadata.domain,
            'original_url': self.metadata.original_url,
            'authors': self.metadata.authors,
            'pub_date': self.metadata.pub_date
        }

class EnhancedStructureParser(HTMLParser):
    """Enhanced parser for HTML structure including headings, lists, tables"""
    
    def __init__(self):
        super().__init__()
        self.elements = []  # All structural elements
        self.current_element = None
        self.capturing_text = False
        self.text_buffer = ""
        self.position = 0
        self.in_list = False
        self.list_level = 0
        
    def handle_starttag(self, tag, attrs):
        self.position += 1
        tag_lower = tag.lower()
        
        # Headings
        if tag_lower in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.current_element = {
                'type': 'heading',
                'level': int(tag_lower[1]),
                'tag': tag_lower,
                'position': self.position,
                'text': ''
            }
            self.capturing_text = True
            self.text_buffer = ""
            
        # List items
        elif tag_lower == 'li':
            self.current_element = {
                'type': 'list_item',
                'level': self.list_level,
                'position': self.position,
                'text': ''
            }
            self.capturing_text = True
            self.text_buffer = ""
            
        # List containers
        elif tag_lower in ['ul', 'ol']:
            self.in_list = True
            self.list_level += 1
            
        # Table captions
        elif tag_lower == 'caption':
            self.current_element = {
                'type': 'table_caption',
                'position': self.position,
                'text': ''
            }
            self.capturing_text = True
            self.text_buffer = ""
            
        # Section elements
        elif tag_lower in ['section', 'article', 'main', 'aside']:
            # Check for semantic classes/ids
            attrs_dict = dict(attrs)
            class_attr = attrs_dict.get('class', '')
            id_attr = attrs_dict.get('id', '')
            
            if any(keyword in (class_attr + ' ' + id_attr).lower() 
                  for keyword in ['content', 'main', 'article', 'section', 'chapter']):
                self.elements.append({
                    'type': 'section_boundary',
                    'tag': tag_lower,
                    'position': self.position,
                    'class': class_attr,
                    'id': id_attr,
                    'text': ''
                })
        
    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        
        if self.capturing_text and self.current_element:
            self.current_element['text'] = self.text_buffer.strip()
            if self.current_element['text']:
                self.elements.append(self.current_element)
            self.capturing_text = False
            self.current_element = None
            self.text_buffer = ""
            
        # Handle list nesting
        if tag_lower in ['ul', 'ol']:
            self.list_level = max(0, self.list_level - 1)
            if self.list_level == 0:
                self.in_list = False
            
    def handle_data(self, data):
        if self.capturing_text:
            self.text_buffer += data
            
    def get_structural_elements(self):
        """Get all structural elements sorted by position"""
        return sorted(self.elements, key=lambda x: x['position'])

class PDFStructureParser:
    """Enhanced PDF structure parser with page boundary detection"""
    
    def __init__(self):
        self.page_breaks = []
        self.headings = []
        
    def parse_pdf_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract structure from PDF content with page awareness"""
        elements = []
        lines = content.split('\n')
        current_page = 1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect page breaks (common patterns)
            if (re.match(r'^\s*Page\s+\d+\s*$', line, re.IGNORECASE) or
                re.match(r'^\s*\d+\s*$', line) and len(line) <= 3):
                current_page += 1
                elements.append({
                    'type': 'page_break',
                    'page': current_page,
                    'line_num': i,
                    'text': line
                })
                continue
                
            # Detect headings with enhanced patterns
            heading_level = self._detect_heading_level(line, i, lines)
            if heading_level:
                elements.append({
                    'type': 'heading',
                    'level': heading_level,
                    'text': line,
                    'line_num': i,
                    'page': current_page
                })
                
            # Detect table captions
            elif re.match(r'^Table\s+\d+[\.:]\s*', line, re.IGNORECASE):
                elements.append({
                    'type': 'table_caption',
                    'text': line,
                    'line_num': i,
                    'page': current_page
                })
                
            # Detect figure captions
            elif re.match(r'^Figure\s+\d+[\.:]\s*', line, re.IGNORECASE):
                elements.append({
                    'type': 'figure_caption',
                    'text': line,
                    'line_num': i,
                    'page': current_page
                })
                
        return elements
        
    def _detect_heading_level(self, line: str, line_num: int, all_lines: List[str]) -> Optional[int]:
        """Enhanced heading detection with multiple patterns"""
        
        # Pattern 1: ALL CAPS (likely heading)
        if (len(line) > 5 and len(line) < 100 and 
            line.isupper() and 
            not re.match(r'^[A-Z]+\s*\d+\s*$', line)):  # Not just "PAGE 1"
            return 1
            
        # Pattern 2: Numbered sections (1.1, 2.3.4, etc.)
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z][a-zA-Z\s]*$', line):
            dots = line.count('.')
            return min(3, dots + 1)
            
        # Pattern 3: Title case with colons
        if (re.match(r'^[A-Z][a-zA-Z\s]*:\s*', line) and 
            len(line) > 10 and len(line) < 80):
            return 2
            
        # Pattern 4: Roman numerals
        if re.match(r'^[IVX]+\.\s+[A-Z][a-zA-Z\s]*$', line):
            return 2
            
        # Pattern 5: Single word titles (Abstract, Introduction, etc.)
        if (line in ['Abstract', 'Introduction', 'Conclusion', 'Results', 
                    'Discussion', 'Methods', 'References', 'Bibliography'] or
            (len(line.split()) == 1 and line.istitle() and len(line) > 3)):
            return 1
            
        return None

class SectionPathBuilder:
    """Build hierarchical section paths from structural elements"""
    
    def __init__(self, max_path_length: int = 3):
        self.max_path_length = max_path_length
        
    def build_path_from_elements(self, elements: List[Dict], current_position: int, 
                                doc_title: str = "") -> str:
        """Build section path from structural elements"""
        
        # Find relevant headings before current position
        relevant_headings = []
        for element in elements:
            if (element.get('type') == 'heading' and 
                element.get('line_num', element.get('position', 0)) <= current_position):
                relevant_headings.append(element)
                
        if not relevant_headings:
            return doc_title or "Document"
            
        # Build hierarchical path
        path_elements = []
        
        # Add document title if available
        if doc_title and doc_title.strip():
            path_elements.append(doc_title.strip()[:50])
            
        # Sort by level and position, take last few
        relevant_headings.sort(key=lambda x: (x.get('level', 1), 
                                            x.get('line_num', x.get('position', 0))))
        
        # Keep last few headings by hierarchy
        current_level = float('inf')
        for heading in reversed(relevant_headings):
            level = heading.get('level', 1)
            if level <= current_level:
                path_elements.insert(-1 if doc_title else 0, 
                                   heading['text'][:40])
                current_level = level
                
        # Limit path length
        if len(path_elements) > self.max_path_length:
            path_elements = path_elements[:1] + path_elements[-(self.max_path_length-1):]
            
        return ' › '.join(path_elements) if path_elements else "Document"
        
    def build_pdf_path(self, elements: List[Dict], current_line: int, 
                      doc_title: str = "") -> Tuple[str, Optional[str]]:
        """Build section path for PDFs with page references"""
        
        # Find current page
        current_page = 1
        page_start = current_page
        
        for element in elements:
            if (element.get('type') == 'page_break' and 
                element.get('line_num', 0) <= current_line):
                current_page = element.get('page', current_page + 1)
                
        # Build regular path
        path = self.build_path_from_elements(elements, current_line, doc_title)
        
        # Add page reference
        page_ref = f"p. {current_page}" if current_page > 0 else None
        
        return path, page_ref

class EnhancedContextualChunker:
    """Production-ready contextual chunker with Wilson Lin's approach"""
    
    def __init__(self, min_tokens: int = 700, max_tokens: int = 1200, 
                 model: str = "claude-3-5-haiku-latest", api_rate_limit: float = 0.1):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.anthropic_client = None
        self.api_rate_limit = api_rate_limit  # seconds between API calls
        self.last_api_call = 0
        self.api_lock = threading.Lock()
        self.path_builder = SectionPathBuilder()
        self._setup_claude_api()
        
    def _setup_claude_api(self):
        """Setup Claude API client using aisitools proxy"""
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
            logger.info("✅ Enhanced Claude API client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Claude API: {e}")
            raise
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
        
    def parse_html_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse HTML content to extract enhanced structural elements"""
        parser = EnhancedStructureParser()
        try:
            parser.feed(content)
            return parser.get_structural_elements()
        except Exception as e:
            logger.warning(f"HTML parsing error: {e}")
            return []
            
    def parse_pdf_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract enhanced structure from PDF content"""
        parser = PDFStructureParser()
        return parser.parse_pdf_structure(content)
        
    def split_into_semantic_chunks(self, content: str, elements: List[Dict], 
                                 content_type: str = "html", doc_title: str = "") -> List[Dict[str, Any]]:
        """Split content into semantic chunks using structural elements"""
        
        chunks = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return chunks
            
        current_chunk = ""
        current_tokens = 0
        chunk_start_idx = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)
            
            # Check if adding this paragraph exceeds max tokens
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Save current chunk if it meets minimum
                if current_tokens >= self.min_tokens:
                    
                    if content_type == "pdf":
                        section_path, page_ref = self.path_builder.build_pdf_path(
                            elements, i, doc_title)
                    else:
                        section_path = self.path_builder.build_path_from_elements(
                            elements, i, doc_title)
                        page_ref = None
                    
                    # Determine structure type
                    structure_type, heading_level = self._get_structure_type(
                        elements, i, current_chunk)
                    
                    chunks.append({
                        'content': current_chunk.strip(),
                        'token_count': current_tokens,
                        'section_path': section_path,
                        'page_refs': page_ref,
                        'start_paragraph': chunk_start_idx,
                        'end_paragraph': i - 1,
                        'structure_type': structure_type,
                        'heading_level': heading_level
                    })
                    
                # Start new chunk
                current_chunk = paragraph
                current_tokens = para_tokens
                chunk_start_idx = i
            else:
                # Add to current chunk
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
                current_tokens += para_tokens
                
        # Don't forget the last chunk
        if current_chunk and current_tokens >= self.min_tokens:
            if content_type == "pdf":
                section_path, page_ref = self.path_builder.build_pdf_path(
                    elements, len(paragraphs), doc_title)
            else:
                section_path = self.path_builder.build_path_from_elements(
                    elements, len(paragraphs), doc_title)
                page_ref = None
                
            structure_type, heading_level = self._get_structure_type(
                elements, len(paragraphs), current_chunk)
            
            chunks.append({
                'content': current_chunk.strip(),
                'token_count': current_tokens,
                'section_path': section_path,
                'page_refs': page_ref,
                'start_paragraph': chunk_start_idx,
                'end_paragraph': len(paragraphs) - 1,
                'structure_type': structure_type,
                'heading_level': heading_level
            })
            
        return chunks
        
    def _get_structure_type(self, elements: List[Dict], position: int, 
                          content: str) -> Tuple[str, Optional[int]]:
        """Determine the structure type of a chunk"""
        
        # Check if chunk starts with a heading
        content_start = content[:100].strip()
        
        for element in elements:
            element_pos = element.get('line_num', element.get('position', 0))
            if abs(element_pos - position) <= 2:  # Close to current position
                
                if element.get('type') == 'heading':
                    return 'heading', element.get('level')
                elif element.get('type') == 'list_item':
                    return 'list_item', None
                elif element.get('type') in ['table_caption', 'figure_caption']:
                    return element['type'], None
                    
        # Default to paragraph
        return 'paragraph', None
        
    def add_contextual_elements(self, chunks: List[Dict], doc_content: str) -> List[Dict]:
        """Add prev_context to chunks"""
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                chunk['prev_context'] = ""
            else:
                # Get 1-3 sentences from previous chunk
                prev_chunk_content = chunks[i-1]['content']
                sentences = re.split(r'[.!?]+\s+', prev_chunk_content)
                
                # Take last 2-3 sentences, ensuring we don't exceed reasonable length
                context_sentences = []
                for sentence in reversed(sentences[-3:]):
                    sentence = sentence.strip()
                    if sentence and len(' '.join(context_sentences + [sentence])) < 300:
                        context_sentences.insert(0, sentence)
                    else:
                        break
                        
                chunk['prev_context'] = '. '.join(context_sentences)
                if chunk['prev_context'] and not chunk['prev_context'].endswith('.'):
                    chunk['prev_context'] += '.'
                
        return chunks
        
    def generate_summary_with_rate_limit(self, chunk_content: str, 
                                       section_path: str) -> str:
        """Generate summary header with API rate limiting"""
        
        with self.api_lock:
            # Enforce rate limit
            time_since_last = time.time() - self.last_api_call
            if time_since_last < self.api_rate_limit:
                time.sleep(self.api_rate_limit - time_since_last)
                
            try:
                prompt = f"""Given this text chunk from section "{section_path}", write a concise 1-2 line summary that captures the main point. Focus on the key insight or concept:

{chunk_content[:800]}...

Summary (1-2 lines, under 150 words):"""

                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                summary = response.content[0].text.strip()
                
                # Clean up summary
                lines = [line.strip() for line in summary.split('\n') if line.strip()]
                final_summary = '. '.join(lines[:2]) if len(lines) > 1 else lines[0] if lines else ""
                
                # Ensure reasonable length
                if len(final_summary) > 200:
                    final_summary = final_summary[:200] + "..."
                    
                self.last_api_call = time.time()
                return final_summary
                
            except Exception as e:
                logger.warning(f"API call failed: {e}")
                # Fallback: extract first meaningful sentence
                sentences = re.split(r'[.!?]+\s+', chunk_content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 200:
                        return sentence + "."
                        
                return "Content summary unavailable."
            
    def process_document(self, doc_data: Dict) -> List[ContextualChunk]:
        """Process a single document into contextual chunks"""
        
        content = doc_data.get('content', '')
        content_type = doc_data.get('content_type', 'html')
        doc_id = doc_data.get('doc_id', 'unknown')
        title = doc_data.get('title', '')
        domain = doc_data.get('domain', '')
        original_url = doc_data.get('original_url', '')
        authors = doc_data.get('authors', '')
        pub_date = doc_data.get('pub_date')
        
        if not content.strip():
            logger.warning(f"Empty content for document {doc_id}")
            return []
            
        logger.info(f"Processing document {doc_id[:12]}... ({self.count_tokens(content)} tokens)")
        
        # Parse structure
        if content_type == "pdf":
            elements = self.parse_pdf_structure(content)
        else:
            elements = self.parse_html_structure(content)
            
        logger.debug(f"Found {len(elements)} structural elements")
        
        # Split into semantic chunks
        raw_chunks = self.split_into_semantic_chunks(
            content, elements, content_type, title)
        logger.info(f"Split into {len(raw_chunks)} raw chunks")
        
        if not raw_chunks:
            return []
            
        # Add contextual elements
        chunks_with_context = self.add_contextual_elements(raw_chunks, content)
        
        # Generate contextual chunks with summaries
        contextual_chunks = []
        
        for i, chunk_data in enumerate(chunks_with_context):
            
            # Generate summary header
            summary_header = self.generate_summary_with_rate_limit(
                chunk_data['content'], 
                chunk_data['section_path']
            )
            
            # Create metadata
            chunk_id = f"{doc_id}_chunk_{i:03d}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=i,
                section_path=chunk_data['section_path'],
                token_count=chunk_data['token_count'],
                page_refs=chunk_data.get('page_refs'),
                content_type=content_type,
                structure_type=chunk_data.get('structure_type', 'paragraph'),
                heading_level=chunk_data.get('heading_level'),
                doc_title=title,
                domain=domain,
                original_url=original_url,
                authors=authors,
                pub_date=pub_date
            )
            
            # Create contextual chunk
            contextual_chunk = ContextualChunk(
                content=chunk_data['content'],
                prev_context=chunk_data['prev_context'],
                section_path=chunk_data['section_path'],
                summary_header=summary_header,
                metadata=metadata
            )
            
            contextual_chunks.append(contextual_chunk)
            
        logger.info(f"Generated {len(contextual_chunks)} contextual chunks for {doc_id[:12]}...")
        return contextual_chunks


def load_documents_batch(parquet_path: str, batch_size: int = 100, 
                        start_idx: int = 0) -> List[Dict]:
    """Load a batch of documents from parquet for processing"""
    
    df = pd.read_parquet(parquet_path)
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    
    documents = []
    for _, row in batch_df.iterrows():
        doc_data = {
            'doc_id': row['doc_id'],
            'content': row['content'],
            'content_type': row['content_type'],
            'title': row.get('title', ''),
            'domain': row.get('domain', ''),
            'original_url': row.get('original_url', ''),
            'authors': row.get('authors', ''),
            'pub_date': row.get('pub_date')
        }
        documents.append(doc_data)
        
    return documents


def test_enhanced_chunker():
    """Test enhanced contextual chunker on samples"""
    
    logger.info("=== TESTING ENHANCED CONTEXTUAL CHUNKER ===")
    
    # Load sample documents
    parquet_path = "/home/ubuntu/LW_scrape/normalized_corpus/document_store.parquet"
    sample_docs = load_documents_batch(parquet_path, batch_size=3, start_idx=100)
    
    if not sample_docs:
        logger.error("No documents loaded")
        return
        
    logger.info(f"Loaded {len(sample_docs)} test documents")
    
    # Initialize enhanced chunker
    chunker = EnhancedContextualChunker(
        min_tokens=400,    # Lower for testing
        max_tokens=800,    # Lower for testing
        api_rate_limit=0.2  # 5 calls per second max
    )
    
    # Process samples
    all_chunks = []
    for doc in sample_docs:
        try:
            chunks = chunker.process_document(doc)
            all_chunks.extend(chunks)
            
            logger.info(f"✅ {doc['doc_id'][:12]}... → {len(chunks)} chunks")
            
            # Show first chunk details
            if chunks:
                chunk = chunks[0]
                logger.info(f"   Sample: {chunk.summary_header}")
                logger.info(f"   Path: {chunk.section_path}")
                logger.info(f"   Type: {chunk.metadata.structure_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to process {doc['doc_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    logger.info(f"\n🎉 ENHANCED CHUNKER TEST COMPLETE!")
    logger.info(f"Generated {len(all_chunks)} contextual chunks")
    
    if all_chunks:
        # Show detailed sample
        sample_chunk = all_chunks[0]
        print("\n=== ENHANCED CHUNK SAMPLE ===")
        print(f"Section Path: {sample_chunk.section_path}")
        print(f"Structure Type: {sample_chunk.metadata.structure_type}")
        print(f"Page Refs: {sample_chunk.metadata.page_refs}")
        print(f"Summary Header: {sample_chunk.summary_header}")
        print(f"Previous Context: {sample_chunk.prev_context}")
        print(f"Content ({sample_chunk.metadata.token_count} tokens):")
        print(sample_chunk.content[:300] + "...")


if __name__ == "__main__":
    test_enhanced_chunker()