#!/usr/bin/env python3
"""
Contextual chunking implementation following Wilson Lin's approach.
Uses Claude API for summarization headers and preserves semantic context.
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from html.parser import HTMLParser
import pandas as pd
import tiktoken
import anthropic
from aisitools.api_key import get_api_key_for_proxy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for a contextual chunk"""
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_path: str
    token_count: int
    page_refs: Optional[str] = None
    content_type: str = "html"

@dataclass 
class ContextualChunk:
    """A contextually enriched chunk following Wilson Lin's approach"""
    content: str
    prev_context: str
    section_path: str
    summary_header: str
    metadata: ChunkMetadata

class StructureParser(HTMLParser):
    """Parse HTML structure to extract headings and sections"""
    
    def __init__(self):
        super().__init__()
        self.headings = []
        self.current_heading = None
        self.heading_level = 0
        self.capturing_text = False
        self.text_buffer = ""
        
    def handle_starttag(self, tag, attrs):
        if tag.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.heading_level = int(tag[1])
            self.capturing_text = True
            self.text_buffer = ""
            
    def handle_endtag(self, tag):
        if tag.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.capturing_text = False
            if self.text_buffer.strip():
                self.headings.append({
                    'level': self.heading_level,
                    'text': self.text_buffer.strip(),
                    'tag': tag.lower()
                })
            self.text_buffer = ""
            
    def handle_data(self, data):
        if self.capturing_text:
            self.text_buffer += data

class ContextualChunker:
    """Wilson Lin's contextual chunking implementation"""
    
    def __init__(self, min_tokens: int = 700, max_tokens: int = 1200, model: str = "claude-3-5-haiku-latest"):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.anthropic_client = None
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
            logger.info("✅ Claude API client initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Claude API: {e}")
            raise
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
        
    def parse_html_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse HTML content to extract structural elements"""
        parser = StructureParser()
        try:
            parser.feed(content)
        except Exception as e:
            logger.warning(f"HTML parsing error: {e}")
            
        return parser.headings
        
    def parse_pdf_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract structure from PDF content (simplified)"""
        # Simple approach: detect potential headings by common patterns
        headings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Common heading patterns
            if (re.match(r'^[A-Z][A-Z\s]{5,}$', line) or  # ALL CAPS
                re.match(r'^\d+\.?\s+[A-Z][a-zA-Z\s]*$', line) or  # Numbered sections
                re.match(r'^[A-Z][a-z]*\s*:.*$', line)):  # Title: pattern
                
                headings.append({
                    'level': 1,  # Simplified - could enhance with level detection
                    'text': line,
                    'line_num': i
                })
                
        return headings
        
    def build_section_path(self, headings: List[Dict], current_pos: int) -> str:
        """Build hierarchical section path"""
        path_elements = []
        
        for heading in headings:
            if heading.get('line_num', 0) <= current_pos:
                path_elements.append(heading['text'])
            else:
                break
                
        return ' › '.join(path_elements[-3:])  # Keep last 3 levels
        
    def split_into_chunks(self, content: str, content_type: str = "html") -> List[Dict[str, Any]]:
        """Split content into semantic chunks with structure preservation"""
        chunks = []
        
        if content_type == "html":
            headings = self.parse_html_structure(content)
        else:
            headings = self.parse_pdf_structure(content)
            
        # Split by paragraphs as base units
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_tokens = 0
        chunk_start_idx = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_tokens = self.count_tokens(paragraph)
            
            # Check if adding this paragraph exceeds max tokens
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Save current chunk if it meets minimum
                if current_tokens >= self.min_tokens:
                    section_path = self.build_section_path(headings, i)
                    
                    chunks.append({
                        'content': current_chunk.strip(),
                        'token_count': current_tokens,
                        'section_path': section_path,
                        'start_paragraph': chunk_start_idx,
                        'end_paragraph': i - 1
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
            section_path = self.build_section_path(headings, len(paragraphs))
            chunks.append({
                'content': current_chunk.strip(),
                'token_count': current_tokens,
                'section_path': section_path,
                'start_paragraph': chunk_start_idx,
                'end_paragraph': len(paragraphs) - 1
            })
            
        return chunks
        
    def add_contextual_elements(self, chunks: List[Dict], doc_content: str) -> List[Dict]:
        """Add prev_context to chunks"""
        paragraphs = [p.strip() for p in doc_content.split('\n\n') if p.strip()]
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                chunk['prev_context'] = ""
            else:
                # Get 1-3 sentences from previous chunk
                prev_chunk = chunks[i-1]['content']
                sentences = re.split(r'[.!?]+', prev_chunk)
                prev_context = '. '.join(sentences[-2:]).strip()
                chunk['prev_context'] = prev_context
                
        return chunks
        
    async def generate_summary_header(self, chunk_content: str, section_path: str) -> str:
        """Generate 1-2 line summary header using Claude"""
        try:
            prompt = f"""Given this text chunk from section "{section_path}", write a concise 1-2 line summary that captures the main point:

{chunk_content[:1000]}...

Summary (1-2 lines max):"""

            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            # Ensure it's actually 1-2 lines
            lines = summary.split('\n')
            return '. '.join(lines[:2]) if len(lines) > 1 else lines[0]
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback: use first sentence
            sentences = re.split(r'[.!?]+', chunk_content)
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
            
    def process_document(self, doc_data: Dict) -> List[ContextualChunk]:
        """Process a single document into contextual chunks"""
        content = doc_data.get('content', '')
        content_type = doc_data.get('metadata', {}).get('content_type', 'html')
        doc_id = doc_data.get('doc_id', 'unknown')
        
        logger.info(f"Processing document {doc_id} ({self.count_tokens(content)} tokens)")
        
        # Step 1: Split into semantic chunks
        raw_chunks = self.split_into_chunks(content, content_type)
        logger.info(f"Split into {len(raw_chunks)} raw chunks")
        
        # Step 2: Add contextual elements
        chunks_with_context = self.add_contextual_elements(raw_chunks, content)
        
        # Step 3: Generate summary headers (synchronously for now)
        contextual_chunks = []
        
        for i, chunk_data in enumerate(chunks_with_context):
            # Generate summary header
            summary_header = self.generate_summary_sync(
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
                content_type=content_type
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
            
        logger.info(f"Generated {len(contextual_chunks)} contextual chunks")
        return contextual_chunks
        
    def generate_summary_sync(self, chunk_content: str, section_path: str) -> str:
        """Synchronous version of summary generation"""
        try:
            prompt = f"""Given this text chunk from section "{section_path}", write a concise 1-2 line summary that captures the main point:

{chunk_content[:1000]}...

Summary (1-2 lines max):"""

            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            lines = summary.split('\n')
            return '. '.join(lines[:2]) if len(lines) > 1 else lines[0]
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback
            sentences = re.split(r'[.!?]+', chunk_content)
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]


def load_sample_documents(parquet_path: str, sample_size: int = 10) -> List[Dict]:
    """Load a small sample of documents for testing"""
    logger.info(f"Loading sample of {sample_size} documents from {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    sample_df = df.head(sample_size)
    
    documents = []
    for _, row in sample_df.iterrows():
        documents.append({
            'doc_id': row['doc_id'],
            'content': row['content'],
            'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
        })
        
    logger.info(f"Loaded {len(documents)} sample documents")
    return documents


def main():
    """Test contextual chunking on sample documents"""
    
    # Load sample documents
    parquet_path = "/home/ubuntu/LW_scrape/normalized_corpus/document_store.parquet"
    sample_docs = load_sample_documents(parquet_path, sample_size=3)
    
    # Initialize chunker
    chunker = ContextualChunker()
    
    # Process sample
    all_chunks = []
    for doc in sample_docs:
        try:
            chunks = chunker.process_document(doc)
            all_chunks.extend(chunks)
            
            logger.info(f"Document {doc['doc_id'][:12]}... → {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to process document {doc['doc_id']}: {e}")
            continue
            
    logger.info(f"Total contextual chunks generated: {len(all_chunks)}")
    
    # Show sample chunk
    if all_chunks:
        sample_chunk = all_chunks[0]
        print("\n=== SAMPLE CONTEXTUAL CHUNK ===")
        print(f"Section Path: {sample_chunk.section_path}")
        print(f"Summary Header: {sample_chunk.summary_header}")
        print(f"Previous Context: {sample_chunk.prev_context}")
        print(f"Content ({sample_chunk.metadata.token_count} tokens):")
        print(sample_chunk.content[:500] + "...")
        

if __name__ == "__main__":
    main()