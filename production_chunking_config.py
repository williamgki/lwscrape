#!/usr/bin/env python3

import re
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from transformers import AutoTokenizer

from splade_retriever import SPLADERetriever

from docunits_model import DocUnit, SourceIds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production chunking settings
CHUNK_TOKENS = 512
STRIDE = 256

@dataclass
class ProductionChunk:
    """Production-ready chunk with all required metadata"""
    chunk_id: str
    paper_id: str
    page_from: int
    page_to: int
    text: str
    headers: List[str]  # Title > Section > Subsection path
    figure_ids: List[str]  # Referenced figure IDs
    caption_text: str = ""  # Attached caption if figure cited
    ref_inline_markers: List[str] = field(default_factory=list)  # [@Doe2023] markers
    length_tokens: int = 0
    
    # Storage paths for embeddings (will be populated during indexing)
    colbert_vectors_path: Optional[str] = None
    bm25_terms: Optional[Dict[str, float]] = None
    splade_sparse: Optional[Dict[str, float]] = None
    
    # Source tracking
    source_docunit_ids: List[str] = field(default_factory=list)
    chunk_type: str = "text"  # text, mixed (text+caption), figure_context


class ProductionChunker:
    """Production chunker with optimal settings for retrieval"""
    
    def __init__(self,
                 tokenizer_model: str = "colbert-ir/colbertv2.0",
                 chunk_tokens: int = CHUNK_TOKENS,
                 stride_tokens: int = STRIDE,
                 splade_retriever: Optional[SPLADERetriever] = None):

        self.chunk_tokens = chunk_tokens
        self.stride_tokens = stride_tokens
        self.splade_retriever = splade_retriever

        # Load tokenizer for accurate token counting
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        # Figure citation patterns for caption attachment
        self.figure_patterns = [
            re.compile(r'\b(Fig(?:ure)?\.?\s*(\d+[a-z]?))', re.IGNORECASE),
            re.compile(r'\b(Table\s*(\d+[a-z]?))', re.IGNORECASE),
            re.compile(r'\b(Appendix\s*([A-Z]|\d+))', re.IGNORECASE)
        ]
        
        # Inline reference patterns [@Author2023] [1] etc.
        self.ref_patterns = [
            re.compile(r'\[@[^\]]+\]'),  # [@Author2023]
            re.compile(r'\[[0-9,\-\s]+\]'),  # [1], [1-3], [1, 2, 5]
            re.compile(r'\([^)]*\d{4}[^)]*\)'),  # (Author et al., 2023)
        ]
    
    def count_tokens(self, text: str) -> int:
        """Accurate token counting"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def extract_figure_citations(self, text: str) -> List[str]:
        """Extract figure citations from text"""
        citations = []
        for pattern in self.figure_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    citations.append(match[0])  # Full match
                else:
                    citations.append(match)
        return list(set(citations))
    
    def extract_inline_refs(self, text: str) -> List[str]:
        """Extract inline reference markers"""
        refs = []
        for pattern in self.ref_patterns:
            refs.extend(pattern.findall(text))
        return list(set(refs))
    
    def build_header_path(self, docunit: DocUnit, paper_metadata: Dict) -> List[str]:
        """Build Title > Section > Subsection header path"""
        headers = []
        
        # Add paper title
        if paper_metadata.get('title'):
            headers.append(paper_metadata['title'])
        
        # Add section hierarchy
        section_id = docunit.section_id
        if section_id and section_id != "unknown":
            # Parse hierarchical sections (e.g., "2.1 Methods > 2.1.1 Data Collection")
            if ">" in section_id:
                section_parts = [s.strip() for s in section_id.split(">")]
                headers.extend(section_parts)
            else:
                headers.append(section_id)
        
        return headers
    
    def find_caption_for_figures(self, 
                                figure_ids: List[str], 
                                current_page: int,
                                all_docunits: List[DocUnit]) -> str:
        """Find caption text for cited figures on same/adjacent pages"""
        
        caption_texts = []
        target_pages = {current_page - 1, current_page, current_page + 1}
        
        for docunit in all_docunits:
            if (docunit.type == "caption" and 
                docunit.page in target_pages and
                docunit.figure_meta):
                
                # Check if this caption matches any cited figure
                fig_id = docunit.figure_meta.figure_id
                for cited_fig in figure_ids:
                    # Extract number from both
                    cited_num = re.search(r'\d+', cited_fig)
                    fig_num = re.search(r'\d+', fig_id)
                    
                    if cited_num and fig_num and cited_num.group() == fig_num.group():
                        caption_texts.append(docunit.text)
                        break
        
        return " ".join(caption_texts)
    
    def create_sliding_window_chunks(self, 
                                   docunits: List[DocUnit],
                                   paper_metadata: Dict,
                                   all_docunits: List[DocUnit]) -> List[ProductionChunk]:
        """Create sliding window chunks within sections"""
        
        # Group by section and sort
        sections = {}
        for unit in docunits:
            if unit.type == "text":  # Only chunk text units
                section_key = f"{unit.section_id}_{unit.page}"
                if section_key not in sections:
                    sections[section_key] = []
                sections[section_key].append(unit)
        
        all_chunks = []
        
        for section_key, units in sections.items():
            # Sort units within section
            units.sort(key=lambda x: (x.page, x.para_id))
            
            # Create sliding window chunks
            section_chunks = self._chunk_section_with_stride(
                units, paper_metadata, all_docunits
            )
            all_chunks.extend(section_chunks)
        
        logger.info(f"Created {len(all_chunks)} production chunks for paper {paper_metadata.get('id', 'unknown')}")
        return all_chunks
    
    def _chunk_section_with_stride(self, 
                                  section_units: List[DocUnit],
                                  paper_metadata: Dict,
                                  all_docunits: List[DocUnit]) -> List[ProductionChunk]:
        """Create overlapping chunks within a section"""
        
        if not section_units:
            return []
        
        chunks = []
        current_start = 0
        
        while current_start < len(section_units):
            chunk = self._create_chunk_from_window(
                section_units, current_start, paper_metadata, all_docunits
            )
            if chunk:
                chunks.append(chunk)
            
            # Move window by stride
            current_start += self._calculate_stride_units(
                section_units, current_start
            )
        
        return chunks
    
    def _create_chunk_from_window(self,
                                 section_units: List[DocUnit],
                                 start_idx: int,
                                 paper_metadata: Dict,
                                 all_docunits: List[DocUnit]) -> Optional[ProductionChunk]:
        """Create a single chunk from a window of units"""
        
        if start_idx >= len(section_units):
            return None
        
        # Build chunk text with target token count
        chunk_texts = []
        source_ids = []
        current_tokens = 0
        end_idx = start_idx
        
        # Get header path from first unit
        first_unit = section_units[start_idx]
        headers = self.build_header_path(first_unit, paper_metadata)
        header_text = " > ".join(headers)
        header_tokens = self.count_tokens(header_text)
        
        # Reserve tokens for header
        available_tokens = self.chunk_tokens - header_tokens - 10  # Small buffer
        
        # Add units until token limit
        for i in range(start_idx, len(section_units)):
            unit = section_units[i]
            unit_tokens = self.count_tokens(unit.text)
            
            if current_tokens + unit_tokens > available_tokens and chunk_texts:
                break
            
            chunk_texts.append(unit.text)
            source_ids.append(unit.para_id)
            current_tokens += unit_tokens
            end_idx = i
        
        if not chunk_texts:
            return None
        
        # Combine chunk text
        base_text = " ".join(chunk_texts)
        
        # Extract figure citations and inline refs
        figure_ids = self.extract_figure_citations(base_text)
        inline_refs = self.extract_inline_refs(base_text)
        
        # Find caption text for cited figures
        current_page = first_unit.page
        caption_text = self.find_caption_for_figures(
            figure_ids, current_page, all_docunits
        )
        
        # Prepare final text with header path
        final_text_parts = [header_text, base_text]
        if caption_text:
            final_text_parts.append(f"Caption: {caption_text}")
        
        final_text = "\n".join(final_text_parts)
        final_tokens = self.count_tokens(final_text)
        
        # Determine page range
        end_unit = section_units[end_idx]
        page_from = first_unit.page
        page_to = end_unit.page
        
        # Determine chunk type
        chunk_type = "mixed" if caption_text else "text"
        
        chunk = ProductionChunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=first_unit.paper_id,
            page_from=page_from,
            page_to=page_to,
            text=final_text,
            headers=headers,
            figure_ids=figure_ids,
            caption_text=caption_text,
            ref_inline_markers=inline_refs,
            length_tokens=final_tokens,
            source_docunit_ids=source_ids,
            chunk_type=chunk_type
        )

        # Optionally generate and cache SPLADE sparse vectors
        if self.splade_retriever:
            try:
                self.splade_retriever.cache_chunk(chunk)
            except Exception as e:
                logger.warning(f"SPLADE encoding failed for chunk {chunk.chunk_id}: {e}")

        return chunk
    
    def _calculate_stride_units(self, 
                               section_units: List[DocUnit], 
                               current_start: int) -> int:
        """Calculate how many units to advance for stride"""
        
        # Find the unit index that gives us approximately STRIDE tokens
        stride_tokens = 0
        stride_units = 1  # At least advance by 1 unit
        
        for i in range(current_start, min(current_start + 5, len(section_units))):
            unit = section_units[i]
            stride_tokens += self.count_tokens(unit.text)
            
            if stride_tokens >= self.stride_tokens:
                stride_units = i - current_start + 1
                break
        
        return max(1, stride_units)  # Always advance at least 1 unit


class ProductionChunkStorage:
    """Storage system for production chunks with all metadata"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "chunks").mkdir(exist_ok=True)
        (self.storage_path / "embeddings").mkdir(exist_ok=True)
        (self.storage_path / "indexes").mkdir(exist_ok=True)
    
    def save_chunks(self, chunks: List[ProductionChunk], paper_id: str):
        """Save chunks for a paper"""
        
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'paper_id': chunk.paper_id,
                'page_from': chunk.page_from,
                'page_to': chunk.page_to,
                'text': chunk.text,
                'headers': chunk.headers,
                'figure_ids': chunk.figure_ids,
                'caption_text': chunk.caption_text,
                'ref_inline_markers': chunk.ref_inline_markers,
                'length_tokens': chunk.length_tokens,
                'source_docunit_ids': chunk.source_docunit_ids,
                'chunk_type': chunk.chunk_type,
                'colbert_vectors_path': chunk.colbert_vectors_path,
                'bm25_terms': chunk.bm25_terms,
                'splade_sparse': chunk.splade_sparse
            }
            chunks_data.append(chunk_data)
        
        # Save to JSON file
        output_file = self.storage_path / "chunks" / f"{paper_id}_chunks.json"
        with open(output_file, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks for paper {paper_id}")
    
    def load_chunks(self, paper_id: str) -> List[ProductionChunk]:
        """Load chunks for a paper"""
        
        input_file = self.storage_path / "chunks" / f"{paper_id}_chunks.json"
        if not input_file.exists():
            return []
        
        with open(input_file, 'r') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for data in chunks_data:
            chunk = ProductionChunk(
                chunk_id=data['chunk_id'],
                paper_id=data['paper_id'],
                page_from=data['page_from'],
                page_to=data['page_to'],
                text=data['text'],
                headers=data['headers'],
                figure_ids=data['figure_ids'],
                caption_text=data.get('caption_text', ''),
                ref_inline_markers=data.get('ref_inline_markers', []),
                length_tokens=data['length_tokens'],
                source_docunit_ids=data.get('source_docunit_ids', []),
                chunk_type=data.get('chunk_type', 'text'),
                colbert_vectors_path=data.get('colbert_vectors_path'),
                bm25_terms=data.get('bm25_terms'),
                splade_sparse=data.get('splade_sparse')
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get statistics about stored chunks"""
        
        chunk_files = list((self.storage_path / "chunks").glob("*_chunks.json"))
        
        total_chunks = 0
        total_tokens = 0
        chunk_types = {}
        papers_processed = len(chunk_files)
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)
            
            total_chunks += len(chunks_data)
            
            for chunk_data in chunks_data:
                total_tokens += chunk_data.get('length_tokens', 0)
                chunk_type = chunk_data.get('chunk_type', 'text')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'papers_processed': papers_processed,
            'total_chunks': total_chunks,
            'avg_chunks_per_paper': total_chunks / max(papers_processed, 1),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': total_tokens / max(total_chunks, 1),
            'chunk_type_distribution': chunk_types,
            'target_chunk_tokens': CHUNK_TOKENS,
            'stride_tokens': STRIDE
        }


def main():
    """Demo production chunking configuration"""
    
    # Initialize production chunker
    chunker = ProductionChunker()
    storage = ProductionChunkStorage(Path("./production_chunks"))
    
    print("=== Production Chunking Configuration ===")
    print(f"Chunk tokens: {CHUNK_TOKENS}")
    print(f"Stride tokens: {STRIDE}")
    print(f"Tokenizer: colbert-ir/colbertv2.0")
    
    print("\nFeatures enabled:")
    print("✓ Header path prepending (Title > Section > Subsection)")
    print("✓ Caption attachment for figure citations")
    print("✓ Inline reference marker preservation")
    print("✓ Sliding window with stride within sections")
    print("✓ Production-ready metadata storage")
    
    # Example chunk structure
    print("\nChunk storage schema:")
    print("- chunk_id, paper_id, page_from, page_to")
    print("- text (with headers), headers, figure_ids")
    print("- caption_text, ref_inline_markers")
    print("- colbert_vectors_path, bm25_terms, splade_sparse")
    print("- length_tokens, source_docunit_ids")


if __name__ == "__main__":
    main()