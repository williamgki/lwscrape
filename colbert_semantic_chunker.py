#!/usr/bin/env python3

import re
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer
import logging

from docunits_model import DocUnit, SourceIds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """Semantic chunk optimized for ColBERT indexing"""
    chunk_id: str
    paper_id: str
    section_id: str
    page: int
    chunk_text: str
    token_count: int
    source_para_ids: List[str]  # Original DocUnit para_ids that formed this chunk
    figure_refs: List[str]  # Referenced figure IDs
    attached_captions: List[str]  # Caption text attached to this chunk
    chunk_type: str  # "text", "mixed" (text+caption), "figure_context"


class ColBERTSemanticChunker:
    """Create semantic chunks optimized for ColBERT late-interaction retrieval"""
    
    def __init__(self, 
                 target_token_range: Tuple[int, int] = (350, 600),
                 model_name: str = "colbert-ir/colbertv2.0"):
        self.target_min, self.target_max = target_token_range
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Figure reference patterns
        self.figure_patterns = [
            re.compile(r'\bFig(ure)?\s*(\d+)', re.IGNORECASE),
            re.compile(r'\bTable\s*(\d+)', re.IGNORECASE),
            re.compile(r'\bAppendix\s*([A-Z]|\d+)', re.IGNORECASE)
        ]
        
        # Section break indicators for semantic windowing
        self.section_breaks = [
            'introduction', 'background', 'related work', 'method', 'methodology',
            'approach', 'model', 'results', 'evaluation', 'experiments', 
            'discussion', 'conclusion', 'limitations', 'future work',
            'acknowledgments', 'references'
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using ColBERT tokenizer"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def extract_figure_references(self, text: str) -> List[str]:
        """Extract figure/table references from text"""
        refs = []
        for pattern in self.figure_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    ref_num = match[-1]  # Get the number part
                else:
                    ref_num = match
                refs.append(ref_num)
        return list(set(refs))  # Remove duplicates
    
    def is_section_boundary(self, section_id: str, next_section_id: str) -> bool:
        """Check if there's a semantic boundary between sections"""
        if section_id == next_section_id:
            return False
        
        # Check if crossing major section boundaries
        current_type = self._classify_section(section_id.lower())
        next_type = self._classify_section(next_section_id.lower())
        
        return current_type != next_type
    
    def _classify_section(self, section_name: str) -> str:
        """Classify section into major types"""
        section_lower = section_name.lower()
        
        if any(term in section_lower for term in ['intro', 'background', 'related']):
            return 'intro'
        elif any(term in section_lower for term in ['method', 'approach', 'model', 'algorithm']):
            return 'methods' 
        elif any(term in section_lower for term in ['result', 'evaluation', 'experiment', 'analysis']):
            return 'results'
        elif any(term in section_lower for term in ['discussion', 'conclusion', 'limitation']):
            return 'conclusion'
        else:
            return 'other'
    
    def find_captions_for_chunk(self, 
                               figure_refs: List[str], 
                               all_units: List[DocUnit]) -> List[str]:
        """Find caption text for figures referenced in chunk"""
        captions = []
        
        for unit in all_units:
            if unit.type == "caption" and unit.figure_meta:
                # Extract figure number from figure_meta
                fig_id = unit.figure_meta.figure_id
                fig_num = re.search(r'\d+', fig_id)
                if fig_num and fig_num.group() in figure_refs:
                    captions.append(unit.text)
        
        return captions
    
    def chunk_docunits_by_paper(self, 
                               docunits: List[DocUnit]) -> List[SemanticChunk]:
        """Create semantic chunks from DocUnits, grouped by paper"""
        
        # Group by paper
        papers = {}
        for unit in docunits:
            if unit.paper_id not in papers:
                papers[unit.paper_id] = []
            papers[unit.paper_id].append(unit)
        
        all_chunks = []
        
        for paper_id, units in papers.items():
            logger.info(f"Chunking paper {paper_id} ({len(units)} units)")
            
            # Sort units by page, then by section for logical order
            text_units = [u for u in units if u.type == "text"]
            text_units.sort(key=lambda x: (x.page, x.section_id, x.para_id))
            
            paper_chunks = self._chunk_paper_units(paper_id, text_units, units)
            all_chunks.extend(paper_chunks)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks total")
        return all_chunks
    
    def _chunk_paper_units(self, 
                          paper_id: str, 
                          text_units: List[DocUnit],
                          all_units: List[DocUnit]) -> List[SemanticChunk]:
        """Create chunks for a single paper"""
        
        chunks = []
        current_chunk_units = []
        current_tokens = 0
        
        i = 0
        while i < len(text_units):
            unit = text_units[i]
            unit_tokens = self.count_tokens(unit.text)
            
            # Check if adding this unit would exceed max tokens
            if current_tokens + unit_tokens > self.target_max and current_chunk_units:
                # Create chunk from current units
                chunk = self._create_chunk_from_units(
                    paper_id, current_chunk_units, all_units
                )
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk_units = []
                current_tokens = 0
            
            # Add current unit to chunk
            current_chunk_units.append(unit)
            current_tokens += unit_tokens
            
            # Check for semantic boundary
            if (i + 1 < len(text_units) and 
                current_tokens >= self.target_min and
                self.is_section_boundary(unit.section_id, text_units[i + 1].section_id)):
                
                # Create chunk at semantic boundary
                chunk = self._create_chunk_from_units(
                    paper_id, current_chunk_units, all_units
                )
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk_units = []
                current_tokens = 0
            
            i += 1
        
        # Handle remaining units
        if current_chunk_units:
            chunk = self._create_chunk_from_units(
                paper_id, current_chunk_units, all_units
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_units(self, 
                               paper_id: str,
                               units: List[DocUnit],
                               all_units: List[DocUnit]) -> SemanticChunk:
        """Create a SemanticChunk from a list of DocUnits"""
        
        # Combine text from units
        chunk_text = " ".join([unit.text for unit in units])
        
        # Extract figure references
        figure_refs = []
        for unit in units:
            unit_refs = self.extract_figure_references(unit.text)
            figure_refs.extend(unit_refs)
        figure_refs = list(set(figure_refs))  # Remove duplicates
        
        # Find and attach relevant captions
        captions = self.find_captions_for_chunk(figure_refs, all_units)
        
        # Attach captions to chunk text if found
        if captions:
            chunk_text += "\n\n" + " ".join(captions)
            chunk_type = "mixed"
        else:
            chunk_type = "text"
        
        # Get section and page info
        section_id = units[0].section_id
        page = units[0].page
        
        # Ensure chunk doesn't exceed max tokens after caption attachment
        final_tokens = self.count_tokens(chunk_text)
        if final_tokens > self.target_max * 1.2:  # Allow 20% overflow for captions
            # Truncate captions if needed
            base_text = " ".join([unit.text for unit in units])
            remaining_tokens = self.target_max - self.count_tokens(base_text)
            
            if remaining_tokens > 50:  # Leave room for captions
                caption_text = " ".join(captions)
                caption_tokens = self.tokenizer.encode(caption_text, add_special_tokens=False)
                
                if len(caption_tokens) > remaining_tokens:
                    # Truncate caption
                    truncated_tokens = caption_tokens[:remaining_tokens]
                    caption_text = self.tokenizer.decode(truncated_tokens)
                
                chunk_text = base_text + "\n\n" + caption_text
            else:
                chunk_text = base_text
                chunk_type = "text"
        
        return SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=paper_id,
            section_id=section_id,
            page=page,
            chunk_text=chunk_text,
            token_count=self.count_tokens(chunk_text),
            source_para_ids=[unit.para_id for unit in units],
            figure_refs=figure_refs,
            attached_captions=captions,
            chunk_type=chunk_type
        )
    
    def chunk_from_docunits_db(self, db_path: str) -> List[SemanticChunk]:
        """Create chunks directly from DocUnits database"""
        import duckdb
        
        conn = duckdb.connect(db_path)
        
        # Load DocUnits from database
        df = conn.execute("""
            SELECT paper_id, source_ids, section_id, page, para_id, type, text, figure_meta, refs
            FROM docunits
            ORDER BY paper_id, page, section_id, para_id
        """).df()
        
        # Convert to DocUnit objects
        docunits = []
        for _, row in df.iterrows():
            source_ids = SourceIds(**row['source_ids']) if row['source_ids'] else SourceIds()
            
            unit = DocUnit(
                paper_id=row['paper_id'],
                source_ids=source_ids,
                section_id=row['section_id'],
                page=row['page'],
                para_id=row['para_id'],
                type=row['type'],
                text=row['text'],
                figure_meta=row['figure_meta'],
                refs=row['refs'] or []
            )
            docunits.append(unit)
        
        conn.close()
        
        return self.chunk_docunits_by_paper(docunits)
    
    def export_chunks(self, chunks: List[SemanticChunk], output_file: Path):
        """Export chunks to JSON for ColBERT indexing"""
        
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                'chunk_id': chunk.chunk_id,
                'paper_id': chunk.paper_id,
                'section_id': chunk.section_id,
                'page': chunk.page,
                'chunk_text': chunk.chunk_text,
                'token_count': chunk.token_count,
                'source_para_ids': chunk.source_para_ids,
                'figure_refs': chunk.figure_refs,
                'attached_captions': chunk.attached_captions,
                'chunk_type': chunk.chunk_type
            })
        
        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        logger.info(f"Exported {len(chunks)} chunks to {output_file}")
    
    def get_chunking_stats(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking process"""
        
        token_counts = [chunk.token_count for chunk in chunks]
        chunk_types = [chunk.chunk_type for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'target_range': f"{self.target_min}-{self.target_max}",
            'chunks_in_range': sum(1 for t in token_counts if self.target_min <= t <= self.target_max),
            'chunk_type_distribution': {
                chunk_type: chunk_types.count(chunk_type) 
                for chunk_type in set(chunk_types)
            },
            'papers_processed': len(set(chunk.paper_id for chunk in chunks)),
            'chunks_with_figures': sum(1 for chunk in chunks if chunk.figure_refs),
            'chunks_with_captions': sum(1 for chunk in chunks if chunk.attached_captions)
        }
        
        stats['in_range_percentage'] = (stats['chunks_in_range'] / stats['total_chunks']) * 100
        
        return stats


def main():
    """Demo semantic chunking for ColBERT"""
    
    chunker = ColBERTSemanticChunker()
    
    # Create chunks from DocUnits database
    chunks = chunker.chunk_from_docunits_db("docunits.db")
    
    # Export for ColBERT indexing
    output_file = Path("colbert_semantic_chunks.json")
    chunker.export_chunks(chunks, output_file)
    
    # Print statistics
    stats = chunker.get_chunking_stats(chunks)
    print("\n=== Semantic Chunking Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()