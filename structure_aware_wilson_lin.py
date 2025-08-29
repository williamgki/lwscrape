#!/usr/bin/env python3
"""
Structure-Aware Wilson Lin Chunker
Enhanced Wilson Lin chunking that leverages GROBID structure parsing
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import hashlib
from datetime import datetime
import nltk
from dataclasses import dataclass

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StructuredChunk:
    """Enhanced chunk with structural information"""
    chunk_id: str
    doc_id: str
    content: str
    prev_context: str
    section_path: str
    summary_header: str
    token_count: int
    char_count: int
    chunk_index: int
    
    # Enhanced structural fields
    section_type: str  # introduction, methodology, results, etc.
    hierarchical_level: int  # 1=section, 2=subsection, etc.
    contains_figures: List[str]  # Figure/table IDs referenced
    citations: List[str]  # Citation references
    structural_metadata: Dict  # Additional structure info
    
    # Original fields
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    title: str = ""
    url: str = ""
    domain: str = ""
    content_type: str = ""
    language: str = ""
    structure_type: str = ""

class StructureAwareWilsonLinChunker:
    def __init__(self, target_tokens: Tuple[int, int] = (700, 1200)):
        self.target_min_tokens = target_tokens[0]
        self.target_max_tokens = target_tokens[1]
        
        # Academic section types and their priorities
        self.section_types = {
            'abstract': {'priority': 1, 'type': 'summary'},
            'introduction': {'priority': 2, 'type': 'introduction'},
            'related work': {'priority': 3, 'type': 'methodology'},
            'background': {'priority': 3, 'type': 'methodology'},
            'methodology': {'priority': 4, 'type': 'methodology'},
            'method': {'priority': 4, 'type': 'methodology'},
            'approach': {'priority': 4, 'type': 'methodology'},
            'experiments': {'priority': 5, 'type': 'results'},
            'results': {'priority': 5, 'type': 'results'},
            'evaluation': {'priority': 5, 'type': 'results'},
            'discussion': {'priority': 6, 'type': 'results'},
            'analysis': {'priority': 6, 'type': 'results'},
            'conclusion': {'priority': 7, 'type': 'conclusion'},
            'future work': {'priority': 8, 'type': 'conclusion'},
            'limitations': {'priority': 8, 'type': 'conclusion'},
            'references': {'priority': 9, 'type': 'references'}
        }
        
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count using word-based approximation"""
        if not text:
            return 0
        word_count = len(text.split())
        return int(word_count * 1.3)  # Conservative estimate for subword tokenization
    
    def classify_section_type(self, section_title: str) -> Tuple[str, int]:
        """Classify section type based on title"""
        title_lower = section_title.lower().strip()
        
        # Direct matches
        for section_name, info in self.section_types.items():
            if section_name in title_lower:
                return info['type'], info['priority']
        
        # Pattern-based classification
        if re.search(r'\b(intro|introduction)\b', title_lower):
            return 'introduction', 2
        elif re.search(r'\b(method|approach|technique|algorithm)\b', title_lower):
            return 'methodology', 4
        elif re.search(r'\b(result|experiment|evaluation|performance)\b', title_lower):
            return 'results', 5
        elif re.search(r'\b(discuss|analysis|finding)\b', title_lower):
            return 'results', 6
        elif re.search(r'\b(conclusion|summary|final)\b', title_lower):
            return 'conclusion', 7
        elif re.search(r'\b(related|prior|previous|background)\b', title_lower):
            return 'methodology', 3
        
        return 'content', 10  # Default
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text"""
        citations = []
        
        # Pattern for numbered citations [1], [2], [1,2,3]
        numbered_pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        numbered_matches = re.findall(numbered_pattern, text)
        for match in numbered_matches:
            citations.extend([f"[{num.strip()}]" for num in match.split(',')])
        
        # Pattern for author-year citations (Smith et al., 2023)
        author_year_pattern = r'\([A-Z][a-z]+(?:\s+et\s+al\.)?[,\s]+\d{4}\)'
        author_year_matches = re.findall(author_year_pattern, text)
        citations.extend(author_year_matches)
        
        return list(set(citations))  # Remove duplicates
    
    def extract_figure_references(self, text: str) -> List[str]:
        """Extract figure and table references"""
        references = []
        
        # Figure references: Figure 1, Fig. 2, etc.
        fig_pattern = r'\b(?:Figure|Fig\.?)\s*(\d+)\b'
        fig_matches = re.findall(fig_pattern, text, re.IGNORECASE)
        references.extend([f"Figure {num}" for num in fig_matches])
        
        # Table references: Table 1, Tab. 2, etc.
        table_pattern = r'\b(?:Table|Tab\.?)\s*(\d+)\b'
        table_matches = re.findall(table_pattern, text, re.IGNORECASE)
        references.extend([f"Table {num}" for num in table_matches])
        
        return list(set(references))
    
    def build_section_hierarchy(self, sections: List[Dict]) -> str:
        """Build hierarchical section path"""
        path_parts = []
        
        for section in sections:
            title = section.get('title', '')
            level = section.get('level', 1)
            
            if title and title.lower() not in ['references', 'bibliography']:
                # Truncate very long section titles
                if len(title) > 50:
                    title = title[:47] + "..."
                path_parts.append(title)
        
        if not path_parts:
            return "Document"
        
        return ' › '.join(path_parts)
    
    def chunk_structured_section(self, section: Dict, doc_context: Dict, parent_sections: List[Dict] = None) -> List[StructuredChunk]:
        """Chunk a structured section with context awareness"""
        chunks = []
        
        if parent_sections is None:
            parent_sections = []
        
        section_title = section.get('title', '')
        section_content = section.get('content', '')
        section_level = section.get('level', 1)
        subsections = section.get('subsections', [])
        
        # Classify section type
        section_type, priority = self.classify_section_type(section_title)
        
        # Build section path
        current_path = parent_sections + [section]
        section_path = self.build_section_hierarchy(current_path)
        
        # If section has substantial content, chunk it
        if section_content and len(section_content.strip()) > 100:
            content_chunks = self.chunk_section_content(
                section_content,
                section_title,
                section_path,
                section_type,
                section_level,
                doc_context
            )
            chunks.extend(content_chunks)
        
        # Process subsections
        for subsection in subsections:
            subsection_chunks = self.chunk_structured_section(
                subsection, 
                doc_context, 
                current_path
            )
            chunks.extend(subsection_chunks)
        
        return chunks
    
    def chunk_section_content(self, content: str, section_title: str, section_path: str, 
                            section_type: str, level: int, doc_context: Dict) -> List[StructuredChunk]:
        """Chunk content within a section"""
        
        chunks = []
        
        # Split content into sentences
        sentences = nltk.sent_tokenize(content)
        
        current_chunk_sentences = []
        current_tokens = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.estimate_token_count(sentence)
            
            # Check if adding this sentence would exceed max tokens
            if (current_tokens + sentence_tokens > self.target_max_tokens and 
                current_chunk_sentences and 
                current_tokens >= self.target_min_tokens):
                
                # Create chunk from current sentences
                chunk = self.create_structured_chunk(
                    current_chunk_sentences,
                    section_title,
                    section_path,
                    section_type,
                    level,
                    chunk_index,
                    doc_context,
                    chunks  # For previous context
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_tokens = sentence_tokens
                chunk_index += 1
                
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle remaining sentences
        if current_chunk_sentences:
            # If chunk is too small, try to merge with previous chunk
            if (chunks and current_tokens < self.target_min_tokens and 
                len(chunks[-1].content.split()) < self.target_max_tokens * 0.8):
                
                # Merge with previous chunk
                prev_chunk = chunks[-1]
                merged_content = prev_chunk.content + " " + " ".join(current_chunk_sentences)
                merged_tokens = self.estimate_token_count(merged_content)
                
                if merged_tokens <= self.target_max_tokens:
                    # Update previous chunk
                    chunks[-1] = StructuredChunk(
                        chunk_id=prev_chunk.chunk_id,
                        doc_id=prev_chunk.doc_id,
                        content=merged_content,
                        prev_context=prev_chunk.prev_context,
                        section_path=prev_chunk.section_path,
                        summary_header=prev_chunk.summary_header,
                        token_count=merged_tokens,
                        char_count=len(merged_content),
                        chunk_index=prev_chunk.chunk_index,
                        section_type=prev_chunk.section_type,
                        hierarchical_level=prev_chunk.hierarchical_level,
                        contains_figures=prev_chunk.contains_figures + self.extract_figure_references(merged_content),
                        citations=prev_chunk.citations + self.extract_citations(merged_content),
                        structural_metadata=prev_chunk.structural_metadata,
                        title=doc_context.get('title', ''),
                        url=doc_context.get('url', ''),
                        domain=doc_context.get('domain', ''),
                        content_type=doc_context.get('content_type', ''),
                        language=doc_context.get('language', ''),
                        structure_type='academic'
                    )
                else:
                    # Create separate chunk
                    chunk = self.create_structured_chunk(
                        current_chunk_sentences,
                        section_title,
                        section_path,
                        section_type,
                        level,
                        chunk_index,
                        doc_context,
                        chunks
                    )
                    chunks.append(chunk)
            else:
                # Create final chunk
                chunk = self.create_structured_chunk(
                    current_chunk_sentences,
                    section_title,
                    section_path,
                    section_type,
                    level,
                    chunk_index,
                    doc_context,
                    chunks
                )
                chunks.append(chunk)
        
        return chunks
    
    def create_structured_chunk(self, sentences: List[str], section_title: str, section_path: str,
                              section_type: str, level: int, chunk_index: int, 
                              doc_context: Dict, previous_chunks: List[StructuredChunk]) -> StructuredChunk:
        """Create a structured chunk with all metadata"""
        
        content = " ".join(sentences)
        token_count = self.estimate_token_count(content)
        
        # Generate chunk ID
        doc_id = doc_context.get('doc_id', '')
        chunk_content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{doc_id}_{chunk_index:03d}_{chunk_content_hash}"
        
        # Generate previous context
        prev_context = ""
        if previous_chunks:
            last_chunk = previous_chunks[-1]
            last_sentences = nltk.sent_tokenize(last_chunk.content)
            if last_sentences:
                # Take last 1-2 sentences from previous chunk
                context_sentences = last_sentences[-2:] if len(last_sentences) > 1 else last_sentences[-1:]
                prev_context = " ".join(context_sentences)
                if len(prev_context) > 200:  # Limit context length
                    prev_context = prev_context[:200] + "..."
        
        # Generate summary header (placeholder - will be filled by AI)
        summary_header = f"{section_title}: {content[:100]}..." if content else section_title
        
        # Extract structural information
        citations = self.extract_citations(content)
        figures = self.extract_figure_references(content)
        
        # Structural metadata
        structural_metadata = {
            'section_title': section_title,
            'original_section_level': level,
            'section_priority': self.section_types.get(section_type.lower(), {}).get('priority', 10),
            'word_count': len(content.split()),
            'sentence_count': len(sentences),
            'citation_count': len(citations),
            'figure_reference_count': len(figures)
        }
        
        return StructuredChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            prev_context=prev_context,
            section_path=section_path,
            summary_header=summary_header,
            token_count=token_count,
            char_count=len(content),
            chunk_index=chunk_index,
            section_type=section_type,
            hierarchical_level=level,
            contains_figures=figures,
            citations=citations,
            structural_metadata=structural_metadata,
            title=doc_context.get('title', ''),
            url=doc_context.get('url', ''),
            domain=doc_context.get('domain', 'academic_structured'),
            content_type=doc_context.get('content_type', 'pdf'),
            language=doc_context.get('language', 'en'),
            structure_type='academic'
        )
    
    def process_structured_paper(self, structured_paper: Dict) -> List[StructuredChunk]:
        """Process a structured paper (from GROBID+pdffigures2) into chunks"""
        
        doc_id = structured_paper.get('doc_id', '')
        
        doc_context = {
            'doc_id': doc_id,
            'title': structured_paper.get('title', ''),
            'url': structured_paper.get('metadata', {}).get('pdf_file', ''),
            'domain': 'academic_structured',
            'content_type': 'pdf',
            'language': 'en'
        }
        
        chunks = []
        
        # Process abstract separately if available
        abstract = structured_paper.get('abstract', '')
        if abstract and len(abstract.strip()) > 50:
            abstract_chunk = self.create_structured_chunk(
                [abstract],
                "Abstract",
                f"{doc_context['title']} › Abstract",
                "summary",
                1,
                len(chunks),
                doc_context,
                chunks
            )
            chunks.append(abstract_chunk)
        
        # Process structured sections
        sections = structured_paper.get('sections', [])
        for section in sections:
            section_chunks = self.chunk_structured_section(section, doc_context)
            chunks.extend(section_chunks)
        
        # Process references section if substantial
        references = structured_paper.get('references', [])
        if references and len(references) > 10:  # Only if substantial reference list
            ref_content = "\n".join([
                f"[{i+1}] {ref.get('title', '')} {ref.get('authors', [])} {ref.get('venue', '')} {ref.get('year', '')}"
                for i, ref in enumerate(references[:50])  # Limit to first 50 refs
            ])
            
            if ref_content.strip():
                ref_chunk = self.create_structured_chunk(
                    [ref_content],
                    "References",
                    f"{doc_context['title']} › References",
                    "references",
                    1,
                    len(chunks),
                    doc_context,
                    chunks
                )
                chunks.append(ref_chunk)
        
        logger.info(f"✅ Processed {doc_id}: {len(chunks)} structured chunks")
        return chunks
    
    def process_structured_corpus(self, structured_papers_file: str) -> str:
        """Process entire corpus of structured papers"""
        
        logger.info(f"📚 Loading structured papers from {structured_papers_file}")
        
        # Load structured papers
        if structured_papers_file.endswith('.json'):
            with open(structured_papers_file, 'r') as f:
                papers = json.load(f)
        else:  # parquet
            df = pd.read_parquet(structured_papers_file)
            papers = df.to_dict('records')
        
        logger.info(f"🔄 Processing {len(papers)} structured papers...")
        
        all_chunks = []
        
        for i, paper in enumerate(papers):
            logger.info(f"📄 Processing paper {i+1}/{len(papers)}: {paper.get('title', '')[:50]}...")
            
            try:
                paper_chunks = self.process_structured_paper(paper)
                all_chunks.extend(paper_chunks)
            except Exception as e:
                logger.error(f"❌ Failed to process paper {paper.get('doc_id', '')}: {e}")
                continue
        
        # Convert to DataFrame for saving
        chunk_dicts = []
        for chunk in all_chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'content': chunk.content,
                'prev_context': chunk.prev_context,
                'section_path': chunk.section_path,
                'summary_header': chunk.summary_header,
                'token_count': chunk.token_count,
                'char_count': chunk.char_count,
                'chunk_index': chunk.chunk_index,
                'section_type': chunk.section_type,
                'hierarchical_level': chunk.hierarchical_level,
                'contains_figures': json.dumps(chunk.contains_figures),
                'citations': json.dumps(chunk.citations),
                'structural_metadata': json.dumps(chunk.structural_metadata),
                'title': chunk.title,
                'url': chunk.url,
                'domain': chunk.domain,
                'content_type': chunk.content_type,
                'language': chunk.language,
                'structure_type': chunk.structure_type
            }
            chunk_dicts.append(chunk_dict)
        
        # Save chunks
        output_dir = Path(structured_papers_file).parent / "structure_aware_chunks"
        output_dir.mkdir(exist_ok=True)
        
        chunks_df = pd.DataFrame(chunk_dicts)
        output_file = output_dir / "structure_aware_wilson_lin_chunks.parquet"
        chunks_df.to_parquet(output_file, index=False)
        
        # Generate statistics
        stats = {
            'total_papers': len(papers),
            'total_chunks': len(all_chunks),
            'avg_chunks_per_paper': len(all_chunks) / len(papers) if papers else 0,
            'avg_tokens_per_chunk': sum(c.token_count for c in all_chunks) / len(all_chunks) if all_chunks else 0,
            'section_type_distribution': chunks_df['section_type'].value_counts().to_dict(),
            'hierarchical_level_distribution': chunks_df['hierarchical_level'].value_counts().to_dict(),
            'processing_timestamp': datetime.now().isoformat(),
            'token_range_compliance': {
                'in_range': len([c for c in all_chunks if self.target_min_tokens <= c.token_count <= self.target_max_tokens]),
                'below_range': len([c for c in all_chunks if c.token_count < self.target_min_tokens]),
                'above_range': len([c for c in all_chunks if c.token_count > self.target_max_tokens])
            }
        }
        
        stats_file = output_dir / "structure_aware_chunking_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("="*60)
        logger.info("🎯 STRUCTURE-AWARE WILSON LIN CHUNKING COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Papers processed: {len(papers)}")
        logger.info(f"📄 Chunks created: {len(all_chunks)}")
        logger.info(f"🎯 Avg tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
        logger.info(f"💾 Output: {output_file}")
        logger.info(f"📊 Stats: {stats_file}")
        logger.info("="*60)
        
        return str(output_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Structure-aware Wilson Lin chunking")
    parser.add_argument("--structured-papers", required=True, help="Path to structured papers file")
    parser.add_argument("--min-tokens", type=int, default=700, help="Minimum tokens per chunk")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum tokens per chunk")
    
    args = parser.parse_args()
    
    chunker = StructureAwareWilsonLinChunker((args.min_tokens, args.max_tokens))
    
    output_file = chunker.process_structured_corpus(args.structured_papers)
    
    logger.info(f"✅ Structure-aware chunks saved to: {output_file}")

if __name__ == "__main__":
    main()