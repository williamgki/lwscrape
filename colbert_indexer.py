#!/usr/bin/env python3

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import pandas as pd

# ColBERT imports
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

from colbert_semantic_chunker import SemanticChunk, ColBERTSemanticChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColBERTSearchResult:
    """Search result from ColBERT retrieval"""
    chunk_id: str
    paper_id: str
    section_id: str
    page: int
    score: float
    text: str
    figure_refs: List[str]
    chunk_type: str


class ColBERTLateInteractionIndexer:
    """ColBERT indexing and retrieval for academic papers"""
    
    def __init__(self, 
                 index_name: str = "academic_papers",
                 model_name: str = "colbert-ir/colbertv2.0",
                 index_root: str = "./colbert_indexes"):
        
        self.index_name = index_name
        self.model_name = model_name
        self.index_root = Path(index_root)
        self.index_root.mkdir(exist_ok=True)
        
        # ColBERT configuration
        self.config = ColBERTConfig(
            nbits=2,  # Compression level
            kmeans_niters=4,
            nranks=1,  # Single GPU/CPU
            index_root=str(self.index_root),
            experiment="academic_retrieval"
        )
        
        # Chunk metadata mapping
        self.chunk_metadata: Dict[str, Dict] = {}
        self.chunk_id_to_pid: Dict[str, int] = {}  # ColBERT passage IDs
        
        # Initialize searcher (will be set after indexing)
        self.searcher: Optional[Searcher] = None
    
    def prepare_collection_from_chunks(self, 
                                     chunks: List[SemanticChunk]) -> Tuple[List[str], str]:
        """Prepare ColBERT collection from semantic chunks"""
        
        collection = []
        metadata = {}
        
        for i, chunk in enumerate(chunks):
            # Store chunk text for ColBERT
            collection.append(chunk.chunk_text)
            
            # Store metadata for result reconstruction
            metadata[chunk.chunk_id] = {
                'paper_id': chunk.paper_id,
                'section_id': chunk.section_id, 
                'page': chunk.page,
                'chunk_type': chunk.chunk_type,
                'figure_refs': chunk.figure_refs,
                'attached_captions': chunk.attached_captions,
                'source_para_ids': chunk.source_para_ids
            }
            
            # Map chunk_id to ColBERT passage ID
            self.chunk_id_to_pid[chunk.chunk_id] = i
            
        self.chunk_metadata = metadata
        
        # Save collection to file
        collection_path = self.index_root / f"{self.index_name}_collection.tsv"
        with open(collection_path, 'w') as f:
            for i, text in enumerate(collection):
                # ColBERT expects format: pid \t passage_text
                f.write(f"{i}\t{text}\n")
        
        logger.info(f"Prepared collection with {len(collection)} chunks")
        return collection, str(collection_path)
    
    def build_index(self, chunks: List[SemanticChunk]) -> None:
        """Build ColBERT index from semantic chunks"""
        
        logger.info(f"Building ColBERT index for {len(chunks)} chunks")
        
        # Prepare collection
        collection_texts, collection_path = self.prepare_collection_from_chunks(chunks)
        
        # Configure indexing run
        with Run().context(RunConfig(nranks=1, experiment="academic_retrieval")):
            
            # Initialize indexer
            indexer = Indexer(
                checkpoint=self.model_name,
                config=self.config
            )
            
            # Build index
            indexer.index(
                name=self.index_name,
                collection=collection_path,
                overwrite=True
            )
        
        # Save metadata
        metadata_path = self.index_root / f"{self.index_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'chunk_metadata': self.chunk_metadata,
                'chunk_id_to_pid': self.chunk_id_to_pid,
                'model_name': self.model_name,
                'total_chunks': len(chunks)
            }, f, indent=2)
        
        logger.info(f"ColBERT index built successfully: {self.index_name}")
        
        # Initialize searcher
        self._initialize_searcher()
    
    def _initialize_searcher(self):
        """Initialize ColBERT searcher"""
        try:
            with Run().context(RunConfig(nranks=1, experiment="academic_retrieval")):
                self.searcher = Searcher(
                    index=self.index_name,
                    config=self.config
                )
            logger.info("ColBERT searcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize searcher: {e}")
    
    def load_existing_index(self) -> bool:
        """Load existing ColBERT index if available"""
        try:
            # Load metadata
            metadata_path = self.index_root / f"{self.index_name}_metadata.json"
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.chunk_metadata = metadata['chunk_metadata']
            self.chunk_id_to_pid = metadata['chunk_id_to_pid']
            
            # Initialize searcher
            self._initialize_searcher()
            
            if self.searcher:
                logger.info(f"Loaded existing ColBERT index: {self.index_name}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
        
        return False
    
    def search(self, 
              query: str, 
              k: int = 20,
              filter_paper_id: Optional[str] = None,
              filter_section: Optional[str] = None) -> List[ColBERTSearchResult]:
        """Search using ColBERT late-interaction"""
        
        if not self.searcher:
            logger.error("Searcher not initialized. Build or load index first.")
            return []
        
        try:
            # Perform ColBERT search
            results = self.searcher.search(query, k=k*2)  # Get extra results for filtering
            
            search_results = []
            
            for passage_id, rank, score in zip(*results):
                # Find chunk_id from passage_id
                chunk_id = None
                for cid, pid in self.chunk_id_to_pid.items():
                    if pid == passage_id:
                        chunk_id = cid
                        break
                
                if not chunk_id or chunk_id not in self.chunk_metadata:
                    continue
                
                metadata = self.chunk_metadata[chunk_id]
                
                # Apply filters
                if filter_paper_id and metadata['paper_id'] != filter_paper_id:
                    continue
                    
                if filter_section and filter_section.lower() not in metadata['section_id'].lower():
                    continue
                
                # Get chunk text (reconstruct from collection if needed)
                chunk_text = self._get_chunk_text(passage_id)
                
                result = ColBERTSearchResult(
                    chunk_id=chunk_id,
                    paper_id=metadata['paper_id'],
                    section_id=metadata['section_id'],
                    page=metadata['page'],
                    score=float(score),
                    text=chunk_text,
                    figure_refs=metadata['figure_refs'],
                    chunk_type=metadata['chunk_type']
                )
                
                search_results.append(result)
                
                if len(search_results) >= k:
                    break
            
            logger.info(f"Found {len(search_results)} results for query: '{query[:50]}...'")
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _get_chunk_text(self, passage_id: int) -> str:
        """Retrieve chunk text by passage ID"""
        try:
            collection_path = self.index_root / f"{self.index_name}_collection.tsv"
            if not collection_path.exists():
                return ""
            
            # Read specific line from collection file
            with open(collection_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == passage_id:
                        parts = line.strip().split('\t', 1)
                        if len(parts) > 1:
                            return parts[1]
            
        except Exception as e:
            logger.warning(f"Failed to retrieve chunk text for passage {passage_id}: {e}")
        
        return ""
    
    def search_similar_to_chunk(self, 
                              reference_chunk_id: str,
                              k: int = 10,
                              exclude_same_paper: bool = True) -> List[ColBERTSearchResult]:
        """Find chunks similar to a reference chunk"""
        
        if reference_chunk_id not in self.chunk_metadata:
            logger.error(f"Reference chunk not found: {reference_chunk_id}")
            return []
        
        # Get reference chunk text
        ref_pid = self.chunk_id_to_pid[reference_chunk_id]
        ref_text = self._get_chunk_text(ref_pid)
        
        if not ref_text:
            return []
        
        # Search using reference chunk text as query
        results = self.search(ref_text, k=k*2)
        
        # Filter out the reference chunk itself and same paper if requested
        ref_paper_id = self.chunk_metadata[reference_chunk_id]['paper_id']
        
        filtered_results = []
        for result in results:
            if result.chunk_id == reference_chunk_id:
                continue
                
            if exclude_same_paper and result.paper_id == ref_paper_id:
                continue
                
            filtered_results.append(result)
            
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def get_paper_chunks(self, paper_id: str) -> List[ColBERTSearchResult]:
        """Get all chunks for a specific paper"""
        
        results = []
        
        for chunk_id, metadata in self.chunk_metadata.items():
            if metadata['paper_id'] == paper_id:
                pid = self.chunk_id_to_pid[chunk_id]
                text = self._get_chunk_text(pid)
                
                result = ColBERTSearchResult(
                    chunk_id=chunk_id,
                    paper_id=metadata['paper_id'],
                    section_id=metadata['section_id'],
                    page=metadata['page'],
                    score=1.0,  # No scoring for direct retrieval
                    text=text,
                    figure_refs=metadata['figure_refs'],
                    chunk_type=metadata['chunk_type']
                )
                results.append(result)
        
        # Sort by page and section
        results.sort(key=lambda x: (x.page, x.section_id))
        return results
    
    def analyze_index_coverage(self) -> Dict[str, Any]:
        """Analyze the coverage and distribution of the index"""
        
        stats = {
            'total_chunks': len(self.chunk_metadata),
            'unique_papers': len(set(m['paper_id'] for m in self.chunk_metadata.values())),
            'chunk_types': {},
            'papers_with_figures': 0,
            'avg_chunks_per_paper': 0
        }
        
        # Analyze chunk types
        for metadata in self.chunk_metadata.values():
            chunk_type = metadata['chunk_type']
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        # Papers with figures
        papers_with_figs = set()
        for metadata in self.chunk_metadata.values():
            if metadata['figure_refs']:
                papers_with_figs.add(metadata['paper_id'])
        
        stats['papers_with_figures'] = len(papers_with_figs)
        
        # Average chunks per paper
        if stats['unique_papers'] > 0:
            stats['avg_chunks_per_paper'] = stats['total_chunks'] / stats['unique_papers']
        
        return stats
    
    def export_search_results(self, 
                            results: List[ColBERTSearchResult],
                            output_file: Path,
                            include_full_text: bool = False):
        """Export search results to file"""
        
        export_data = []
        for result in results:
            data = {
                'chunk_id': result.chunk_id,
                'paper_id': result.paper_id,
                'section_id': result.section_id,
                'page': result.page,
                'score': result.score,
                'chunk_type': result.chunk_type,
                'figure_refs': result.figure_refs
            }
            
            if include_full_text:
                data['text'] = result.text
            else:
                data['text_preview'] = result.text[:200] + "..." if len(result.text) > 200 else result.text
            
            export_data.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(results)} search results to {output_file}")


def main():
    """Demo ColBERT indexing and search"""
    
    # Initialize chunker and indexer
    chunker = ColBERTSemanticChunker()
    indexer = ColBERTLateInteractionIndexer()
    
    # Try to load existing index
    if not indexer.load_existing_index():
        logger.info("No existing index found. Creating semantic chunks...")
        
        # Create chunks from DocUnits
        chunks = chunker.chunk_from_docunits_db("docunits.db")
        
        if not chunks:
            logger.error("No chunks created. Check DocUnits database.")
            return
        
        # Build ColBERT index
        indexer.build_index(chunks)
    
    # Demo searches
    queries = [
        "artificial intelligence safety alignment",
        "large language model capabilities",
        "reinforcement learning human feedback",
        "neural network interpretability"
    ]
    
    print("\n=== ColBERT Search Demo ===")
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = indexer.search(query, k=5)
        
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. Paper: {result.paper_id} | Section: {result.section_id}")
            print(f"   Score: {result.score:.3f} | Type: {result.chunk_type}")
            print(f"   Text: {result.text[:150]}...")
            if result.figure_refs:
                print(f"   Figures: {result.figure_refs}")
            print()
    
    # Index statistics
    stats = indexer.analyze_index_coverage()
    print("=== Index Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()