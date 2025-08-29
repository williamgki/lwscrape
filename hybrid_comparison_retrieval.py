#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict

from colbert_indexer import ColBERTLateInteractionIndexer, ColBERTSearchResult
from docunits_query_engine import DocUnitsQueryEngine, QueryResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Unified search result combining multiple retrieval methods"""
    chunk_id: str
    paper_id: str
    section_id: str
    page: int
    text: str
    figure_refs: List[str]
    chunk_type: str
    
    # Scoring components
    colbert_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    
    # Additional metadata
    source_method: str = "hybrid"
    rank_position: int = 0


class HybridComparisonRetrieval:
    """Multi-layered retrieval system combining ColBERT and traditional methods"""
    
    def __init__(self,
                 colbert_index_name: str = "academic_papers",
                 docunits_db: str = "docunits.db",
                 fusion_method: str = "rrf",  # "rrf", "weighted", "max"
                 weights: Dict[str, float] = None):
        
        # Initialize retrieval components
        self.colbert_retriever = ColBERTLateInteractionIndexer(index_name=colbert_index_name)
        self.docunits_engine = DocUnitsQueryEngine(docunits_db)
        
        # Fusion configuration
        self.fusion_method = fusion_method
        self.weights = weights or {"colbert": 0.7, "bm25": 0.3}
        
        # Load or build indexes
        self._initialize_indexes()
    
    def _initialize_indexes(self):
        """Initialize all retrieval indexes"""
        
        # Try to load existing ColBERT index
        if not self.colbert_retriever.load_existing_index():
            logger.warning("ColBERT index not found. Build index first with colbert_indexer.py")
        
        logger.info("Hybrid retrieval system initialized")
    
    def search_comparison_mode(self,
                             query: str,
                             k: int = 20,
                             enable_colbert: bool = True,
                             enable_bm25: bool = True,
                             paper_filter: Optional[str] = None,
                             section_filter: Optional[str] = None) -> List[HybridSearchResult]:
        """
        Multi-layered search optimized for comparison across long documents
        """
        
        all_results = {}  # chunk_id -> result mapping
        
        # 1. ColBERT late-interaction retrieval (great recall across long docs)
        if enable_colbert and self.colbert_retriever.searcher:
            colbert_results = self.colbert_retriever.search(
                query, k=k*2, 
                filter_paper_id=paper_filter,
                filter_section=section_filter
            )
            
            for result in colbert_results:
                hybrid_result = HybridSearchResult(
                    chunk_id=result.chunk_id,
                    paper_id=result.paper_id,
                    section_id=result.section_id,
                    page=result.page,
                    text=result.text,
                    figure_refs=result.figure_refs,
                    chunk_type=result.chunk_type,
                    colbert_score=result.score,
                    source_method="colbert"
                )
                all_results[result.chunk_id] = hybrid_result
        
        # 2. BM25 full-text search (precise term matching)
        if enable_bm25:
            bm25_results = self.docunits_engine.search_text(
                query, limit=k*2, 
                section_filter=section_filter,
                paper_filter=paper_filter
            )
            
            for result in bm25_results:
                chunk_id = result.para_id
                
                if chunk_id in all_results:
                    # Update existing result
                    all_results[chunk_id].bm25_score = result.score
                else:
                    # Create new hybrid result
                    hybrid_result = HybridSearchResult(
                        chunk_id=chunk_id,
                        paper_id=result.paper_id,
                        section_id=result.section_id,
                        page=result.page,
                        text=result.text,
                        figure_refs=[],  # BM25 doesn't include figure refs
                        chunk_type="text",
                        bm25_score=result.score,
                        source_method="bm25"
                    )
                    all_results[chunk_id] = hybrid_result
        
        # 3. Fusion scoring
        fused_results = self._apply_fusion_scoring(list(all_results.values()))
        
        # 4. Sort and limit results
        fused_results.sort(key=lambda x: x.fusion_score, reverse=True)
        final_results = fused_results[:k]
        
        # Add rank positions
        for i, result in enumerate(final_results):
            result.rank_position = i + 1
        
        logger.info(f"Hybrid search returned {len(final_results)} results for: '{query[:50]}...'")
        return final_results
    
    def _apply_fusion_scoring(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Apply score fusion between different retrieval methods"""
        
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(results)
        elif self.fusion_method == "weighted":
            return self._weighted_score_fusion(results)
        elif self.fusion_method == "max":
            return self._max_score_fusion(results)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}. Using RRF.")
            return self._reciprocal_rank_fusion(results)
    
    def _reciprocal_rank_fusion(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Reciprocal Rank Fusion (RRF) scoring"""
        
        # Sort by each method's scores
        colbert_sorted = sorted([r for r in results if r.colbert_score > 0], 
                               key=lambda x: x.colbert_score, reverse=True)
        bm25_sorted = sorted([r for r in results if r.bm25_score > 0],
                            key=lambda x: x.bm25_score, reverse=True)
        
        # Create rank mappings
        colbert_ranks = {r.chunk_id: i+1 for i, r in enumerate(colbert_sorted)}
        bm25_ranks = {r.chunk_id: i+1 for i, r in enumerate(bm25_sorted)}
        
        # Apply RRF formula: 1/(k + rank)
        k = 60  # RRF parameter
        
        for result in results:
            rrf_score = 0.0
            
            if result.chunk_id in colbert_ranks:
                rrf_score += self.weights["colbert"] / (k + colbert_ranks[result.chunk_id])
            
            if result.chunk_id in bm25_ranks:
                rrf_score += self.weights["bm25"] / (k + bm25_ranks[result.chunk_id])
            
            result.fusion_score = rrf_score
        
        return results
    
    def _weighted_score_fusion(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Weighted linear combination of normalized scores"""
        
        # Normalize scores to [0, 1] range
        if results:
            max_colbert = max((r.colbert_score for r in results if r.colbert_score > 0), default=1.0)
            max_bm25 = max((r.bm25_score for r in results if r.bm25_score > 0), default=1.0)
            
            for result in results:
                norm_colbert = result.colbert_score / max_colbert if max_colbert > 0 else 0
                norm_bm25 = result.bm25_score / max_bm25 if max_bm25 > 0 else 0
                
                result.fusion_score = (
                    self.weights["colbert"] * norm_colbert + 
                    self.weights["bm25"] * norm_bm25
                )
        
        return results
    
    def _max_score_fusion(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """Take maximum normalized score across methods"""
        
        if results:
            max_colbert = max((r.colbert_score for r in results if r.colbert_score > 0), default=1.0)
            max_bm25 = max((r.bm25_score for r in results if r.bm25_score > 0), default=1.0)
            
            for result in results:
                norm_colbert = result.colbert_score / max_colbert if max_colbert > 0 else 0
                norm_bm25 = result.bm25_score / max_bm25 if max_bm25 > 0 else 0
                
                result.fusion_score = max(norm_colbert, norm_bm25)
        
        return results
    
    def find_similar_chunks_across_papers(self,
                                        reference_chunk_id: str,
                                        k: int = 10,
                                        cross_paper_only: bool = True) -> List[HybridSearchResult]:
        """Find similar chunks across different papers using ColBERT"""
        
        if not self.colbert_retriever.searcher:
            logger.error("ColBERT index not available")
            return []
        
        # Use ColBERT for similarity search
        colbert_results = self.colbert_retriever.search_similar_to_chunk(
            reference_chunk_id, k=k, exclude_same_paper=cross_paper_only
        )
        
        # Convert to hybrid results
        hybrid_results = []
        for result in colbert_results:
            hybrid_result = HybridSearchResult(
                chunk_id=result.chunk_id,
                paper_id=result.paper_id,
                section_id=result.section_id,
                page=result.page,
                text=result.text,
                figure_refs=result.figure_refs,
                chunk_type=result.chunk_type,
                colbert_score=result.score,
                fusion_score=result.score,
                source_method="colbert_similarity",
                rank_position=len(hybrid_results) + 1
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def compare_papers_on_topic(self,
                              topic_query: str,
                              paper_ids: List[str],
                              chunks_per_paper: int = 5) -> Dict[str, List[HybridSearchResult]]:
        """Compare specific papers on a given topic"""
        
        comparison_results = {}
        
        for paper_id in paper_ids:
            paper_results = self.search_comparison_mode(
                topic_query,
                k=chunks_per_paper,
                paper_filter=paper_id
            )
            
            comparison_results[paper_id] = paper_results
        
        logger.info(f"Compared {len(paper_ids)} papers on topic: '{topic_query}'")
        return comparison_results
    
    def analyze_cross_paper_patterns(self,
                                   queries: List[str],
                                   min_papers: int = 3) -> Dict[str, Any]:
        """Analyze how different concepts appear across papers"""
        
        analysis = {}
        
        for query in queries:
            results = self.search_comparison_mode(query, k=50)
            
            # Group by paper
            paper_coverage = defaultdict(list)
            for result in results:
                paper_coverage[result.paper_id].append(result)
            
            # Analyze pattern
            pattern_analysis = {
                'total_chunks': len(results),
                'papers_covered': len(paper_coverage),
                'avg_chunks_per_paper': np.mean([len(chunks) for chunks in paper_coverage.values()]),
                'top_papers': sorted(
                    [(pid, len(chunks), np.mean([c.fusion_score for c in chunks])) 
                     for pid, chunks in paper_coverage.items()],
                    key=lambda x: x[2], reverse=True
                )[:5]
            }
            
            analysis[query] = pattern_analysis
        
        return analysis
    
    def export_comparison_results(self,
                                results: List[HybridSearchResult],
                                output_file: Path,
                                include_scores: bool = True):
        """Export comparison results with detailed scoring"""
        
        export_data = []
        for result in results:
            data = {
                'chunk_id': result.chunk_id,
                'paper_id': result.paper_id,
                'section_id': result.section_id,
                'page': result.page,
                'text': result.text[:500] + "..." if len(result.text) > 500 else result.text,
                'chunk_type': result.chunk_type,
                'figure_refs': result.figure_refs,
                'source_method': result.source_method,
                'rank_position': result.rank_position
            }
            
            if include_scores:
                data.update({
                    'colbert_score': result.colbert_score,
                    'bm25_score': result.bm25_score, 
                    'fusion_score': result.fusion_score
                })
            
            export_data.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(results)} comparison results to {output_file}")
    
    def close(self):
        """Clean up resources"""
        if self.docunits_engine:
            self.docunits_engine.close()


def main():
    """Demo hybrid comparison retrieval system"""
    
    # Initialize hybrid retrieval
    retrieval = HybridComparisonRetrieval()
    
    try:
        # Demo comparison searches
        test_queries = [
            "artificial intelligence alignment safety",
            "large language model capabilities emergent behavior", 
            "reinforcement learning human feedback RLHF",
            "neural network interpretability mechanistic"
        ]
        
        print("=== Hybrid Comparison Retrieval Demo ===")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)
            
            # Hybrid search
            results = retrieval.search_comparison_mode(query, k=5)
            
            for result in results:
                print(f"Rank {result.rank_position}: {result.paper_id}")
                print(f"  Section: {result.section_id} | Page: {result.page}")
                print(f"  Scores - ColBERT: {result.colbert_score:.3f}, BM25: {result.bm25_score:.3f}, Fusion: {result.fusion_score:.3f}")
                print(f"  Type: {result.chunk_type} | Method: {result.source_method}")
                print(f"  Text: {result.text[:200]}...")
                if result.figure_refs:
                    print(f"  Figures: {result.figure_refs}")
                print()
        
        # Cross-paper pattern analysis
        print("\n=== Cross-Paper Pattern Analysis ===")
        analysis_queries = ["safety alignment", "emergent capabilities"]
        patterns = retrieval.analyze_cross_paper_patterns(analysis_queries)
        
        for query, pattern in patterns.items():
            print(f"\nTopic: {query}")
            print(f"  Papers covered: {pattern['papers_covered']}")
            print(f"  Total chunks: {pattern['total_chunks']}")
            print(f"  Avg chunks/paper: {pattern['avg_chunks_per_paper']:.1f}")
            print("  Top papers:")
            for paper_id, chunk_count, avg_score in pattern['top_papers'][:3]:
                print(f"    {paper_id}: {chunk_count} chunks, avg score {avg_score:.3f}")
    
    finally:
        retrieval.close()


if __name__ == "__main__":
    main()