#!/usr/bin/env python3

import logging
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import time

from hybrid_comparison_retrieval import HybridSearchResult, HybridComparisonRetrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Result after BGE cross-encoder reranking"""
    chunk_id: str
    paper_id: str
    section_id: str
    page: int
    text: str
    figure_refs: List[str]
    chunk_type: str
    
    # Original retrieval scores
    colbert_score: float = 0.0
    bm25_score: float = 0.0
    fusion_score: float = 0.0
    
    # Reranking scores
    reranker_score: float = 0.0
    final_score: float = 0.0
    
    # Rankings
    initial_rank: int = 0
    final_rank: int = 0
    rank_change: int = 0


class BGECrossEncoderReranker:
    """BGE-based cross-encoder reranker for precision at the top"""
    
    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 8192,
                 batch_size: int = 16,
                 device: str = None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing BGE reranker: {model_name} on {self.device}")
        
        # Load cross-encoder model
        self.model = CrossEncoder(
            model_name, 
            max_length=max_length,
            device=self.device
        )
        
        logger.info("BGE cross-encoder loaded successfully")
    
    def prepare_reranking_pairs(self, 
                               query: str,
                               candidates: List[HybridSearchResult],
                               max_candidates: int = 200) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for reranking"""
        
        # Limit candidates for efficiency
        top_candidates = candidates[:max_candidates]
        
        pairs = []
        for candidate in top_candidates:
            # Prepare text for reranking
            # Include paper context for better relevance assessment
            doc_text = self._prepare_document_text(candidate)
            
            # Create query-document pair
            pairs.append((query, doc_text))
        
        logger.info(f"Prepared {len(pairs)} query-document pairs for reranking")
        return pairs
    
    def _prepare_document_text(self, candidate: HybridSearchResult) -> str:
        """Prepare document text with context for reranking"""
        
        # Start with main text
        doc_parts = [candidate.text]
        
        # Add section context
        if candidate.section_id and candidate.section_id != "unknown":
            doc_parts.insert(0, f"Section: {candidate.section_id}")
        
        # Add figure context if available
        if candidate.figure_refs:
            figure_context = f"References figures: {', '.join(candidate.figure_refs)}"
            doc_parts.append(figure_context)
        
        # Add paper context
        if candidate.paper_id:
            doc_parts.insert(0, f"Paper: {candidate.paper_id}")
        
        # Combine parts
        full_text = "\n".join(doc_parts)
        
        # Truncate if too long (leave room for query)
        max_doc_length = self.max_length - 200  # Reserve space for query
        if len(full_text) > max_doc_length:
            full_text = full_text[:max_doc_length] + "..."
        
        return full_text
    
    def rerank_candidates(self,
                         query: str,
                         candidates: List[HybridSearchResult],
                         max_candidates: int = 200,
                         return_top_k: int = 50) -> List[RerankedResult]:
        """Rerank candidates using BGE cross-encoder"""
        
        if not candidates:
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates (limit: {max_candidates})")
        
        # Prepare query-document pairs
        pairs = self.prepare_reranking_pairs(query, candidates, max_candidates)
        top_candidates = candidates[:len(pairs)]
        
        # Batch reranking for efficiency
        reranker_scores = self._batch_rerank(pairs)
        
        # Create reranked results
        reranked_results = []
        
        for i, (candidate, score) in enumerate(zip(top_candidates, reranker_scores)):
            reranked_result = RerankedResult(
                chunk_id=candidate.chunk_id,
                paper_id=candidate.paper_id,
                section_id=candidate.section_id,
                page=candidate.page,
                text=candidate.text,
                figure_refs=candidate.figure_refs,
                chunk_type=candidate.chunk_type,
                colbert_score=candidate.colbert_score,
                bm25_score=candidate.bm25_score,
                fusion_score=candidate.fusion_score,
                reranker_score=float(score),
                final_score=float(score),  # Use reranker score as final
                initial_rank=candidate.rank_position,
                final_rank=0  # Will be set after sorting
            )
            reranked_results.append(reranked_result)
        
        # Sort by reranker score
        reranked_results.sort(key=lambda x: x.reranker_score, reverse=True)
        
        # Update final ranks and rank changes
        for i, result in enumerate(reranked_results):
            result.final_rank = i + 1
            result.rank_change = result.initial_rank - result.final_rank
        
        # Return top-k results
        final_results = reranked_results[:return_top_k]
        
        logger.info(f"Reranking complete. Returning top {len(final_results)} results")
        return final_results
    
    def _batch_rerank(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Perform batch reranking for efficiency"""
        
        all_scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Reranking batches"):
            batch = pairs[i:i + self.batch_size]
            
            try:
                # Get batch scores
                batch_scores = self.model.predict(batch)
                all_scores.extend(batch_scores.tolist())
                
            except Exception as e:
                logger.warning(f"Batch reranking error: {e}. Using fallback scores.")
                # Fallback to zero scores for this batch
                all_scores.extend([0.0] * len(batch))
        
        return all_scores
    
    def analyze_reranking_impact(self, 
                               results: List[RerankedResult],
                               top_k: int = 20) -> Dict[str, Any]:
        """Analyze the impact of reranking on result ordering"""
        
        analysis = {
            'total_results': len(results),
            'top_k_analyzed': min(top_k, len(results)),
            'avg_rank_change': 0.0,
            'significant_improvements': 0,  # Moved up >10 positions
            'significant_degradations': 0,  # Moved down >10 positions
            'score_correlation': 0.0,
            'reranking_stats': {}
        }
        
        if not results:
            return analysis
        
        top_results = results[:top_k]
        
        # Rank change analysis
        rank_changes = [r.rank_change for r in top_results]
        analysis['avg_rank_change'] = np.mean(rank_changes)
        analysis['significant_improvements'] = sum(1 for rc in rank_changes if rc > 10)
        analysis['significant_degradations'] = sum(1 for rc in rank_changes if rc < -10)
        
        # Score analysis
        initial_scores = [r.fusion_score for r in top_results]
        reranker_scores = [r.reranker_score for r in top_results]
        
        # Correlation between initial and reranker scores
        if len(initial_scores) > 1:
            correlation_matrix = np.corrcoef(initial_scores, reranker_scores)
            analysis['score_correlation'] = correlation_matrix[0, 1]
        
        # Reranking statistics
        analysis['reranking_stats'] = {
            'avg_reranker_score': np.mean(reranker_scores),
            'max_reranker_score': max(reranker_scores),
            'min_reranker_score': min(reranker_scores),
            'score_std': np.std(reranker_scores)
        }
        
        return analysis


class PipelineWithReranking:
    """Complete retrieval pipeline with BGE reranking"""
    
    def __init__(self,
                 hybrid_retrieval: HybridComparisonRetrieval,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        
        self.retrieval = hybrid_retrieval
        self.reranker = BGECrossEncoderReranker(model_name=reranker_model)
    
    def search_with_reranking(self,
                            query: str,
                            initial_k: int = 200,
                            final_k: int = 20,
                            enable_colbert: bool = True,
                            enable_bm25: bool = True,
                            paper_filter: Optional[str] = None,
                            section_filter: Optional[str] = None) -> List[RerankedResult]:
        """Complete search pipeline with reranking"""
        
        logger.info(f"Starting search with reranking: '{query[:50]}...'")
        
        # Step 1: Initial retrieval (get more candidates for reranking)
        initial_results = self.retrieval.search_comparison_mode(
            query=query,
            k=initial_k,
            enable_colbert=enable_colbert,
            enable_bm25=enable_bm25,
            paper_filter=paper_filter,
            section_filter=section_filter
        )
        
        if not initial_results:
            logger.warning("No initial results found")
            return []
        
        logger.info(f"Retrieved {len(initial_results)} initial candidates")
        
        # Step 2: Cross-encoder reranking
        reranked_results = self.reranker.rerank_candidates(
            query=query,
            candidates=initial_results,
            max_candidates=min(200, len(initial_results)),  # BGE can handle ~200 efficiently
            return_top_k=final_k
        )
        
        return reranked_results
    
    def batch_search_with_reranking(self,
                                   queries: List[str],
                                   initial_k: int = 200,
                                   final_k: int = 20) -> Dict[str, List[RerankedResult]]:
        """Batch search with reranking for multiple queries"""
        
        results = {}
        
        for query in tqdm(queries, desc="Processing queries"):
            query_results = self.search_with_reranking(
                query=query,
                initial_k=initial_k,
                final_k=final_k
            )
            results[query] = query_results
        
        return results
    
    def compare_with_without_reranking(self,
                                     query: str,
                                     k: int = 20) -> Dict[str, Any]:
        """Compare results with and without reranking"""
        
        # Without reranking
        without_reranking = self.retrieval.search_comparison_mode(query, k=k)
        
        # With reranking
        with_reranking = self.search_with_reranking(
            query, initial_k=200, final_k=k
        )
        
        # Analysis
        comparison = {
            'query': query,
            'without_reranking': len(without_reranking),
            'with_reranking': len(with_reranking),
            'overlap': 0,
            'new_in_top_k': 0,
            'reranking_impact': {}
        }
        
        if with_reranking:
            # Calculate overlap
            without_ids = {r.chunk_id for r in without_reranking}
            with_ids = {r.chunk_id for r in with_reranking}
            
            comparison['overlap'] = len(without_ids & with_ids)
            comparison['new_in_top_k'] = len(with_ids - without_ids)
            
            # Reranking impact analysis
            comparison['reranking_impact'] = self.reranker.analyze_reranking_impact(
                with_reranking, k
            )
        
        return comparison
    
    def export_reranked_results(self,
                              results: List[RerankedResult],
                              output_file: Path,
                              include_analysis: bool = True):
        """Export reranked results with detailed scoring"""
        
        export_data = {
            'results': [],
            'metadata': {
                'total_results': len(results),
                'model': self.reranker.model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Export individual results
        for result in results:
            result_data = {
                'chunk_id': result.chunk_id,
                'paper_id': result.paper_id,
                'section_id': result.section_id,
                'page': result.page,
                'text': result.text[:500] + "..." if len(result.text) > 500 else result.text,
                'chunk_type': result.chunk_type,
                'figure_refs': result.figure_refs,
                'scores': {
                    'colbert': result.colbert_score,
                    'bm25': result.bm25_score,
                    'fusion': result.fusion_score,
                    'reranker': result.reranker_score,
                    'final': result.final_score
                },
                'ranking': {
                    'initial_rank': result.initial_rank,
                    'final_rank': result.final_rank,
                    'rank_change': result.rank_change
                }
            }
            export_data['results'].append(result_data)
        
        # Add analysis if requested
        if include_analysis and results:
            export_data['analysis'] = self.reranker.analyze_reranking_impact(results)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported reranked results to {output_file}")


def main():
    """Demo BGE cross-encoder reranking"""
    
    # Initialize retrieval pipeline with reranking
    hybrid_retrieval = HybridComparisonRetrieval()
    pipeline = PipelineWithReranking(hybrid_retrieval)
    
    try:
        # Demo queries
        test_queries = [
            "artificial intelligence safety alignment mechanisms",
            "large language model emergent capabilities scaling",
            "reinforcement learning human feedback RLHF techniques"
        ]
        
        print("=== BGE Cross-Encoder Reranking Demo ===")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("=" * 80)
            
            # Search with reranking
            results = pipeline.search_with_reranking(
                query, initial_k=100, final_k=10
            )
            
            print(f"Top {len(results)} reranked results:")
            
            for result in results[:5]:  # Show top 5
                print(f"\nRank {result.final_rank}: {result.paper_id}")
                print(f"  Section: {result.section_id} | Page: {result.page}")
                print(f"  Initial Rank: {result.initial_rank} → Final Rank: {result.final_rank}")
                print(f"  Rank Change: {result.rank_change:+d}")
                print(f"  Scores - Fusion: {result.fusion_score:.3f}, Reranker: {result.reranker_score:.3f}")
                print(f"  Text: {result.text[:200]}...")
                if result.figure_refs:
                    print(f"  Figures: {result.figure_refs}")
            
            # Reranking impact analysis
            if results:
                analysis = pipeline.reranker.analyze_reranking_impact(results)
                print(f"\nReranking Impact:")
                print(f"  Avg rank change: {analysis['avg_rank_change']:.1f}")
                print(f"  Significant improvements: {analysis['significant_improvements']}")
                print(f"  Score correlation: {analysis['score_correlation']:.3f}")
            
            print("\n" + "-" * 80)
    
    finally:
        hybrid_retrieval.close()


if __name__ == "__main__":
    main()