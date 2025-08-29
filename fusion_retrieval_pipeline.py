#!/usr/bin/env python3

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import time

from colbert_indexer import ColBERTLateInteractionIndexer
from bge_cross_encoder_reranker import BGECrossEncoderReranker
from production_chunking_config import ProductionChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Result after multi-stage fusion pipeline"""
    chunk_id: str
    paper_id: str
    page_from: int
    page_to: int
    text: str
    headers: List[str]
    figure_ids: List[str]
    caption_text: str
    
    # Multi-stage scores
    colbert_score: float = 0.0
    bm25_score: float = 0.0
    splade_score: float = 0.0
    rrf_score: float = 0.0
    reranker_score: float = 0.0
    final_score: float = 0.0
    
    # Rankings at each stage
    colbert_rank: int = 0
    bm25_rank: int = 0
    splade_rank: int = 0
    rrf_rank: int = 0
    final_rank: int = 0
    
    # Metadata
    length_tokens: int = 0
    chunk_type: str = "text"
    retrieval_stage: str = "fusion"


class SPLADERetriever:
    """SPLADE sparse retrieval implementation"""
    
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        # This would be implemented with actual SPLADE model
        # For now, placeholder that uses BM25-like scoring
        logger.info(f"SPLADE retriever initialized: {model_name}")
    
    def search(self, query: str, k: int = 400) -> List[Dict]:
        """SPLADE sparse retrieval (placeholder implementation)"""
        # In production, this would use actual SPLADE model
        # For now, return placeholder results
        logger.info(f"SPLADE search for: '{query[:50]}...' (k={k})")
        return []


class ProductionFusionPipeline:
    """Complete fusion pipeline: ColBERT + BM25 + SPLADE → RRF → BGE reranker"""
    
    def __init__(self,
                 colbert_index_name: str = "production_papers",
                 chunks_storage_path: str = "./production_chunks",
                 bge_model: str = "BAAI/bge-reranker-v2-m3"):
        
        # Initialize retrievers
        self.colbert_retriever = ColBERTLateInteractionIndexer(
            index_name=colbert_index_name
        )
        self.splade_retriever = SPLADERetriever()
        self.bge_reranker = BGECrossEncoderReranker(model_name=bge_model)
        
        # Storage
        self.chunks_storage = Path(chunks_storage_path)
        
        # Load chunk metadata for BM25/SPLADE
        self.chunk_metadata = self._load_chunk_metadata()
        
        # RRF parameters
        self.rrf_k = 60  # Standard RRF parameter
        
        logger.info("Production fusion pipeline initialized")
    
    def _load_chunk_metadata(self) -> Dict[str, ProductionChunk]:
        """Load all chunk metadata for retrieval"""
        metadata = {}
        
        if not self.chunks_storage.exists():
            return metadata
        
        chunk_files = list((self.chunks_storage / "chunks").glob("*_chunks.json"))
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r') as f:
                    chunks_data = json.load(f)
                
                for chunk_data in chunks_data:
                    chunk_id = chunk_data['chunk_id']
                    chunk = ProductionChunk(
                        chunk_id=chunk_id,
                        paper_id=chunk_data['paper_id'],
                        page_from=chunk_data['page_from'],
                        page_to=chunk_data['page_to'],
                        text=chunk_data['text'],
                        headers=chunk_data['headers'],
                        figure_ids=chunk_data['figure_ids'],
                        caption_text=chunk_data.get('caption_text', ''),
                        ref_inline_markers=chunk_data.get('ref_inline_markers', []),
                        length_tokens=chunk_data['length_tokens'],
                        source_docunit_ids=chunk_data.get('source_docunit_ids', []),
                        chunk_type=chunk_data.get('chunk_type', 'text'),
                        bm25_terms=chunk_data.get('bm25_terms'),
                        splade_sparse=chunk_data.get('splade_sparse')
                    )
                    metadata[chunk_id] = chunk
            
            except Exception as e:
                logger.warning(f"Failed to load chunk metadata from {chunk_file}: {e}")
        
        logger.info(f"Loaded metadata for {len(metadata)} chunks")
        return metadata
    
    def bm25_search(self, query: str, k: int = 400) -> List[Dict]:
        """BM25 search using precomputed terms"""
        results = []
        
        query_terms = query.lower().split()
        
        for chunk_id, chunk in self.chunk_metadata.items():
            if not chunk.bm25_terms:
                continue
            
            # Simple BM25-like scoring
            score = 0.0
            for term in query_terms:
                if term in chunk.bm25_terms:
                    score += chunk.bm25_terms[term]
            
            if score > 0:
                results.append({
                    'chunk_id': chunk_id,
                    'score': score,
                    'method': 'bm25'
                })
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"BM25 search returned {len(results[:k])} results")
        return results[:k]
    
    def splade_search(self, query: str, k: int = 400) -> List[Dict]:
        """SPLADE search using sparse representations"""
        # Placeholder - would use actual SPLADE model
        return self.splade_retriever.search(query, k)
    
    def reciprocal_rank_fusion(self, 
                              colbert_results: List[Dict],
                              bm25_results: List[Dict], 
                              splade_results: List[Dict],
                              k_final: int = 300) -> List[Dict]:
        """Apply RRF fusion to combine multiple retrieval methods"""
        
        # Create rank mappings
        colbert_ranks = {r['chunk_id']: i+1 for i, r in enumerate(colbert_results)}
        bm25_ranks = {r['chunk_id']: i+1 for i, r in enumerate(bm25_results)}
        splade_ranks = {r['chunk_id']: i+1 for i, r in enumerate(splade_results)}
        
        # Collect all unique chunk IDs
        all_chunk_ids = set()
        all_chunk_ids.update(colbert_ranks.keys())
        all_chunk_ids.update(bm25_ranks.keys())
        all_chunk_ids.update(splade_ranks.keys())
        
        # Calculate RRF scores
        rrf_results = []
        
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            
            # RRF formula: sum(1 / (k + rank)) for each method
            if chunk_id in colbert_ranks:
                rrf_score += 1.0 / (self.rrf_k + colbert_ranks[chunk_id])
            
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (self.rrf_k + bm25_ranks[chunk_id])
            
            if chunk_id in splade_ranks:
                rrf_score += 1.0 / (self.rrf_k + splade_ranks[chunk_id])
            
            # Store individual scores and ranks
            result = {
                'chunk_id': chunk_id,
                'rrf_score': rrf_score,
                'colbert_score': next((r['score'] for r in colbert_results if r['chunk_id'] == chunk_id), 0.0),
                'bm25_score': next((r['score'] for r in bm25_results if r['chunk_id'] == chunk_id), 0.0),
                'splade_score': next((r['score'] for r in splade_results if r['chunk_id'] == chunk_id), 0.0),
                'colbert_rank': colbert_ranks.get(chunk_id, 0),
                'bm25_rank': bm25_ranks.get(chunk_id, 0),
                'splade_rank': splade_ranks.get(chunk_id, 0),
                'method': 'rrf_fusion'
            }
            
            rrf_results.append(result)
        
        # Sort by RRF score and return top-k
        rrf_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        final_results = rrf_results[:k_final]
        
        logger.info(f"RRF fusion: {len(final_results)} results from {len(all_chunk_ids)} candidates")
        return final_results
    
    async def full_fusion_search(self,
                                query: str,
                                colbert_k: int = 400,
                                bm25_k: int = 400,
                                splade_k: int = 400,
                                rrf_k: int = 300,
                                rerank_k: int = 100) -> List[FusedResult]:
        """Complete fusion pipeline"""
        
        logger.info(f"Starting full fusion search: '{query[:50]}...'")
        start_time = time.time()
        
        # Stage 1: Multi-method retrieval (parallel)
        tasks = []
        
        # ColBERT retrieval
        if self.colbert_retriever.searcher:
            colbert_task = asyncio.create_task(
                self._async_colbert_search(query, colbert_k)
            )
            tasks.append(('colbert', colbert_task))
        
        # BM25 retrieval  
        bm25_task = asyncio.create_task(
            self._async_bm25_search(query, bm25_k)
        )
        tasks.append(('bm25', bm25_task))
        
        # SPLADE retrieval
        splade_task = asyncio.create_task(
            self._async_splade_search(query, splade_k)
        )
        tasks.append(('splade', splade_task))
        
        # Execute retrieval in parallel
        retrieval_results = {}
        for method, task in tasks:
            try:
                retrieval_results[method] = await task
            except Exception as e:
                logger.warning(f"{method} retrieval failed: {e}")
                retrieval_results[method] = []
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieval stage completed in {retrieval_time:.2f}s")
        
        # Stage 2: RRF Fusion
        fusion_start = time.time()
        fused_results = self.reciprocal_rank_fusion(
            retrieval_results.get('colbert', []),
            retrieval_results.get('bm25', []),
            retrieval_results.get('splade', []),
            k_final=rrf_k
        )
        
        fusion_time = time.time() - fusion_start
        logger.info(f"RRF fusion completed in {fusion_time:.2f}s")
        
        # Stage 3: BGE Reranking
        if fused_results:
            rerank_start = time.time()
            
            # Convert to format expected by reranker
            candidate_chunks = []
            for result in fused_results:
                chunk_id = result['chunk_id']
                if chunk_id in self.chunk_metadata:
                    chunk = self.chunk_metadata[chunk_id]
                    
                    # Create pseudo HybridSearchResult for reranker
                    from hybrid_comparison_retrieval import HybridSearchResult
                    pseudo_result = HybridSearchResult(
                        chunk_id=chunk_id,
                        paper_id=chunk.paper_id,
                        section_id=" > ".join(chunk.headers[-2:]) if len(chunk.headers) >= 2 else chunk.headers[-1] if chunk.headers else "unknown",
                        page=(chunk.page_from + chunk.page_to) // 2,
                        text=chunk.text,
                        figure_refs=chunk.figure_ids,
                        chunk_type=chunk.chunk_type,
                        colbert_score=result['colbert_score'],
                        bm25_score=result['bm25_score'],
                        fusion_score=result['rrf_score']
                    )
                    candidate_chunks.append(pseudo_result)
            
            # Apply BGE reranking
            reranked_results = self.bge_reranker.rerank_candidates(
                query=query,
                candidates=candidate_chunks,
                max_candidates=min(200, len(candidate_chunks)),
                return_top_k=rerank_k
            )
            
            rerank_time = time.time() - rerank_start
            logger.info(f"BGE reranking completed in {rerank_time:.2f}s")
            
            # Convert to final format
            final_results = []
            for i, reranked in enumerate(reranked_results):
                chunk_id = reranked.chunk_id
                chunk = self.chunk_metadata[chunk_id]
                
                # Find original fusion result for scores
                fusion_result = next((r for r in fused_results if r['chunk_id'] == chunk_id), {})
                
                fused_result = FusedResult(
                    chunk_id=chunk_id,
                    paper_id=chunk.paper_id,
                    page_from=chunk.page_from,
                    page_to=chunk.page_to,
                    text=chunk.text,
                    headers=chunk.headers,
                    figure_ids=chunk.figure_ids,
                    caption_text=chunk.caption_text,
                    colbert_score=fusion_result.get('colbert_score', 0.0),
                    bm25_score=fusion_result.get('bm25_score', 0.0),
                    splade_score=fusion_result.get('splade_score', 0.0),
                    rrf_score=fusion_result.get('rrf_score', 0.0),
                    reranker_score=reranked.reranker_score,
                    final_score=reranked.reranker_score,
                    colbert_rank=fusion_result.get('colbert_rank', 0),
                    bm25_rank=fusion_result.get('bm25_rank', 0),
                    splade_rank=fusion_result.get('splade_rank', 0),
                    rrf_rank=next((j+1 for j, r in enumerate(fused_results) if r['chunk_id'] == chunk_id), 0),
                    final_rank=i + 1,
                    length_tokens=chunk.length_tokens,
                    chunk_type=chunk.chunk_type,
                    retrieval_stage="final"
                )
                final_results.append(fused_result)
        
        else:
            final_results = []
        
        total_time = time.time() - start_time
        logger.info(f"Full fusion pipeline completed in {total_time:.2f}s")
        logger.info(f"Final results: {len(final_results)}")
        
        return final_results
    
    async def _async_colbert_search(self, query: str, k: int) -> List[Dict]:
        """Async wrapper for ColBERT search"""
        colbert_results = self.colbert_retriever.search(query, k)
        return [
            {'chunk_id': r.chunk_id, 'score': r.score, 'method': 'colbert'}
            for r in colbert_results
        ]
    
    async def _async_bm25_search(self, query: str, k: int) -> List[Dict]:
        """Async wrapper for BM25 search"""
        return self.bm25_search(query, k)
    
    async def _async_splade_search(self, query: str, k: int) -> List[Dict]:
        """Async wrapper for SPLADE search"""
        return self.splade_search(query, k)
    
    def export_fusion_results(self, 
                            results: List[FusedResult],
                            output_file: Path,
                            include_full_pipeline: bool = True):
        """Export fusion results with complete pipeline details"""
        
        export_data = {
            'results': [],
            'pipeline_config': {
                'colbert_k': 400,
                'bm25_k': 400, 
                'splade_k': 400,
                'rrf_k': 300,
                'final_k': len(results),
                'rrf_parameter': self.rrf_k
            },
            'metadata': {
                'total_results': len(results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for result in results:
            result_data = {
                'chunk_id': result.chunk_id,
                'paper_id': result.paper_id,
                'page_range': f"{result.page_from}-{result.page_to}",
                'headers': result.headers,
                'text_preview': result.text[:300] + "..." if len(result.text) > 300 else result.text,
                'figure_ids': result.figure_ids,
                'caption_text': result.caption_text[:200] + "..." if len(result.caption_text) > 200 else result.caption_text,
                'length_tokens': result.length_tokens,
                'chunk_type': result.chunk_type,
                'final_rank': result.final_rank
            }
            
            if include_full_pipeline:
                result_data['scores'] = {
                    'colbert': result.colbert_score,
                    'bm25': result.bm25_score,
                    'splade': result.splade_score,
                    'rrf': result.rrf_score,
                    'reranker': result.reranker_score,
                    'final': result.final_score
                }
                result_data['ranks'] = {
                    'colbert': result.colbert_rank,
                    'bm25': result.bm25_rank,
                    'splade': result.splade_rank,
                    'rrf': result.rrf_rank,
                    'final': result.final_rank
                }
            
            export_data['results'].append(result_data)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported fusion results to {output_file}")


async def main():
    """Demo production fusion pipeline"""
    
    pipeline = ProductionFusionPipeline()
    
    # Demo queries
    test_queries = [
        "artificial intelligence alignment safety mechanisms",
        "large language model emergent capabilities scaling laws",
        "reinforcement learning human feedback RLHF training"
    ]
    
    print("=== Production Fusion Pipeline Demo ===")
    print("Pipeline: ColBERT (k=400) + BM25 (k=400) + SPLADE (k=400) → RRF (k=300) → BGE (k=50-100)")
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 80)
        
        results = await pipeline.full_fusion_search(
            query=query,
            colbert_k=400,
            bm25_k=400,
            splade_k=400,
            rrf_k=300,
            rerank_k=50
        )
        
        print(f"Final results: {len(results)}")
        
        for result in results[:5]:  # Show top 5
            print(f"\nRank {result.final_rank}: {result.paper_id}")
            print(f"  Pages: {result.page_from}-{result.page_to} | Tokens: {result.length_tokens}")
            print(f"  Headers: {' > '.join(result.headers[-2:])}")
            print(f"  Scores - RRF: {result.rrf_score:.4f}, BGE: {result.reranker_score:.3f}")
            if result.figure_ids:
                print(f"  Figures: {result.figure_ids}")
            if result.caption_text:
                print(f"  Caption: {result.caption_text[:100]}...")
            print(f"  Text: {result.text[:150]}...")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())