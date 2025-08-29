#!/usr/bin/env python3
"""
Parallel BGE-M3 Embedding Builder for LessWrong Contextual Chunks
Uses multiple workers to dramatically speed up embedding generation
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import time
from dataclasses import dataclass

# ML libraries
import faiss
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingBatch:
    """Batch of texts and their embeddings"""
    batch_id: int
    start_idx: int
    end_idx: int
    texts: List[str]
    embeddings: Optional[np.ndarray] = None

def worker_embed_batch(args):
    """
    Worker function to embed a batch of texts
    Args: (batch_id, texts, model_name, device)
    """
    batch_id, texts, model_name, device = args
    
    try:
        # Load model in worker process
        model = SentenceTransformer(model_name, device=device)
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        return {
            'batch_id': batch_id,
            'embeddings': embeddings,
            'success': True,
            'num_texts': len(texts)
        }
        
    except Exception as e:
        logger.error(f"Worker {batch_id} failed: {e}")
        return {
            'batch_id': batch_id,
            'embeddings': None,
            'success': False,
            'error': str(e),
            'num_texts': len(texts)
        }

class ParallelEmbeddingBuilder:
    """
    Build BGE-M3 embeddings using multiple parallel workers
    """
    
    def __init__(self, 
                 chunks_path: str = "chunked_corpus/contextual_chunks_complete.parquet",
                 output_dir: str = "retrieval_indexes",
                 model_name: str = "BAAI/bge-m3",
                 batch_size: int = 100,  # Larger batches for efficiency
                 num_workers: int = None):
        
        self.chunks_path = chunks_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect optimal workers (leave 2 cores for system)
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 2)
        else:
            self.num_workers = num_workers
            
        logger.info(f"🚀 Parallel Embedding Builder initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Workers: {self.num_workers}")
        logger.info(f"   Batch size: {batch_size}")
        
    def load_chunks(self) -> pd.DataFrame:
        """Load contextual chunks"""
        logger.info(f"📚 Loading chunks from {self.chunks_path}")
        
        chunks_df = pd.read_parquet(self.chunks_path)
        logger.info(f"✅ Loaded {len(chunks_df)} contextual chunks")
        
        return chunks_df
        
    def prepare_texts(self, chunks_df: pd.DataFrame) -> List[str]:
        """Prepare searchable texts from chunks"""
        logger.info("📝 Preparing searchable texts...")
        
        texts = []
        for _, row in chunks_df.iterrows():
            # Combine content, section path, and summary for better search
            searchable_text = f"{row['content']} {row.get('section_path', '')} {row.get('summary_header', '')}"
            texts.append(searchable_text.strip())
            
        logger.info(f"✅ Prepared {len(texts)} searchable texts")
        return texts
        
    def create_batches(self, texts: List[str]) -> List[EmbeddingBatch]:
        """Split texts into batches for parallel processing"""
        batches = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch = EmbeddingBatch(
                batch_id=len(batches),
                start_idx=i,
                end_idx=min(i + self.batch_size, len(texts)),
                texts=batch_texts
            )
            batches.append(batch)
            
        logger.info(f"📦 Created {len(batches)} batches of ~{self.batch_size} texts each")
        return batches
        
    def build_embeddings_parallel(self, texts: List[str], force_rebuild: bool = False) -> np.ndarray:
        """Build embeddings using parallel workers"""
        
        embeddings_path = self.output_dir / "bge_m3_embeddings_parallel.pkl"
        
        # Check if already exists
        if embeddings_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing embeddings...")
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            return embeddings
            
        logger.info(f"🔨 Building embeddings with {self.num_workers} parallel workers...")
        
        # Create batches
        batches = self.create_batches(texts)
        total_batches = len(batches)
        
        # Prepare worker arguments (batch_id, texts, model_name, device)
        worker_args = []
        for batch in batches:
            # Distribute across CPU (no GPU parallelization with sentence-transformers)
            worker_args.append((batch.batch_id, batch.texts, self.model_name, "cpu"))
            
        # Process batches in parallel
        start_time = time.time()
        completed_batches = {}
        
        logger.info(f"⚡ Starting parallel embedding generation...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            future_to_batch = {
                executor.submit(worker_embed_batch, args): args[0] 
                for args in worker_args
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        completed_batches[batch_id] = result['embeddings']
                        
                        # Progress logging
                        completed_count = len(completed_batches)
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        eta = (total_batches - completed_count) / rate if rate > 0 else 0
                        
                        logger.info(f"✅ Batch {batch_id:4d}/{total_batches} complete "
                                   f"({completed_count/total_batches*100:.1f}%) | "
                                   f"Rate: {rate:.2f} batches/sec | "
                                   f"ETA: {eta/60:.1f}min")
                    else:
                        logger.error(f"❌ Batch {batch_id} failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_id}: {e}")
                    
        # Combine all embeddings in order
        logger.info("📋 Combining embeddings from all batches...")
        
        all_embeddings = []
        for batch_id in range(total_batches):
            if batch_id in completed_batches:
                all_embeddings.append(completed_batches[batch_id])
            else:
                logger.error(f"Missing embeddings for batch {batch_id}")
                
        if len(all_embeddings) != total_batches:
            logger.warning(f"Only {len(all_embeddings)}/{total_batches} batches completed successfully")
            
        # Stack into final array
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        elapsed = time.time() - start_time
        texts_per_sec = len(texts) / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Parallel embedding generation complete!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Time: {elapsed/60:.1f} minutes")
        logger.info(f"   Rate: {texts_per_sec:.1f} texts/second")
        logger.info(f"   Speedup: ~{self.num_workers}x vs single worker")
        
        return embeddings
        
    def build_faiss_index(self, embeddings: np.ndarray, force_rebuild: bool = False):
        """Build FAISS HNSW index from embeddings"""
        faiss_path = self.output_dir / "bge_m3_faiss_parallel.index"
        
        if faiss_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing FAISS index...")
            index = faiss.read_index(str(faiss_path))
            logger.info(f"✅ FAISS index loaded: {index.ntotal} vectors")
            return index
            
        logger.info("🏗️  Building FAISS HNSW index...")
        
        # Build HNSW index
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
        index.hnsw.ef_construction = 200  # Higher quality construction
        index.hnsw.ef_search = 100  # Search quality
        
        # Add vectors to index
        logger.info(f"📈 Adding {len(embeddings)} vectors to FAISS index...")
        index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, str(faiss_path))
        
        logger.info(f"✅ FAISS index built and saved: {index.ntotal} vectors")
        return index
        
    def build_complete_index(self, force_rebuild: bool = False):
        """Build complete embedding index (embeddings + FAISS)"""
        logger.info("🚀 Starting complete parallel index build...")
        
        # Load chunks
        chunks_df = self.load_chunks()
        
        # Prepare texts
        texts = self.prepare_texts(chunks_df)
        
        # Build embeddings with parallel workers
        embeddings = self.build_embeddings_parallel(texts, force_rebuild)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings, force_rebuild)
        
        logger.info("🎉 Complete parallel index build finished!")
        logger.info(f"   Chunks: {len(chunks_df):,}")
        logger.info(f"   Embeddings: {embeddings.shape}")
        logger.info(f"   FAISS vectors: {index.ntotal}")
        
        return embeddings, index

def main():
    """Main execution"""
    
    # Initialize parallel builder
    builder = ParallelEmbeddingBuilder(
        chunks_path="chunked_corpus/contextual_chunks_complete.parquet",
        batch_size=200,  # Larger batches for efficiency
        num_workers=14   # Use 14 of 16 cores
    )
    
    # Build complete index
    embeddings, index = builder.build_complete_index(force_rebuild=True)
    
    # Test search
    logger.info("🧪 Testing search functionality...")
    
    # Load model for query encoding
    model = SentenceTransformer("BAAI/bge-m3")
    
    test_queries = [
        "AI alignment and safety research",
        "machine learning interpretability"
    ]
    
    for query in test_queries:
        query_embedding = model.encode([query], convert_to_numpy=True)
        scores, indices = index.search(query_embedding.astype('float32'), 5)
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Top results: {indices[0]}")
        logger.info(f"Scores: {scores[0]}")

if __name__ == "__main__":
    main()