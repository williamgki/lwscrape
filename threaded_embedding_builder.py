#!/usr/bin/env python3
"""
Threaded BGE-M3 Embedding Builder - More reliable than multiprocessing
Uses ThreadPoolExecutor for parallel embedding generation
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import pickle
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# ML libraries
import faiss
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BatchResult:
    """Result from embedding a batch"""
    batch_id: int
    embeddings: np.ndarray
    success: bool
    error: Optional[str] = None

class ThreadedEmbeddingBuilder:
    """
    Build BGE-M3 embeddings using threaded approach
    More reliable than multiprocessing for this use case
    """
    
    def __init__(self, 
                 chunks_path: str = "chunked_corpus/contextual_chunks_complete.parquet",
                 output_dir: str = "retrieval_indexes",
                 model_name: str = "BAAI/bge-m3",
                 batch_size: int = 64,
                 num_threads: int = 4):  # Conservative thread count
        
        self.chunks_path = chunks_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_threads = num_threads
        
        # Load model once (shared across threads)
        self.model = None
        
        logger.info(f"🧵 Threaded Embedding Builder initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Threads: {num_threads}")
        logger.info(f"   Batch size: {batch_size}")
        
    def load_model(self):
        """Load BGE-M3 model once"""
        if self.model is None:
            logger.info(f"🤖 Loading BGE-M3 model...")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Model loaded - Max tokens: {self.model.max_seq_length}")
            
    def embed_batch(self, batch_id: int, texts: List[str]) -> BatchResult:
        """Embed a single batch of texts"""
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,  # Internal batch size
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            
            return BatchResult(
                batch_id=batch_id,
                embeddings=embeddings,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_id} failed: {e}")
            return BatchResult(
                batch_id=batch_id,
                embeddings=None,
                success=False,
                error=str(e)
            )
            
    def build_embeddings_threaded(self, texts: List[str], force_rebuild: bool = False) -> np.ndarray:
        """Build embeddings using threaded approach"""
        
        embeddings_path = self.output_dir / "bge_m3_embeddings_threaded.pkl"
        
        # Check if already exists
        if embeddings_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing embeddings...")
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            return embeddings
            
        # Load model
        self.load_model()
        
        # Create batches
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch_texts))
            
        total_batches = len(batches)
        logger.info(f"📦 Created {total_batches} batches of ~{self.batch_size} texts each")
        
        # Process batches with thread pool
        start_time = time.time()
        completed_results = {}
        
        logger.info(f"⚡ Starting threaded embedding generation...")
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.embed_batch, batch_id, batch_texts): batch_id
                for batch_id, batch_texts in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    result = future.result()
                    
                    if result.success:
                        completed_results[batch_id] = result.embeddings
                        
                        # Progress logging
                        completed_count = len(completed_results)
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        eta = (total_batches - completed_count) / rate if rate > 0 else 0
                        
                        logger.info(f"✅ Batch {batch_id:4d}/{total_batches} complete "
                                   f"({completed_count/total_batches*100:.1f}%) | "
                                   f"Rate: {rate:.2f} batches/sec | "
                                   f"ETA: {eta/60:.1f}min")
                    else:
                        logger.error(f"❌ Batch {batch_id} failed: {result.error}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_id}: {e}")
                    
        # Combine embeddings in order
        logger.info("📋 Combining embeddings from all batches...")
        
        all_embeddings = []
        missing_batches = []
        
        for batch_id in range(total_batches):
            if batch_id in completed_results:
                all_embeddings.append(completed_results[batch_id])
            else:
                missing_batches.append(batch_id)
                
        if missing_batches:
            logger.warning(f"Missing embeddings for batches: {missing_batches}")
            
        # Stack into final array
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        elapsed = time.time() - start_time
        texts_per_sec = len(texts) / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Threaded embedding generation complete!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Time: {elapsed/60:.1f} minutes")
        logger.info(f"   Rate: {texts_per_sec:.1f} texts/second")
        logger.info(f"   Success: {len(all_embeddings)}/{total_batches} batches")
        
        return embeddings
        
    def build_faiss_index(self, embeddings: np.ndarray, force_rebuild: bool = False):
        """Build FAISS HNSW index"""
        faiss_path = self.output_dir / "bge_m3_faiss_threaded.index"
        
        if faiss_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing FAISS index...")
            index = faiss.read_index(str(faiss_path))
            logger.info(f"✅ FAISS index loaded: {index.ntotal} vectors")
            return index
            
        logger.info("🏗️  Building FAISS HNSW index...")
        
        # Build index
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.ef_construction = 200
        index.hnsw.ef_search = 100
        
        # Add vectors
        logger.info(f"📈 Adding {len(embeddings)} vectors to FAISS index...")
        index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, str(faiss_path))
        
        logger.info(f"✅ FAISS index built: {index.ntotal} vectors")
        return index
        
    def build_complete_index(self, force_rebuild: bool = False):
        """Build complete embedding index"""
        logger.info("🚀 Starting threaded index build...")
        
        # Load chunks
        logger.info(f"📚 Loading chunks from {self.chunks_path}")
        chunks_df = pd.read_parquet(self.chunks_path)
        logger.info(f"✅ Loaded {len(chunks_df)} chunks")
        
        # Prepare texts
        logger.info("📝 Preparing searchable texts...")
        texts = []
        for _, row in chunks_df.iterrows():
            searchable_text = f"{row['content']} {row.get('section_path', '')} {row.get('summary_header', '')}"
            texts.append(searchable_text.strip())
        logger.info(f"✅ Prepared {len(texts)} texts")
        
        # Build embeddings
        embeddings = self.build_embeddings_threaded(texts, force_rebuild)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings, force_rebuild)
        
        logger.info("🎉 Complete threaded index build finished!")
        return embeddings, index

def main():
    """Main execution with conservative settings"""
    
    builder = ThreadedEmbeddingBuilder(
        batch_size=32,    # Smaller batches for stability
        num_threads=8     # Conservative thread count
    )
    
    embeddings, index = builder.build_complete_index(force_rebuild=True)
    
    # Quick test
    logger.info("🧪 Testing search...")
    model = SentenceTransformer("BAAI/bge-m3")
    query_emb = model.encode(["AI alignment"], convert_to_numpy=True)
    scores, indices = index.search(query_emb.astype('float32'), 3)
    logger.info(f"Test search results: {indices[0]} with scores: {scores[0]}")

if __name__ == "__main__":
    main()