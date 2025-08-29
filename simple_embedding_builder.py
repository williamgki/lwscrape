#!/usr/bin/env python3
"""
Simple Single-Process BGE-M3 Embedding Builder
No threading, no multiprocessing - just reliable sequential processing with progress tracking
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
import time

# ML libraries
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleEmbeddingBuilder:
    """
    Simple, reliable embedding builder - no parallelization complications
    """
    
    def __init__(self, 
                 chunks_path: str = "chunked_corpus/contextual_chunks_complete.parquet",
                 output_dir: str = "retrieval_indexes",
                 model_name: str = "BAAI/bge-m3",
                 batch_size: int = 16):  # Small batches for memory efficiency
        
        self.chunks_path = chunks_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info(f"🔧 Simple Embedding Builder initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Batch size: {batch_size}")
        
    def build_embeddings_simple(self, texts, force_rebuild=False):
        """Build embeddings with simple sequential processing"""
        
        embeddings_path = self.output_dir / "bge_m3_embeddings_simple.pkl"
        progress_path = self.output_dir / "embedding_progress.json"
        
        # Check if already exists
        if embeddings_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing embeddings...")
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            return embeddings
            
        logger.info(f"🤖 Loading BGE-M3 model...")
        model = SentenceTransformer(self.model_name)
        logger.info(f"✅ Model loaded - Max tokens: {model.max_seq_length}")
        
        # Process in batches with progress tracking
        total_texts = len(texts)
        num_batches = (total_texts + self.batch_size - 1) // self.batch_size
        
        logger.info(f"📦 Processing {total_texts} texts in {num_batches} batches")
        
        all_embeddings = []
        start_time = time.time()
        
        # Process each batch sequentially
        for i in tqdm(range(0, total_texts, self.batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Progress logging every 100 batches
                batch_num = i // self.batch_size + 1
                if batch_num % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = batch_num / elapsed if elapsed > 0 else 0
                    eta = (num_batches - batch_num) / rate if rate > 0 else 0
                    
                    logger.info(f"✅ Batch {batch_num:4d}/{num_batches} complete "
                               f"({batch_num/num_batches*100:.1f}%) | "
                               f"Rate: {rate:.2f} batches/sec | "
                               f"ETA: {eta/60:.1f}min")
                    
                    # Save progress checkpoint every 200 batches
                    if batch_num % 200 == 0:
                        partial_embeddings = np.vstack(all_embeddings)
                        checkpoint_path = self.output_dir / f"embeddings_checkpoint_{batch_num}.pkl"
                        with open(checkpoint_path, 'wb') as f:
                            pickle.dump(partial_embeddings, f)
                        logger.info(f"💾 Checkpoint saved at batch {batch_num}")
                        
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_num}: {e}")
                # Continue with next batch rather than failing completely
                continue
                
        # Combine all embeddings
        logger.info("📋 Combining all embeddings...")
        embeddings = np.vstack(all_embeddings)
        
        # Save final embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        elapsed = time.time() - start_time
        texts_per_sec = total_texts / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Simple embedding generation complete!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Time: {elapsed/60:.1f} minutes")
        logger.info(f"   Rate: {texts_per_sec:.1f} texts/second")
        
        return embeddings
        
    def build_faiss_index(self, embeddings, force_rebuild=False):
        """Build FAISS index"""
        faiss_path = self.output_dir / "bge_m3_faiss_simple.index"
        
        if faiss_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing FAISS index...")
            index = faiss.read_index(str(faiss_path))
            logger.info(f"✅ FAISS index loaded: {index.ntotal} vectors")
            return index
            
        logger.info("🏗️  Building FAISS HNSW index...")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.ef_construction = 200
        index.hnsw.ef_search = 100
        
        logger.info(f"📈 Adding {len(embeddings)} vectors...")
        index.add(embeddings.astype('float32'))
        
        faiss.write_index(index, str(faiss_path))
        
        logger.info(f"✅ FAISS index built: {index.ntotal} vectors")
        return index
        
    def build_complete_index(self, force_rebuild=False):
        """Build complete index with simple approach"""
        logger.info("🚀 Starting simple index build...")
        
        # Load and prepare data
        logger.info(f"📚 Loading chunks...")
        chunks_df = pd.read_parquet(self.chunks_path)
        logger.info(f"✅ Loaded {len(chunks_df)} chunks")
        
        texts = []
        for _, row in chunks_df.iterrows():
            text = f"{row['content']} {row.get('section_path', '')} {row.get('summary_header', '')}"
            texts.append(text.strip())
        logger.info(f"✅ Prepared {len(texts)} texts")
        
        # Build embeddings (this will take time but be reliable)
        embeddings = self.build_embeddings_simple(texts, force_rebuild)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings, force_rebuild)
        
        logger.info("🎉 Simple index build complete!")
        return embeddings, index

def main():
    """Main execution with simple approach"""
    
    builder = SimpleEmbeddingBuilder(batch_size=8)  # Very small batches
    embeddings, index = builder.build_complete_index(force_rebuild=True)
    
    # Test
    logger.info("🧪 Quick test...")
    model = SentenceTransformer("BAAI/bge-m3") 
    query = model.encode(["artificial intelligence"], convert_to_numpy=True)
    scores, indices = index.search(query.astype('float32'), 3)
    logger.info(f"Test results: {indices[0]} scores: {scores[0]}")

if __name__ == "__main__":
    main()