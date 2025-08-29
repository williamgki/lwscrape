#!/usr/bin/env python3
"""
Mac-Optimized BGE-M3 Embedding Builder
Designed for Mac hardware with memory-efficient processing and Apple Silicon support
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
import time
import argparse
import platform
import psutil

# ML libraries
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MacOptimizedEmbedder:
    """
    Mac-optimized embedding builder with memory management and Apple Silicon support
    """
    
    def __init__(self, 
                 chunks_path: str = "chunked_corpus/contextual_chunks_complete.parquet",
                 output_dir: str = "retrieval_indexes",
                 model_name: str = "BAAI/bge-m3",
                 batch_size: int = 32,
                 use_mps: bool = True):
        
        self.chunks_path = chunks_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Detect Mac hardware
        self.system_info = self._detect_system()
        self.device = self._select_device(use_mps)
        
        logger.info(f"🍎 Mac-Optimized Embedder initialized")
        logger.info(f"   System: {self.system_info}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Available RAM: {psutil.virtual_memory().available / 1e9:.1f}GB")
        
    def _detect_system(self):
        """Detect Mac system information"""
        system = platform.system()
        machine = platform.machine()
        
        if machine == "arm64":
            return "Apple Silicon Mac (M1/M2/M3)"
        elif machine == "x86_64" and system == "Darwin":
            return "Intel Mac"
        else:
            return f"{system} {machine}"
            
    def _select_device(self, use_mps):
        """Select optimal device for Mac"""
        if use_mps and torch.backends.mps.is_available():
            logger.info("🚀 Using Apple Silicon MPS acceleration")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("🚀 Using CUDA acceleration") 
            return "cuda"
        else:
            logger.info("⚙️ Using CPU processing")
            return "cpu"
            
    def _check_memory_available(self, required_gb=8):
        """Check if sufficient memory is available"""
        available_gb = psutil.virtual_memory().available / 1e9
        if available_gb < required_gb:
            logger.warning(f"⚠️  Low memory: {available_gb:.1f}GB available, {required_gb}GB recommended")
            logger.warning("Consider closing other applications or reducing batch size")
            return False
        return True
        
    def _save_checkpoint(self, embeddings_so_far, batch_num):
        """Save embedding checkpoint"""
        checkpoint_path = self.output_dir / f"mac_checkpoint_batch_{batch_num}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(embeddings_so_far, f)
        logger.info(f"💾 Checkpoint saved at batch {batch_num}")
        
    def build_embeddings_mac(self, texts, force_rebuild=False):
        """Build embeddings optimized for Mac hardware"""
        
        embeddings_path = self.output_dir / "bge_m3_embeddings_mac.pkl"
        
        # Check existing
        if embeddings_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing Mac embeddings...")
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            return embeddings
            
        # Memory check
        estimated_memory_gb = len(texts) * 1024 * 4 / 1e9  # Rough estimate
        if not self._check_memory_available(estimated_memory_gb + 4):  # +4GB for model
            logger.error("❌ Insufficient memory. Please close applications or reduce corpus size.")
            return None
            
        # Load model
        logger.info(f"🤖 Loading BGE-M3 model on {self.device}")
        model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"✅ Model loaded - Max tokens: {model.max_seq_length}")
        
        # Process with memory monitoring
        total_texts = len(texts)
        num_batches = (total_texts + self.batch_size - 1) // self.batch_size
        
        logger.info(f"📦 Processing {total_texts:,} texts in {num_batches:,} batches")
        logger.info(f"🧮 Estimated memory usage: {estimated_memory_gb:.1f}GB")
        
        all_embeddings = []
        start_time = time.time()
        
        # Progress tracking
        progress_bar = tqdm(total=num_batches, desc="Mac Embedding Progress")
        
        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                # Monitor memory before batch
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90:
                    logger.warning(f"⚠️  High memory usage: {memory_percent:.1f}%")
                    
                # Generate embeddings
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=min(self.batch_size, len(batch_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                
                all_embeddings.append(batch_embeddings)
                progress_bar.update(1)
                
                # Progress logging and checkpoints
                if batch_num % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = batch_num / elapsed if elapsed > 0 else 0
                    eta = (num_batches - batch_num) / rate if rate > 0 else 0
                    
                    logger.info(f"✅ Batch {batch_num:,}/{num_batches:,} "
                               f"({batch_num/num_batches*100:.1f}%) | "
                               f"Rate: {rate:.2f} batches/sec | "
                               f"ETA: {eta/3600:.1f}h | "
                               f"Memory: {psutil.virtual_memory().percent:.1f}%")
                    
                # Save checkpoint every 500 batches
                if batch_num % 500 == 0:
                    partial_embeddings = np.vstack(all_embeddings)
                    self._save_checkpoint(partial_embeddings, batch_num)
                    
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_num}: {e}")
                # Try to continue with smaller batch size
                if self.batch_size > 8:
                    logger.info("🔄 Reducing batch size and retrying...")
                    self.batch_size = max(8, self.batch_size // 2)
                continue
                
        progress_bar.close()
        
        # Combine embeddings
        logger.info("🔗 Combining all embeddings...")
        embeddings = np.vstack(all_embeddings)
        
        # Save final embeddings
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        # Performance summary
        elapsed = time.time() - start_time
        texts_per_sec = total_texts / elapsed if elapsed > 0 else 0
        
        logger.info(f"🎉 Mac embedding generation complete!")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Time: {elapsed/3600:.2f} hours")
        logger.info(f"   Rate: {texts_per_sec:.1f} texts/second")
        logger.info(f"   Device: {self.device}")
        
        return embeddings
        
    def build_faiss_index(self, embeddings, force_rebuild=False):
        """Build FAISS index optimized for Mac"""
        faiss_path = self.output_dir / "bge_m3_faiss_mac.index"
        
        if faiss_path.exists() and not force_rebuild:
            logger.info("📖 Loading existing Mac FAISS index...")
            index = faiss.read_index(str(faiss_path))
            logger.info(f"✅ FAISS index loaded: {index.ntotal:,} vectors")
            return index
            
        logger.info("🏗️  Building Mac-optimized FAISS HNSW index...")
        
        dimension = embeddings.shape[1]
        
        # Mac-optimized HNSW parameters
        index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
        index.hnsw.ef_construction = 200  # High quality construction
        index.hnsw.ef_search = 100  # Search quality
        
        logger.info(f"📈 Adding {len(embeddings):,} vectors to index...")
        
        # Add in chunks to avoid memory spikes on Mac
        chunk_size = 10000
        for i in tqdm(range(0, len(embeddings), chunk_size), desc="Building FAISS"):
            chunk = embeddings[i:i + chunk_size].astype('float32')
            index.add(chunk)
            
        # Save index
        faiss.write_index(index, str(faiss_path))
        
        logger.info(f"✅ Mac FAISS index built: {index.ntotal:,} vectors")
        return index
        
    def build_complete_mac_system(self, force_rebuild=False):
        """Build complete embedding system optimized for Mac"""
        logger.info("🍎 Starting Mac-optimized index build...")
        
        # Load data
        logger.info(f"📚 Loading chunks from {self.chunks_path}")
        chunks_df = pd.read_parquet(self.chunks_path)
        logger.info(f"✅ Loaded {len(chunks_df):,} chunks")
        
        # Prepare texts
        logger.info("📝 Preparing searchable texts...")
        texts = []
        for _, row in chunks_df.iterrows():
            text = f"{row['content']} {row.get('section_path', '')} {row.get('summary_header', '')}"
            texts.append(text.strip())
        logger.info(f"✅ Prepared {len(texts):,} texts")
        
        # Build embeddings
        embeddings = self.build_embeddings_mac(texts, force_rebuild)
        if embeddings is None:
            return None, None
            
        # Build FAISS index
        index = self.build_faiss_index(embeddings, force_rebuild)
        
        logger.info("🎉 Mac-optimized system complete!")
        return embeddings, index

def main():
    parser = argparse.ArgumentParser(description='Mac-Optimized BGE-M3 Embedder')
    parser.add_argument('--mode', choices=['single', 'dual', 'mps', 'conservative'], 
                       default='single', help='Processing mode')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--no-mps', action='store_true', help='Disable MPS acceleration')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild')
    
    args = parser.parse_args()
    
    # Set batch size based on mode
    batch_sizes = {
        'conservative': 8,
        'single': 32, 
        'dual': 64,
        'mps': 128
    }
    
    if args.batch_size == 32:  # Default not overridden
        batch_size = batch_sizes.get(args.mode, 32)
    else:
        batch_size = args.batch_size
        
    # Initialize embedder
    embedder = MacOptimizedEmbedder(
        batch_size=batch_size,
        use_mps=not args.no_mps
    )
    
    # Build system
    embeddings, index = embedder.build_complete_mac_system(force_rebuild=args.force_rebuild)
    
    if embeddings is not None:
        # Quick test
        logger.info("🧪 Testing search functionality...")
        model = SentenceTransformer("BAAI/bge-m3", device=embedder.device)
        query = model.encode(["artificial intelligence safety"], convert_to_numpy=True)
        scores, indices = index.search(query.astype('float32'), 5)
        
        logger.info(f"✅ Test search successful!")
        logger.info(f"   Top results: {indices[0]}")
        logger.info(f"   Scores: {scores[0]}")
        logger.info("🚀 Mac hybrid retrieval system ready!")

if __name__ == "__main__":
    main()