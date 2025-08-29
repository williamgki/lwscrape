#!/usr/bin/env python3
"""
Split Corpus Parallel Embedder
Pre-split corpus into chunks, run independent processes, then combine
Much simpler than complex threading/multiprocessing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
import time
import sys
import argparse

# ML libraries
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusSplitter:
    """Split corpus into independent chunks for parallel processing"""
    
    def __init__(self, chunks_path: str, num_splits: int = 4):
        self.chunks_path = chunks_path
        self.num_splits = num_splits
        self.output_dir = Path("corpus_splits")
        self.output_dir.mkdir(exist_ok=True)
        
    def split_corpus(self):
        """Split corpus into N independent files"""
        logger.info(f"📚 Loading full corpus from {self.chunks_path}")
        chunks_df = pd.read_parquet(self.chunks_path)
        total_chunks = len(chunks_df)
        
        logger.info(f"✂️  Splitting {total_chunks} chunks into {self.num_splits} parts")
        
        chunk_size = total_chunks // self.num_splits
        split_files = []
        
        for i in range(self.num_splits):
            start_idx = i * chunk_size
            if i == self.num_splits - 1:  # Last split gets remainder
                end_idx = total_chunks
            else:
                end_idx = (i + 1) * chunk_size
                
            split_df = chunks_df.iloc[start_idx:end_idx]
            split_file = self.output_dir / f"corpus_part_{i:02d}.parquet"
            split_df.to_parquet(split_file)
            split_files.append(split_file)
            
            logger.info(f"  Part {i}: {len(split_df)} chunks → {split_file}")
            
        logger.info(f"✅ Corpus split into {len(split_files)} files")
        return split_files

class SinglePartEmbedder:
    """Embed a single part of the corpus"""
    
    def __init__(self, part_id: int, corpus_file: str, batch_size: int = 32):
        self.part_id = part_id
        self.corpus_file = corpus_file
        self.batch_size = batch_size
        self.model_name = "BAAI/bge-m3"
        
        self.output_dir = Path("embedding_parts")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔧 Part {part_id} embedder initialized")
        
    def embed_part(self):
        """Embed this part of the corpus"""
        embeddings_file = self.output_dir / f"embeddings_part_{self.part_id:02d}.pkl"
        
        # Load this part
        logger.info(f"📖 Loading corpus part {self.part_id}")
        chunks_df = pd.read_parquet(self.corpus_file)
        
        # Prepare texts
        texts = []
        for _, row in chunks_df.iterrows():
            text = f"{row['content']} {row.get('section_path', '')} {row.get('summary_header', '')}"
            texts.append(text.strip())
            
        logger.info(f"✅ Prepared {len(texts)} texts for part {self.part_id}")
        
        # Load model
        logger.info(f"🤖 Loading BGE-M3 model for part {self.part_id}")
        model = SentenceTransformer(self.model_name)
        logger.info(f"✅ Model loaded for part {self.part_id}")
        
        # Process in batches
        all_embeddings = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        
        logger.info(f"🚀 Starting embedding generation for part {self.part_id} ({num_batches} batches)")
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     desc=f"Part {self.part_id} batches"):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
                
                # Progress every 50 batches
                batch_num = i // self.batch_size + 1
                if batch_num % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = batch_num / elapsed if elapsed > 0 else 0
                    eta = (num_batches - batch_num) / rate if rate > 0 else 0
                    
                    logger.info(f"Part {self.part_id}: Batch {batch_num}/{num_batches} "
                               f"({batch_num/num_batches*100:.1f}%) ETA: {eta/60:.1f}min")
                    
            except Exception as e:
                logger.error(f"❌ Part {self.part_id} batch {batch_num} failed: {e}")
                continue
                
        # Combine and save
        embeddings = np.vstack(all_embeddings)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
            
        elapsed = time.time() - start_time
        logger.info(f"✅ Part {self.part_id} complete: {embeddings.shape} in {elapsed/60:.1f}min")
        
        return embeddings_file

class EmbeddingCombiner:
    """Combine embeddings from all parts and build FAISS index"""
    
    def __init__(self, num_parts: int):
        self.num_parts = num_parts
        self.parts_dir = Path("embedding_parts")
        self.output_dir = Path("retrieval_indexes")
        self.output_dir.mkdir(exist_ok=True)
        
    def combine_embeddings(self):
        """Load and combine all embedding parts"""
        logger.info(f"🔗 Combining embeddings from {self.num_parts} parts")
        
        all_embeddings = []
        total_vectors = 0
        
        for i in range(self.num_parts):
            part_file = self.parts_dir / f"embeddings_part_{i:02d}.pkl"
            
            if not part_file.exists():
                logger.error(f"❌ Missing embeddings for part {i}: {part_file}")
                continue
                
            logger.info(f"📖 Loading part {i}")
            with open(part_file, 'rb') as f:
                part_embeddings = pickle.load(f)
                
            all_embeddings.append(part_embeddings)
            total_vectors += len(part_embeddings)
            logger.info(f"  Part {i}: {part_embeddings.shape}")
            
        # Combine all embeddings
        logger.info(f"🔗 Stacking {total_vectors} total vectors")
        combined_embeddings = np.vstack(all_embeddings)
        
        # Save combined embeddings
        combined_file = self.output_dir / "bge_m3_embeddings_combined.pkl"
        with open(combined_file, 'wb') as f:
            pickle.dump(combined_embeddings, f)
            
        logger.info(f"✅ Combined embeddings saved: {combined_embeddings.shape}")
        return combined_embeddings
        
    def build_faiss_index(self, embeddings):
        """Build FAISS index from combined embeddings"""
        faiss_file = self.output_dir / "bge_m3_faiss_combined.index"
        
        logger.info("🏗️  Building FAISS HNSW index")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.ef_construction = 200
        index.hnsw.ef_search = 100
        
        logger.info(f"📈 Adding {len(embeddings)} vectors to index")
        index.add(embeddings.astype('float32'))
        
        faiss.write_index(index, str(faiss_file))
        
        logger.info(f"✅ FAISS index saved: {index.ntotal} vectors")
        return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['split', 'embed', 'combine'], required=True)
    parser.add_argument('--part-id', type=int, help='Part ID for embed mode')
    parser.add_argument('--num-splits', type=int, default=4, help='Number of splits')
    args = parser.parse_args()
    
    if args.mode == 'split':
        splitter = CorpusSplitter(
            "chunked_corpus/contextual_chunks_complete.parquet",
            num_splits=args.num_splits
        )
        split_files = splitter.split_corpus()
        logger.info(f"Ready to embed {len(split_files)} parts in parallel")
        
    elif args.mode == 'embed':
        if args.part_id is None:
            logger.error("--part-id required for embed mode")
            sys.exit(1)
            
        corpus_file = f"corpus_splits/corpus_part_{args.part_id:02d}.parquet"
        embedder = SinglePartEmbedder(args.part_id, corpus_file, batch_size=64)
        embeddings_file = embedder.embed_part()
        logger.info(f"Part {args.part_id} embeddings saved to {embeddings_file}")
        
    elif args.mode == 'combine':
        combiner = EmbeddingCombiner(args.num_splits)
        embeddings = combiner.combine_embeddings()
        index = combiner.build_faiss_index(embeddings)
        logger.info("🎉 Complete hybrid retrieval system ready!")

if __name__ == "__main__":
    main()