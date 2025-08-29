#!/usr/bin/env python3
"""
Corpus Merger - Combine LW and GChildren Corpora
Merges two compatible contextual chunk corpora with deduplication
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import hashlib
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusMerger:
    """Merge main LW corpus with gchildren corpus"""
    
    def __init__(self, 
                 main_corpus_dir: str,
                 gchildren_corpus_dir: str,
                 output_dir: str):
        
        self.main_corpus_dir = Path(main_corpus_dir)
        self.gchildren_corpus_dir = Path(gchildren_corpus_dir) 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / 'corpus_merger.log'
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
    def validate_schema_compatibility(self) -> bool:
        """Validate that both corpora have compatible schemas"""
        logger.info("Validating schema compatibility...")
        
        # Load sample from main corpus
        main_files = list(self.main_corpus_dir.glob("chunks_batch_*.parquet"))
        if not main_files:
            logger.error("No main corpus files found")
            return False
            
        main_sample = pd.read_parquet(main_files[0]).head(10)
        
        # Load sample from gchildren corpus
        gchildren_files = list(self.gchildren_corpus_dir.glob("gchildren_chunks_batch_*.parquet"))
        if not gchildren_files:
            logger.error("No gchildren corpus files found") 
            return False
            
        gchildren_sample = pd.read_parquet(gchildren_files[0]).head(10)
        
        # Compare schemas
        main_cols = set(main_sample.columns)
        gchildren_cols = set(gchildren_sample.columns)
        
        if main_cols != gchildren_cols:
            logger.error(f"Schema mismatch! Main: {main_cols}, GChildren: {gchildren_cols}")
            missing_in_main = gchildren_cols - main_cols
            missing_in_gchildren = main_cols - gchildren_cols
            if missing_in_main:
                logger.error(f"Missing in main: {missing_in_main}")
            if missing_in_gchildren:
                logger.error(f"Missing in gchildren: {missing_in_gchildren}")
            return False
            
        # Compare data types
        for col in main_cols:
            if main_sample[col].dtype != gchildren_sample[col].dtype:
                logger.warning(f"Data type mismatch for {col}: main={main_sample[col].dtype}, gchildren={gchildren_sample[col].dtype}")
        
        logger.info("✅ Schema compatibility validated")
        return True
    
    def load_corpus_chunks(self, corpus_dir: Path, corpus_name: str) -> pd.DataFrame:
        """Load all chunks from a corpus directory"""
        logger.info(f"Loading {corpus_name} corpus from {corpus_dir}")
        
        if corpus_name == "main":
            pattern = "chunks_batch_*.parquet"
        else:
            pattern = "gchildren_chunks_batch_*.parquet"
            
        chunk_files = list(corpus_dir.glob(pattern))
        logger.info(f"Found {len(chunk_files)} files for {corpus_name} corpus")
        
        if not chunk_files:
            logger.error(f"No chunk files found for {corpus_name} corpus")
            return pd.DataFrame()
        
        # Load all chunks
        chunks_list = []
        for file_path in chunk_files:
            try:
                batch_df = pd.read_parquet(file_path)
                chunks_list.append(batch_df)
                if len(chunks_list) % 50 == 0:
                    logger.info(f"Loaded {len(chunks_list)} files for {corpus_name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Combine all chunks
        if chunks_list:
            combined_df = pd.concat(chunks_list, ignore_index=True)
            logger.info(f"✅ Loaded {len(combined_df):,} chunks from {corpus_name} corpus")
            return combined_df
        else:
            logger.error(f"No valid chunks loaded from {corpus_name} corpus")
            return pd.DataFrame()
    
    def detect_duplicates(self, main_df: pd.DataFrame, gchildren_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect and remove duplicates between corpora"""
        logger.info("Detecting cross-corpus duplicates...")
        
        # Create content hashes for deduplication
        main_df['content_hash'] = main_df['content'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )
        gchildren_df['content_hash'] = gchildren_df['content'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )
        
        # Find duplicates
        main_hashes = set(main_df['content_hash'])
        duplicate_mask = gchildren_df['content_hash'].isin(main_hashes)
        
        duplicates_found = duplicate_mask.sum()
        logger.info(f"Found {duplicates_found:,} duplicate chunks in gchildren corpus")
        
        # Remove duplicates from gchildren
        gchildren_clean = gchildren_df[~duplicate_mask].copy()
        
        # Clean up hash columns
        main_df = main_df.drop('content_hash', axis=1)
        gchildren_clean = gchildren_clean.drop('content_hash', axis=1)
        
        dedup_stats = {
            'duplicates_found': int(duplicates_found),
            'gchildren_before': len(gchildren_df),
            'gchildren_after': len(gchildren_clean),
            'dedup_rate': float(duplicates_found / len(gchildren_df) * 100)
        }
        
        logger.info(f"✅ Deduplication complete: {duplicates_found:,} duplicates removed ({dedup_stats['dedup_rate']:.1f}%)")
        return gchildren_clean, dedup_stats
    
    def merge_corpora(self) -> Tuple[pd.DataFrame, Dict]:
        """Main merger function"""
        logger.info("Starting corpus merger...")
        
        # Step 1: Validate compatibility
        if not self.validate_schema_compatibility():
            raise ValueError("Schema compatibility check failed")
        
        # Step 2: Load both corpora
        main_df = self.load_corpus_chunks(self.main_corpus_dir, "main")
        gchildren_df = self.load_corpus_chunks(self.gchildren_corpus_dir, "gchildren")
        
        if main_df.empty or gchildren_df.empty:
            raise ValueError("Failed to load one or both corpora")
        
        # Step 3: Deduplication
        gchildren_clean, dedup_stats = self.detect_duplicates(main_df, gchildren_df)
        
        # Step 4: Add corpus source tags
        main_df['corpus_source'] = 'main'
        gchildren_clean['corpus_source'] = 'gchildren'
        
        # Step 5: Merge corpora
        logger.info("Merging corpora...")
        combined_df = pd.concat([main_df, gchildren_clean], ignore_index=True)
        
        # Step 6: Generate merger statistics
        merger_stats = {
            'main_corpus_chunks': len(main_df),
            'gchildren_original_chunks': len(gchildren_df),
            'gchildren_after_dedup': len(gchildren_clean),
            'combined_chunks': len(combined_df),
            'deduplication': dedup_stats,
            'expansion_rate': float(len(gchildren_clean) / len(main_df) * 100),
            'merge_timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ Corpus merger completed successfully")
        logger.info(f"Final corpus: {len(combined_df):,} chunks")
        logger.info(f"Expansion rate: {merger_stats['expansion_rate']:.1f}%")
        
        return combined_df, merger_stats
    
    def save_merged_corpus(self, combined_df: pd.DataFrame, stats: Dict):
        """Save merged corpus to parquet files"""
        logger.info("Saving merged corpus...")
        
        # Save in batches like original corpus
        batch_size = 1000
        num_batches = (len(combined_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(combined_df))
            batch_df = combined_df.iloc[start_idx:end_idx]
            
            batch_file = self.output_dir / f"unified_chunks_batch_{batch_idx:04d}.parquet"
            batch_df.to_parquet(batch_file, index=False)
            
            if batch_idx % 10 == 0:
                logger.info(f"Saved batch {batch_idx:04d}/{num_batches-1:04d}")
        
        # Save complete corpus in single file
        complete_file = self.output_dir / "unified_contextual_chunks_complete.parquet"
        combined_df.to_parquet(complete_file, index=False)
        
        # Save merger statistics
        stats_file = self.output_dir / "merger_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save corpus summary
        summary = {
            'total_chunks': len(combined_df),
            'total_batches': num_batches,
            'content_types': combined_df['content_type'].value_counts().to_dict(),
            'corpus_sources': combined_df['corpus_source'].value_counts().to_dict(),
            'domains': combined_df['domain'].value_counts().head(20).to_dict(),
            'avg_tokens': combined_df['token_count'].mean(),
            'total_tokens': combined_df['token_count'].sum(),
        }
        
        summary_file = self.output_dir / "unified_corpus_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Merged corpus saved to {self.output_dir}")
        logger.info(f"Total files: {num_batches} batches + 1 complete file")
        return summary

def main():
    """Run corpus merger"""
    merger = CorpusMerger(
        main_corpus_dir="/home/ubuntu/LW_scrape/chunked_corpus",
        gchildren_corpus_dir="/home/ubuntu/LW_scrape/gchildren/chunked_corpus", 
        output_dir="/home/ubuntu/LW_scrape/unified_corpus"
    )
    
    try:
        # Merge corpora
        combined_df, stats = merger.merge_corpora()
        
        # Save merged corpus
        summary = merger.save_merged_corpus(combined_df, stats)
        
        # Final report
        print("\n" + "="*60)
        print("🚀 CORPUS MERGER COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Main corpus: {stats['main_corpus_chunks']:,} chunks")
        print(f"GChildren corpus: {stats['gchildren_original_chunks']:,} chunks")
        print(f"After deduplication: {stats['gchildren_after_dedup']:,} chunks")
        print(f"Combined corpus: {stats['combined_chunks']:,} chunks")
        print(f"Expansion rate: {stats['expansion_rate']:.1f}%")
        print(f"Deduplication rate: {stats['deduplication']['dedup_rate']:.1f}%")
        print(f"Saved to: /home/ubuntu/LW_scrape/unified_corpus")
        print("✅ Ready for retrieval system integration!")
        
        return combined_df, stats
        
    except Exception as e:
        logger.error(f"Corpus merger failed: {e}")
        raise

if __name__ == '__main__':
    main()