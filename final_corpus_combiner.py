#!/usr/bin/env python3
"""
Final Corpus Combiner
Combines all enhanced chunks from all sources into final unified corpus
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_final_corpus():
    """Combine all enhanced corpus sources into final unified corpus"""
    
    output_dir = Path("/home/ubuntu/LW_scrape/final_combined_corpus")
    output_dir.mkdir(exist_ok=True)
    
    # Source corpus files
    corpus_sources = {
        'main_with_survey': '/home/ubuntu/LW_scrape/corpus_with_survey/integrated_corpus_with_survey.parquet',
        'wilson_lin_survey': '/home/ubuntu/LW_scrape/wilson_lin_enhanced/wilson_lin_enhanced_chunks.parquet'
    }
    
    logger.info("🤖 Starting final corpus combination...")
    
    combined_chunks = []
    source_stats = {}
    
    for source_name, file_path in corpus_sources.items():
        if not Path(file_path).exists():
            logger.warning(f"⚠️ Missing source: {file_path}")
            continue
            
        logger.info(f"📖 Loading {source_name}...")
        df = pd.read_parquet(file_path)
        
        # Add source identifier
        df['corpus_source'] = source_name
        
        logger.info(f"✅ {source_name}: {len(df):,} chunks")
        source_stats[source_name] = len(df)
        combined_chunks.append(df)
    
    if not combined_chunks:
        raise ValueError("No corpus files found!")
    
    # Combine all chunks
    logger.info("🔄 Combining all chunks...")
    final_corpus = pd.concat(combined_chunks, ignore_index=True)
    
    # Deduplicate based on content hash (if chunks overlap between sources)
    logger.info("🔍 Deduplicating chunks...")
    initial_count = len(final_corpus)
    final_corpus = final_corpus.drop_duplicates(subset=['chunk_id'], keep='first')
    final_count = len(final_corpus)
    
    dedup_removed = initial_count - final_count
    logger.info(f"🧹 Removed {dedup_removed:,} duplicate chunks")
    
    # Generate final statistics
    stats = {
        'combination_timestamp': datetime.now().isoformat(),
        'total_chunks': final_count,
        'initial_chunks_before_dedup': initial_count,
        'duplicates_removed': dedup_removed,
        'source_contributions': source_stats,
        'final_schema_columns': list(final_corpus.columns),
        'content_types': final_corpus['content_type'].value_counts().to_dict(),
        'domains': final_corpus['domain'].value_counts().head(10).to_dict(),
        'avg_chunk_size': final_corpus['char_count'].mean(),
        'total_content_chars': final_corpus['char_count'].sum()
    }
    
    # Save final corpus
    output_file = output_dir / 'final_unified_corpus.parquet'
    logger.info(f"💾 Saving final corpus: {output_file}")
    final_corpus.to_parquet(output_file, index=False)
    
    # Save statistics
    stats_file = output_dir / 'final_corpus_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info("="*60)
    logger.info("🎯 FINAL CORPUS COMBINATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"📊 Total chunks: {final_count:,}")
    logger.info(f"📚 Content types: {len(final_corpus['content_type'].unique())}")
    logger.info(f"🌐 Domains: {len(final_corpus['domain'].unique())}")
    logger.info(f"📝 Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
    logger.info(f"💾 Output: {output_file}")
    logger.info("="*60)
    
    return output_file, stats

if __name__ == "__main__":
    combine_final_corpus()