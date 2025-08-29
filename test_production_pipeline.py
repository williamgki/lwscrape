#!/usr/bin/env python3
"""
Test production chunking pipeline on small subset
"""

import sys
from production_chunking_pipeline import ProductionChunkingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_small_pipeline():
    """Test pipeline on small subset"""
    
    logger.info("🧪 Testing production pipeline on small subset")
    
    # Configuration for testing
    parquet_path = "/home/ubuntu/LW_scrape/normalized_corpus/document_store.parquet"
    output_dir = "/home/ubuntu/LW_scrape/test_chunked_corpus"
    
    # Create test pipeline - smaller scale
    pipeline = ProductionChunkingPipeline(
        parquet_path=parquet_path,
        output_dir=output_dir,
        num_workers=2,           # Fewer workers for testing
        batch_size=10,           # Small batches
        api_rate_limit=0.2,      # Conservative rate limit
        checkpoint_interval=5    # Frequent checkpoints
    )
    
    # Mock total documents to small number for testing
    original_total = pipeline.total_documents
    pipeline.total_documents = min(50, original_total)  # Test with 50 docs max
    
    logger.info(f"Testing with {pipeline.total_documents} documents")
    
    try:
        # Run test
        results = pipeline.process_corpus(resume=False)
        
        # Consolidate
        consolidated_file = pipeline.consolidate_chunks()
        
        # Print results
        print(f"\n✅ TEST COMPLETE:")
        print(f"   Processed: {results['processed_documents']} docs")
        print(f"   Chunks: {results['total_chunks']}")
        print(f"   Rate: {results['average_rate_docs_per_sec']:.2f} docs/sec")
        print(f"   Output: {consolidated_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_pipeline()
    sys.exit(0 if success else 1)