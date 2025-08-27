#!/usr/bin/env python3
"""
Conservative full corpus chunking with smaller resource usage
"""

from production_chunking_pipeline import ProductionChunkingPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run conservative full corpus chunking"""
    
    logger.info("🐢 Starting CONSERVATIVE full corpus chunking")
    
    # Conservative configuration
    parquet_path = "/home/ubuntu/LW_scrape/normalized_corpus/document_store.parquet"
    output_dir = "/home/ubuntu/LW_scrape/chunked_corpus"
    
    # More conservative settings
    pipeline = ProductionChunkingPipeline(
        parquet_path=parquet_path,
        output_dir=output_dir,
        num_workers=2,           # Fewer workers
        batch_size=15,           # Smaller batches  
        api_rate_limit=0.3,      # More conservative API rate
        checkpoint_interval=10   # Frequent checkpoints
    )
    
    try:
        # Process corpus
        logger.info(f"Processing {pipeline.total_documents} documents")
        results = pipeline.process_corpus(resume=True)
        
        # Consolidate results
        consolidated_file = pipeline.consolidate_chunks()
        
        # Print final summary
        print(f"\n🎉 CONSERVATIVE CHUNKING COMPLETE")
        print(f"Documents processed: {results['processed_documents']:,}")
        print(f"Total chunks: {results['total_chunks']:,}")
        print(f"Time: {results['processing_time_seconds']/3600:.2f} hours") 
        print(f"Rate: {results['average_rate_docs_per_sec']:.2f} docs/sec")
        print(f"Output: {consolidated_file}")
        
    except Exception as e:
        logger.error(f"Conservative chunking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()