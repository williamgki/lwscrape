#!/usr/bin/env python3
"""
Production-ready multi-worker contextual chunking pipeline.
Processes full corpus with API rate limiting, progress tracking, and error handling.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
import time
from pathlib import Path
from dataclasses import asdict
import queue
import signal
import traceback

from enhanced_contextual_chunker import (
    EnhancedContextualChunker, 
    ContextualChunk, 
    load_documents_batch
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionChunkingPipeline:
    """Production pipeline for processing full corpus with multi-worker architecture"""
    
    def __init__(self, 
                 parquet_path: str,
                 output_dir: str = "./chunked_corpus",
                 num_workers: int = 16,
                 batch_size: int = 12,
                 api_rate_limit: float = 0.03,  # 33 calls/sec max
                 checkpoint_interval: int = 100):
        
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.api_rate_limit = api_rate_limit
        self.checkpoint_interval = checkpoint_interval
        
        # Progress tracking
        self.total_documents = 0
        self.processed_documents = 0
        self.total_chunks = 0
        self.failed_documents = []
        self.start_time = None
        
        # Thread safety
        self.progress_lock = threading.Lock()
        self.output_lock = threading.Lock()
        
        # Graceful shutdown
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load document count
        self._load_corpus_info()
        
    def _load_corpus_info(self):
        """Load corpus information"""
        try:
            df = pd.read_parquet(self.parquet_path)
            self.total_documents = len(df)
            logger.info(f"Loaded corpus info: {self.total_documents} documents")
            
            # Check for existing progress
            progress_file = self.output_dir / "progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_documents = progress.get('processed_documents', 0)
                    self.total_chunks = progress.get('total_chunks', 0)
                    logger.info(f"Resuming from: {self.processed_documents} processed docs")
                    
        except Exception as e:
            logger.error(f"Failed to load corpus info: {e}")
            raise
            
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.shutdown_event.set()
        
    def _save_progress(self):
        """Save current progress"""
        progress_data = {
            'processed_documents': self.processed_documents,
            'total_documents': self.total_documents,
            'total_chunks': self.total_chunks,
            'failed_documents': self.failed_documents,
            'completion_percentage': (self.processed_documents / self.total_documents) * 100 if self.total_documents > 0 else 0
        }
        
        progress_file = self.output_dir / "progress.json"
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def _save_chunks_batch(self, chunks: List[ContextualChunk], batch_id: int):
        """Save a batch of chunks to parquet"""
        if not chunks:
            return
            
        # Convert chunks to records
        records = [chunk.to_dict() for chunk in chunks]
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save to parquet
        with self.output_lock:
            output_file = self.output_dir / f"chunks_batch_{batch_id:04d}.parquet"
            df.to_parquet(output_file, compression='snappy')
            logger.info(f"Saved {len(chunks)} chunks to {output_file}")
            
    def _process_document_batch(self, batch_docs: List[Dict], 
                               chunker: EnhancedContextualChunker, 
                               batch_id: int) -> Tuple[List[ContextualChunk], int, List[str]]:
        """Process a batch of documents"""
        
        batch_chunks = []
        processed_count = 0
        batch_failures = []
        
        for doc in batch_docs:
            if self.shutdown_event.is_set():
                break
                
            try:
                doc_chunks = chunker.process_document(doc)
                batch_chunks.extend(doc_chunks)
                processed_count += 1
                
                logger.debug(f"Processed {doc['doc_id'][:12]}... → {len(doc_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process document {doc['doc_id']}: {e}")
                batch_failures.append(doc['doc_id'])
                traceback.print_exc()
                continue
                
        return batch_chunks, processed_count, batch_failures
        
    def _worker_process_batch(self, batch_start: int, batch_id: int) -> Dict[str, Any]:
        """Worker function to process a document batch"""
        
        try:
            # Load batch
            batch_docs = load_documents_batch(
                self.parquet_path, 
                batch_size=self.batch_size,
                start_idx=batch_start
            )
            
            if not batch_docs:
                return {'batch_id': batch_id, 'chunks': 0, 'processed': 0, 'failures': []}
            
            # Initialize chunker for this worker
            chunker = EnhancedContextualChunker(
                min_tokens=700,
                max_tokens=1200,
                api_rate_limit=self.api_rate_limit
            )
            
            # Process batch
            batch_chunks, processed_count, batch_failures = self._process_document_batch(
                batch_docs, chunker, batch_id
            )
            
            # Save chunks
            if batch_chunks:
                self._save_chunks_batch(batch_chunks, batch_id)
                
            return {
                'batch_id': batch_id,
                'chunks': len(batch_chunks),
                'processed': processed_count,
                'failures': batch_failures
            }
            
        except Exception as e:
            logger.error(f"Worker batch {batch_id} failed: {e}")
            traceback.print_exc()
            return {
                'batch_id': batch_id,
                'chunks': 0,
                'processed': 0,
                'failures': [f"batch_{batch_id}_total_failure"],
                'error': str(e)
            }
            
    def _update_progress(self, processed: int, chunks: int, failures: List[str]):
        """Update progress tracking"""
        with self.progress_lock:
            self.processed_documents += processed
            self.total_chunks += chunks
            self.failed_documents.extend(failures)
            
    def _log_progress(self):
        """Log current progress"""
        if self.total_documents == 0:
            return
            
        completion_pct = (self.processed_documents / self.total_documents) * 100
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        rate = self.processed_documents / elapsed_time if elapsed_time > 0 else 0
        
        remaining_docs = self.total_documents - self.processed_documents
        eta_seconds = remaining_docs / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60
        
        logger.info(f"Progress: {self.processed_documents}/{self.total_documents} docs "
                   f"({completion_pct:.1f}%) | {self.total_chunks} chunks | "
                   f"Rate: {rate:.2f} docs/sec | ETA: {eta_minutes:.1f}min | "
                   f"Failures: {len(self.failed_documents)}")
                   
    def process_corpus(self, resume: bool = True) -> Dict[str, Any]:
        """Process the complete corpus with multi-worker pipeline"""
        
        logger.info("🚀 Starting production contextual chunking pipeline")
        logger.info(f"Configuration: {self.num_workers} workers, {self.batch_size} batch size")
        
        self.start_time = time.time()
        
        # Determine starting point
        start_doc = self.processed_documents if resume else 0
        
        # Create batch tasks
        batch_tasks = []
        batch_id = 0
        
        for batch_start in range(start_doc, self.total_documents, self.batch_size):
            if self.shutdown_event.is_set():
                break
                
            batch_tasks.append((batch_start, batch_id))
            batch_id += 1
            
        logger.info(f"Created {len(batch_tasks)} batch tasks starting from doc {start_doc}")
        
        # Process batches with thread pool
        completed_batches = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(self._worker_process_batch, batch_start, bid): bid
                for batch_start, bid in batch_tasks
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                if self.shutdown_event.is_set():
                    break
                    
                batch_id = future_to_batch[future]
                
                try:
                    result = future.result()
                    
                    # Update progress
                    self._update_progress(
                        result['processed'],
                        result['chunks'],
                        result['failures']
                    )
                    
                    completed_batches += 1
                    
                    # Log progress periodically
                    if completed_batches % 5 == 0:  # Every 5 batches
                        self._log_progress()
                        
                    # Save progress checkpoint
                    if completed_batches % self.checkpoint_interval == 0:
                        self._save_progress()
                        logger.info("📊 Checkpoint saved")
                        
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")
                    
        # Final progress save
        self._save_progress()
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        avg_rate = self.processed_documents / total_time if total_time > 0 else 0
        
        results = {
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'total_chunks': self.total_chunks,
            'failed_documents': len(self.failed_documents),
            'processing_time_seconds': total_time,
            'average_rate_docs_per_sec': avg_rate,
            'completion_percentage': (self.processed_documents / self.total_documents) * 100,
            'chunks_per_document': self.total_chunks / self.processed_documents if self.processed_documents > 0 else 0
        }
        
        logger.info("🎉 Production chunking pipeline completed!")
        logger.info(f"📊 Final stats: {self.processed_documents} docs → {self.total_chunks} chunks")
        logger.info(f"⚡ Rate: {avg_rate:.2f} docs/sec | ❌ Failures: {len(self.failed_documents)}")
        
        return results
        
    def consolidate_chunks(self) -> str:
        """Consolidate all chunk batches into single parquet file"""
        
        logger.info("🔄 Consolidating chunk batches...")
        
        chunk_files = list(self.output_dir.glob("chunks_batch_*.parquet"))
        
        if not chunk_files:
            logger.error("No chunk files found to consolidate")
            return ""
            
        logger.info(f"Found {len(chunk_files)} chunk batch files")
        
        # Load all chunk DataFrames
        dfs = []
        total_chunks = 0
        
        for chunk_file in sorted(chunk_files):
            try:
                df = pd.read_parquet(chunk_file)
                dfs.append(df)
                total_chunks += len(df)
                logger.debug(f"Loaded {len(df)} chunks from {chunk_file}")
            except Exception as e:
                logger.error(f"Failed to load {chunk_file}: {e}")
                continue
                
        if not dfs:
            logger.error("No chunk dataframes loaded")
            return ""
            
        # Concatenate all chunks
        logger.info(f"Consolidating {total_chunks} total chunks...")
        consolidated_df = pd.concat(dfs, ignore_index=True)
        
        # Save consolidated file
        output_file = self.output_dir / "contextual_chunks_complete.parquet"
        consolidated_df.to_parquet(output_file, compression='snappy')
        
        logger.info(f"✅ Consolidated chunks saved: {output_file}")
        logger.info(f"📦 Final chunk store: {total_chunks} chunks ({output_file.stat().st_size / 1024 / 1024:.1f}MB)")
        
        return str(output_file)


def main():
    """Run production chunking pipeline"""
    
    # Configuration
    parquet_path = "/home/ubuntu/LW_scrape/normalized_corpus/document_store.parquet"
    output_dir = "/home/ubuntu/LW_scrape/chunked_corpus"
    
    # Initialize pipeline
    pipeline = ProductionChunkingPipeline(
        parquet_path=parquet_path,
        output_dir=output_dir,
        num_workers=4,
        batch_size=25,        # Smaller batches for API rate limiting
        api_rate_limit=0.15,  # ~6-7 calls/sec to be conservative
        checkpoint_interval=20  # Save progress every 20 batches
    )
    
    try:
        # Process corpus
        results = pipeline.process_corpus(resume=True)
        
        # Consolidate results
        consolidated_file = pipeline.consolidate_chunks()
        
        # Print final summary
        print("\n" + "="*60)
        print("🎉 PRODUCTION CONTEXTUAL CHUNKING COMPLETE")
        print("="*60)
        print(f"Documents processed: {results['processed_documents']:,}")
        print(f"Total chunks created: {results['total_chunks']:,}")
        print(f"Processing time: {results['processing_time_seconds']/3600:.2f} hours")
        print(f"Average rate: {results['average_rate_docs_per_sec']:.2f} docs/sec")
        print(f"Chunks per document: {results['chunks_per_document']:.1f}")
        print(f"Success rate: {100 - (results['failed_documents']/results['total_documents']*100):.1f}%")
        print(f"Final chunk store: {consolidated_file}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"💥 Pipeline failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()