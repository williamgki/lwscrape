#!/usr/bin/env python3
"""
Fixed multi-worker crawler with proper SQLite concurrency handling
"""

import multiprocessing as mp
import time
import json
import os
import signal
import sys
import threading
from datetime import datetime
from dataclasses import asdict
from multi_crawler import DomainSpecificExtractor, CrawledContent
from db_utils import SafeCrawlerDB


class FixedWorkerCrawler:
    """Worker with safe database operations"""
    
    def __init__(self, worker_id: int, db_path: str, output_dir: str):
        self.worker_id = worker_id
        self.db = SafeCrawlerDB(db_path)
        self.output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        self.extractor = DomainSpecificExtractor()
        
        # Stats
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'partial': 0,
            'start_time': datetime.now()
        }
    
    def save_content(self, content: CrawledContent):
        """Save content to JSONL file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            json.dump(asdict(content), f, ensure_ascii=False)
            f.write('\n')
    
    def print_progress(self):
        """Print worker progress"""
        elapsed = datetime.now() - self.stats['start_time']
        rate = self.stats['processed'] / elapsed.total_seconds() * 3600 if elapsed.total_seconds() > 0 else 0
        
        print(f"Worker {self.worker_id}: {self.stats['processed']} processed | "
              f"Success: {self.stats['successful']} | "
              f"Failed: {self.stats['failed']} | "
              f"Partial: {self.stats['partial']} | "
              f"Rate: {rate:.1f}/hour")
    
    def crawl_batch(self, assigned_domains: list, stop_event: mp.Event):
        """Main crawling loop for worker"""
        print(f"Worker {self.worker_id} starting with {len(assigned_domains)} domains")
        
        while not stop_event.is_set():
            batch_processed = 0
            
            # Process each assigned domain
            for domain in assigned_domains:
                if stop_event.is_set():
                    break
                
                # Get URLs for this domain (smaller batches for better concurrency)
                urls = self.db.get_next_urls(self.worker_id, domain_filter=domain, limit=3)
                
                if not urls:
                    continue
                
                # Mark URLs as processing (batch operation)
                url_ids = [url[0] for url in urls]
                self.db.mark_processing(url_ids, self.worker_id)
                
                for url_id, url, domain, frequency in urls:
                    if stop_event.is_set():
                        break
                    
                    print(f"Worker {self.worker_id}: {url} (freq: {frequency})")
                    
                    # Get crawl delay
                    delay = self.db.get_crawl_delay(domain)
                    time.sleep(delay)
                    
                    # Extract content
                    result = self.extractor.fetch_and_extract(url)
                    
                    # Update stats
                    self.stats['processed'] += 1
                    if result.status == 'success':
                        self.stats['successful'] += 1
                    elif result.status == 'partial':
                        self.stats['partial'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    # Save results
                    self.save_content(result)
                    self.db.mark_completed(
                        url_id, 
                        result.status, 
                        result.content_length,
                        self.output_file if result.status in ['success', 'partial'] else None,
                        result.error_message
                    )
                    self.db.update_domain_last_crawled(domain)
                    
                    batch_processed += 1
                    
                    # Print progress every 5 URLs
                    if self.stats['processed'] % 5 == 0:
                        self.print_progress()
            
            # If no work done, sleep before retrying
            if batch_processed == 0:
                print(f"Worker {self.worker_id}: No URLs found, sleeping 60s...")
                time.sleep(60)
        
        print(f"Worker {self.worker_id} shutting down gracefully")


class FixedMultiCrawler:
    """Fixed multi-worker crawler manager"""
    
    def __init__(self, db_path: str, output_dir: str = "data/crawled", num_workers: int = 4):
        self.db_path = db_path
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = mp.Event()
        self.db = SafeCrawlerDB(db_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}. Shutting down workers...")
        self.stop_event.set()
    
    def start_worker(self, worker_id: int, assigned_domains: list):
        """Start worker process with assigned domains"""
        def worker_main():
            crawler = FixedWorkerCrawler(worker_id, self.db_path, self.output_dir)
            crawler.crawl_batch(assigned_domains, self.stop_event)
        
        worker_process = mp.Process(target=worker_main)
        worker_process.start()
        return worker_process
    
    def monitor_progress(self):
        """Monitor overall progress"""
        def monitor_loop():
            while not self.stop_event.is_set():
                try:
                    total, pending, processing, success, failed = self.db.get_progress_stats()
                    worker_stats = self.db.get_worker_stats()
                    
                    completed = success + failed
                    progress_pct = (completed / total * 100) if total > 0 else 0
                    
                    print(f"\n=== PROGRESS REPORT ===")
                    print(f"Total: {total:,} | Completed: {completed:,} ({progress_pct:.1f}%) | Pending: {pending:,}")
                    print(f"Success: {success:,} | Failed: {failed:,} | Processing: {processing:,}")
                    
                    for worker_id in range(1, self.num_workers + 1):
                        count = worker_stats.get(worker_id, 0)
                        print(f"Worker {worker_id}: {count:,} URLs")
                    
                    print("=" * 23)
                    
                    # Stop if all done
                    if pending == 0 and processing == 0:
                        print("All URLs processed! Shutting down...")
                        self.stop_event.set()
                        break
                    
                    time.sleep(120)  # Update every 2 minutes
                    
                except Exception as e:
                    print(f"Monitor error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def run(self):
        """Run the multi-worker crawler"""
        print(f"Starting fixed multi-worker crawler with {self.num_workers} workers")
        
        # Get domain assignments
        domain_assignments = self.db.get_domain_assignments(self.num_workers)
        
        print("Worker domain assignments:")
        for worker_id, domains in domain_assignments.items():
            print(f"  Worker {worker_id}: {len(domains)} domains")
        
        # Start monitor
        monitor_thread = self.monitor_progress()
        
        # Start workers
        for worker_id in range(1, self.num_workers + 1):
            assigned_domains = domain_assignments[worker_id]
            worker_process = self.start_worker(worker_id, assigned_domains)
            self.workers.append(worker_process)
        
        print(f"All {len(self.workers)} workers started")
        
        try:
            # Wait for completion or interruption
            while not self.stop_event.is_set():
                # Check if any workers died unexpectedly
                for i, worker in enumerate(self.workers):
                    if not worker.is_alive():
                        print(f"Worker {i+1} died unexpectedly, restarting...")
                        assigned_domains = domain_assignments[i+1]
                        new_worker = self.start_worker(i+1, assigned_domains)
                        self.workers[i] = new_worker
                
                time.sleep(30)
            
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received...")
            self.stop_event.set()
        
        # Shutdown workers
        print("Waiting for workers to finish...")
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                print(f"Force terminating worker {worker.pid}")
                worker.terminate()
        
        print("Multi-worker crawler completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed multi-worker crawler')
    parser.add_argument('--db', default='data/crawl_urls.db', help='Database path')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output-dir', default='data/crawled', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found")
        return
    
    crawler = FixedMultiCrawler(
        db_path=args.db,
        output_dir=args.output_dir,
        num_workers=args.workers
    )
    
    crawler.run()


if __name__ == "__main__":
    main()