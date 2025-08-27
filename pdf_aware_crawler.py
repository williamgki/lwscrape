#!/usr/bin/env python3
"""
PDF-Aware Multi-Worker Crawler
Automatically detects and queues PDFs for later processing
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
from pdf_queue_utils import PDFQueueManager
from urllib.parse import urlparse

class PDFAwareExtractor(DomainSpecificExtractor):
    """Extended extractor that automatically queues PDFs"""
    
    def __init__(self, db_path: str = "data/crawl_urls.db"):
        super().__init__()
        self.pdf_queue = PDFQueueManager(db_path)
        self.pdfs_detected = 0
    
    def extract_content(self, url: str) -> CrawledContent:
        """Extract content with PDF detection"""
        try:
            # Get response first
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if this is a PDF
            if self.pdf_queue.is_pdf_content_type(content_type):
                # Add to PDF queue for later processing
                if self.pdf_queue.add_pdf_to_queue(url, content_type):
                    self.pdfs_detected += 1
                
                return CrawledContent(
                    url=url,
                    title="PDF queued for processing",
                    content="",
                    domain=urlparse(url).netloc,
                    content_type=content_type,
                    status='pdf_queued',
                    scraped_at=datetime.now().isoformat(),
                    content_length=len(response.content),
                    error_message=f"PDF queued: {content_type}"
                )
            
            # For non-PDF, non-HTML content, return failed status
            if 'html' not in content_type:
                return CrawledContent(
                    url=url,
                    title="Non-HTML content",
                    content="",
                    domain=urlparse(url).netloc,
                    content_type=content_type,
                    status='failed',
                    scraped_at=datetime.now().isoformat(),
                    content_length=0,
                    error_message=f"Non-HTML content type: {content_type}"
                )
            
            # Process HTML content normally
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            domain = urlparse(url).netloc.lower()
            
            # Use domain-specific extraction
            if 'wikipedia.org' in domain:
                return self.extract_wikipedia(soup, url)
            elif 'arxiv.org' in domain:
                return self.extract_arxiv(soup, url)
            elif 'youtube.com' in domain or 'youtu.be' in domain:
                return self.extract_youtube(soup, url)
            elif 'github.com' in domain:
                return self.extract_github(soup, url)
            else:
                return self.extract_generic(soup, url)
                
        except Exception as e:
            return CrawledContent(
                url=url,
                title="Error",
                content="",
                domain=urlparse(url).netloc,
                content_type="",
                status='failed',
                scraped_at=datetime.now().isoformat(),
                content_length=0,
                error_message=str(e)
            )

class PDFAwareWorker:
    """Worker with PDF detection capabilities"""
    
    def __init__(self, worker_id: int, db_path: str, output_dir: str):
        self.worker_id = worker_id
        self.db = SafeCrawlerDB(db_path)
        self.output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        self.extractor = PDFAwareExtractor(db_path)
        
        # Stats
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'partial': 0,
            'pdfs_queued': 0,
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
              f"PDFs queued: {self.stats['pdfs_queued']} | "
              f"Rate: {rate:.1f}/hour")
    
    def crawl_batch(self, assigned_domains: list, stop_event: mp.Event):
        """Main crawling loop with PDF detection"""
        print(f"Worker {self.worker_id} starting with {len(assigned_domains)} domains (PDF-aware)")
        
        while not stop_event.is_set():
            batch_processed = 0
            
            for domain in assigned_domains:
                if stop_event.is_set():
                    break
                
                urls = self.db.get_next_urls(self.worker_id, domain_filter=domain, limit=3)
                
                if not urls:
                    continue
                
                for url_id, url, domain, frequency in urls:
                    if stop_event.is_set():
                        break
                    
                    try:
                        # Mark as processing
                        self.db.update_url_status(url_id, 'processing', self.worker_id)
                        
                        print(f"Worker {self.worker_id}: {url} (freq: {frequency})")
                        
                        # Extract content (with PDF detection)
                        content = self.extractor.extract_content(url)
                        
                        # Update stats based on status
                        if content.status == 'success':
                            self.stats['successful'] += 1
                            status = 'success'
                        elif content.status == 'pdf_queued':
                            self.stats['pdfs_queued'] += 1
                            self.extractor.pdfs_detected += 1
                            status = 'pdf_queued'  # Special status for queued PDFs
                        elif content.status == 'partial':
                            self.stats['partial'] += 1
                            status = 'partial'
                        else:
                            self.stats['failed'] += 1
                            status = 'failed'
                        
                        self.stats['processed'] += 1
                        batch_processed += 1
                        
                        # Save content
                        self.save_content(content)
                        
                        # Update database
                        file_path = f"worker_{self.worker_id}.jsonl"
                        self.db.update_url_completion(
                            url_id, status, file_path, 
                            len(content.content), content.error_message
                        )
                        
                        # Progress reporting
                        if self.stats['processed'] % 5 == 0:
                            self.print_progress()
                        
                        # Rate limiting
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Worker {self.worker_id} error processing {url}: {e}")
                        self.stats['failed'] += 1
                        self.db.update_url_completion(url_id, 'failed', None, 0, str(e))
            
            if batch_processed == 0:
                time.sleep(5)  # No work found, wait before retrying

def run_pdf_aware_crawler(args):
    """Run the PDF-aware crawler"""
    from fixed_multi_crawler import FixedMultiCrawler
    
    class PDFAwareMultiCrawler(FixedMultiCrawler):
        """Multi-crawler with PDF detection"""
        
        def start_worker(self, worker_id: int, assigned_domains: list):
            """Start a PDF-aware worker process"""
            def worker_process():
                try:
                    worker = PDFAwareWorker(worker_id, self.db_path, self.output_dir)
                    worker.crawl_batch(assigned_domains, self.stop_event)
                except Exception as e:
                    print(f"Worker {worker_id} died with error: {e}")
                    # Auto-restart logic would go here
            
            process = mp.Process(target=worker_process)
            process.start()
            return process
        
        def run(self):
            """Run with periodic PDF queue reporting"""
            print("Starting PDF-aware multi-worker crawler")
            
            # Get domain assignments
            assignments = self.db.get_domain_assignments(self.num_workers)
            print("Worker domain assignments:")
            for worker_id, domains in assignments.items():
                print(f"  Worker {worker_id}: {len(domains)} domains")
            
            # Start workers
            for worker_id in range(1, self.num_workers + 1):
                assigned_domains = assignments[worker_id]
                worker_process = self.start_worker(worker_id, assigned_domains)
                self.workers.append(worker_process)
            
            # Monitor progress with PDF queue stats
            pdf_queue = PDFQueueManager(self.db_path)
            
            try:
                while True:
                    time.sleep(300)  # Report every 5 minutes
                    
                    # Check overall progress
                    total, pending, processing, success, failed = self.db.get_progress_stats()
                    print(f"\n=== CRAWLER STATUS ===")
                    print(f"URLs: {total:,} total | {pending:,} pending | {success:,} success | {failed:,} failed")
                    
                    # Check PDF queue
                    pdf_stats = pdf_queue.get_queue_stats()
                    print(f"PDFs: {pdf_stats.get('total', 0)} total | "
                          f"{pdf_stats.get('pending', 0)} pending | "
                          f"{pdf_stats.get('success', 0)} processed")
                    
                    # Check if workers are still alive
                    alive_workers = [p for p in self.workers if p.is_alive()]
                    if len(alive_workers) < self.num_workers:
                        print(f"Warning: {self.num_workers - len(alive_workers)} workers died")
                    
            except KeyboardInterrupt:
                print("\nShutting down crawler...")
                self.stop_event.set()
                
                for worker in self.workers:
                    if worker.is_alive():
                        worker.join(timeout=10)
                        if worker.is_alive():
                            worker.terminate()
    
    # Run the PDF-aware crawler
    crawler = PDFAwareMultiCrawler(
        db_path=args.db,
        output_dir=args.output_dir,
        num_workers=args.workers
    )
    
    crawler.run()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF-Aware Multi-Worker Crawler')
    parser.add_argument('--db', default='data/crawl_urls.db', help='Database path')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output-dir', default='data/crawled', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found")
        sys.exit(1)
    
    run_pdf_aware_crawler(args)