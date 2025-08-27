#!/usr/bin/env python3
"""
Multi-worker crawler manager for scaling up link crawling across all domains
"""

import requests
import sqlite3
import json
import time
import subprocess
import multiprocessing as mp
from datetime import datetime
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import os
import argparse
from bs4 import BeautifulSoup
import re
import threading
import queue
import signal
import sys


@dataclass
class CrawledContent:
    """Structure for crawled content"""
    url: str
    title: str
    content: str
    domain: str
    content_type: str
    status: str  # 'success', 'partial', 'failed'
    scraped_at: str
    content_length: int
    error_message: Optional[str] = None
    extract_method: Optional[str] = None
    metadata: Optional[Dict] = None


class DomainSpecificExtractor:
    """Enhanced content extraction with domain-specific strategies"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LW-Research-Crawler/1.0; Educational Use)'
        })
    
    def extract_wikipedia(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """Wikipedia-specific extraction (proven working)"""
        try:
            title_elem = soup.find('h1', {'class': 'firstHeading'}) or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            for elem in content_div.find_all(['script', 'style', 'sup', '.navbox', '.infobox']):
                elem.decompose()
            
            paragraphs = content_div.find_all('p')
            content_parts = []
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:
                    content_parts.append(text)
            
            content = '\n\n'.join(content_parts)
            
            metadata = {
                'categories': [cat.get_text(strip=True) for cat in soup.find_all('a', href=re.compile(r'/wiki/Category:'))[:10]],
                'sections': [h.get_text(strip=True) for h in soup.find_all(['h2', 'h3'])[:10]]
            }
            
            return CrawledContent(
                url=url,
                title=title,
                content=content,
                domain='en.wikipedia.org',
                content_type='wikipedia_article',
                status='success',
                scraped_at=datetime.now().isoformat(),
                content_length=len(content),
                extract_method='wikipedia_specific',
                metadata=metadata
            )
        except Exception as e:
            return self._create_error_result(url, e, 'wikipedia_specific')
    
    def extract_arxiv(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """ArXiv paper extraction"""
        try:
            # Extract title
            title_elem = (soup.find('h1', class_='title') or 
                         soup.find('title'))
            title = title_elem.get_text(strip=True).replace('Title:', '').strip() if title_elem else "Unknown Paper"
            
            # Extract abstract
            abstract_elem = soup.find('blockquote', class_='abstract')
            abstract = ""
            if abstract_elem:
                abstract = abstract_elem.get_text(strip=True).replace('Abstract:', '').strip()
            
            # Extract authors
            authors_elem = soup.find('div', class_='authors')
            authors = []
            if authors_elem:
                for author in authors_elem.find_all('a'):
                    authors.append(author.get_text(strip=True))
            
            # Extract submission info
            submission_elem = soup.find('div', class_='submission-history')
            submission_info = submission_elem.get_text(strip=True) if submission_elem else ""
            
            content = f"Title: {title}\n\nAbstract:\n{abstract}\n\nSubmission Info:\n{submission_info}"
            
            metadata = {
                'authors': authors,
                'paper_id': re.search(r'/abs/(\d+\.\d+)', url).group(1) if re.search(r'/abs/(\d+\.\d+)', url) else None,
                'abstract_length': len(abstract)
            }
            
            return CrawledContent(
                url=url,
                title=title,
                content=content,
                domain='arxiv.org',
                content_type='academic_paper',
                status='success',
                scraped_at=datetime.now().isoformat(),
                content_length=len(content),
                extract_method='arxiv_specific',
                metadata=metadata
            )
        except Exception as e:
            return self._create_error_result(url, e, 'arxiv_specific')
    
    def extract_youtube(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """YouTube video extraction (basic metadata only - transcripts would need API)"""
        try:
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Video"
            
            # Extract description from meta tags
            description_elem = soup.find('meta', {'name': 'description'})
            description = description_elem.get('content', '') if description_elem else ""
            
            # Extract video ID
            video_id = re.search(r'[?&]v=([^&]+)', url)
            video_id = video_id.group(1) if video_id else None
            
            content = f"Title: {title}\n\nDescription:\n{description}"
            
            metadata = {
                'video_id': video_id,
                'platform': 'youtube',
                'description_length': len(description)
            }
            
            return CrawledContent(
                url=url,
                title=title,
                content=content,
                domain='youtube.com',
                content_type='video_metadata',
                status='partial',  # Only metadata, no transcript
                scraped_at=datetime.now().isoformat(),
                content_length=len(content),
                extract_method='youtube_basic',
                metadata=metadata
            )
        except Exception as e:
            return self._create_error_result(url, e, 'youtube_basic')
    
    def extract_github(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """GitHub repository/file extraction"""
        try:
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Repository"
            
            # Extract README content if present
            readme_elem = soup.find('div', {'id': 'readme'})
            readme_content = ""
            if readme_elem:
                readme_content = readme_elem.get_text(strip=True)
            
            # Extract description
            desc_elem = soup.find('p', class_='f4') or soup.find('span', {'itemprop': 'about'})
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            content = f"Title: {title}\n\nDescription: {description}\n\nREADME:\n{readme_content}"
            
            metadata = {
                'repository_type': 'github',
                'has_readme': bool(readme_content),
                'readme_length': len(readme_content)
            }
            
            return CrawledContent(
                url=url,
                title=title,
                content=content,
                domain='github.com',
                content_type='repository',
                status='success',
                scraped_at=datetime.now().isoformat(),
                content_length=len(content),
                extract_method='github_specific',
                metadata=metadata
            )
        except Exception as e:
            return self._create_error_result(url, e, 'github_specific')
    
    def extract_generic(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """Generic content extraction fallback"""
        try:
            domain = urlparse(url).netloc
            
            title_elem = (soup.find('title') or 
                         soup.find('h1') or 
                         soup.find('h2'))
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            # Remove unwanted elements
            for elem in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                elem.decompose()
            
            # Find main content
            content_candidates = [
                soup.find('main'),
                soup.find('article'),
                soup.find('div', {'class': re.compile(r'content|main|article|post')}),
                soup.find('body')
            ]
            
            content_elem = None
            for candidate in content_candidates:
                if candidate:
                    content_elem = candidate
                    break
            
            if not content_elem:
                return None
            
            text = content_elem.get_text(separator='\n', strip=True)
            
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.startswith(('Copyright', '©', 'Privacy', 'Terms')):
                    cleaned_lines.append(line)
            
            content = '\n\n'.join(cleaned_lines)
            
            return CrawledContent(
                url=url,
                title=title,
                content=content,
                domain=domain,
                content_type='generic_html',
                status='success' if len(content) > 100 else 'partial',
                scraped_at=datetime.now().isoformat(),
                content_length=len(content),
                extract_method='generic',
                metadata={'original_length': len(text)}
            )
        except Exception as e:
            return self._create_error_result(url, e, 'generic')
    
    def _create_error_result(self, url: str, error: Exception, method: str) -> CrawledContent:
        """Create standardized error result"""
        return CrawledContent(
            url=url,
            title="",
            content="",
            domain=urlparse(url).netloc,
            content_type='unknown',
            status='failed',
            scraped_at=datetime.now().isoformat(),
            content_length=0,
            error_message=str(error),
            extract_method=method
        )
    
    def fetch_and_extract(self, url: str, timeout: int = 30) -> Optional[CrawledContent]:
        """Fetch URL and extract content using appropriate domain-specific method"""
        try:
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
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
            
            soup = BeautifulSoup(response.content, 'html.parser')
            domain = urlparse(url).netloc.lower()
            
            # Choose extraction method based on domain
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
                
        except requests.exceptions.RequestException as e:
            return self._create_error_result(url, e, 'request_failed')
        except Exception as e:
            return self._create_error_result(url, e, 'unexpected_error')


class MultiWorkerCrawler:
    """Multi-worker crawler manager"""
    
    def __init__(self, db_path: str, output_dir: str = "data/crawled", num_workers: int = 4):
        self.db_path = db_path
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = mp.Event()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}. Shutting down workers...")
        self.stop_event.set()
        sys.exit(0)
    
    def get_domain_assignments(self) -> Dict[int, List[str]]:
        """Assign domains to workers for load balancing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get domains sorted by URL count (largest first)
        cursor.execute("""
            SELECT domain, url_count, domain_type 
            FROM domains 
            WHERE url_count > 0
            ORDER BY url_count DESC
        """)
        domains = cursor.fetchall()
        conn.close()
        
        # Round-robin assignment to balance load
        assignments = {i: [] for i in range(1, self.num_workers + 1)}
        
        for i, (domain, count, dtype) in enumerate(domains):
            worker_id = (i % self.num_workers) + 1
            assignments[worker_id].append(domain)
        
        # Print assignments
        print("Worker domain assignments:")
        for worker_id, domain_list in assignments.items():
            total_urls = sum(d[1] for d in domains if d[0] in domain_list)
            print(f"  Worker {worker_id}: {len(domain_list)} domains, {total_urls} URLs")
        
        return assignments
    
    def start_worker(self, worker_id: int, assigned_domains: List[str]):
        """Start a single worker process"""
        def worker_main():
            from single_crawler import SingleWorkerCrawler
            
            # Create specialized extractor
            crawler = SingleWorkerCrawler(
                db_path=self.db_path,
                output_dir=self.output_dir,
                worker_id=worker_id
            )
            
            # Replace extractor with domain-specific one
            crawler.extractor = DomainSpecificExtractor()
            
            print(f"Worker {worker_id} starting with domains: {assigned_domains[:3]}{'...' if len(assigned_domains) > 3 else ''}")
            
            while not self.stop_event.is_set():
                batch_processed = 0
                
                # Process each assigned domain in round-robin
                for domain in assigned_domains:
                    if self.stop_event.is_set():
                        break
                    
                    # Get URLs for this domain
                    urls = crawler.get_next_urls(domain_filter=domain, limit=5)
                    
                    if not urls:
                        continue
                    
                    for url_id, url, domain, frequency in urls:
                        if self.stop_event.is_set():
                            break
                        
                        print(f"Worker {worker_id}: {url} (freq: {frequency})")
                        
                        crawler.mark_processing(url_id)
                        
                        # Get delay and wait
                        delay = crawler.get_crawl_delay(domain)
                        time.sleep(delay)
                        
                        # Extract content
                        result = crawler.extractor.fetch_and_extract(url)
                        
                        # Update stats
                        crawler.stats['processed'] += 1
                        if result.status == 'success':
                            crawler.stats['successful'] += 1
                        elif result.status == 'partial':
                            crawler.stats['partial'] += 1
                        else:
                            crawler.stats['failed'] += 1
                        
                        # Save results
                        crawler.save_content(result)
                        crawler.mark_completed(url_id, result)
                        crawler.update_domain_last_crawled(domain)
                        
                        batch_processed += 1
                        
                        # Progress every 10 URLs
                        if crawler.stats['processed'] % 10 == 0:
                            crawler.print_progress()
                
                # If no URLs processed in this round, sleep before trying again
                if batch_processed == 0:
                    print(f"Worker {worker_id}: No URLs found, sleeping 30s...")
                    time.sleep(30)
            
            print(f"Worker {worker_id} shutting down")
        
        # Start worker process
        worker_process = mp.Process(target=worker_main)
        worker_process.start()
        return worker_process
    
    def monitor_progress(self):
        """Monitor overall progress across all workers"""
        def monitor_loop():
            while not self.stop_event.is_set():
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Get overall stats
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                        FROM urls
                    """)
                    total, pending, processing, success, failed = cursor.fetchone()
                    
                    # Get worker stats
                    cursor.execute("""
                        SELECT worker_id, COUNT(*) 
                        FROM urls 
                        WHERE worker_id IS NOT NULL 
                        GROUP BY worker_id
                    """)
                    worker_stats = dict(cursor.fetchall())
                    
                    conn.close()
                    
                    completed = success + failed
                    progress_pct = (completed / total * 100) if total > 0 else 0
                    
                    print(f"\n=== PROGRESS REPORT ===")
                    print(f"Total: {total:,} | Completed: {completed:,} ({progress_pct:.1f}%) | Pending: {pending:,}")
                    print(f"Success: {success:,} | Failed: {failed:,} | Processing: {processing:,}")
                    
                    for worker_id in range(1, self.num_workers + 1):
                        count = worker_stats.get(worker_id, 0)
                        print(f"Worker {worker_id}: {count:,} URLs")
                    
                    print("=" * 23)
                    
                    time.sleep(60)  # Update every minute
                    
                except Exception as e:
                    print(f"Monitor error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def run(self):
        """Start multi-worker crawling system"""
        print(f"Starting multi-worker crawler with {self.num_workers} workers")
        
        # Get domain assignments
        domain_assignments = self.get_domain_assignments()
        
        # Start progress monitor
        monitor_thread = self.monitor_progress()
        
        # Start all workers
        for worker_id in range(1, self.num_workers + 1):
            assigned_domains = domain_assignments[worker_id]
            worker_process = self.start_worker(worker_id, assigned_domains)
            self.workers.append(worker_process)
        
        print(f"All {len(self.workers)} workers started")
        
        try:
            # Wait for all workers to complete
            for worker in self.workers:
                worker.join()
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            self.stop_event.set()
            
            # Wait for workers to shutdown gracefully
            for worker in self.workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    print(f"Force terminating worker {worker.pid}")
                    worker.terminate()
        
        print("Multi-worker crawler completed")


def main():
    parser = argparse.ArgumentParser(description='Multi-worker crawler')
    parser.add_argument('--db', default='data/crawl_urls.db', help='Database path')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--output-dir', default='data/crawled', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found")
        return
    
    crawler = MultiWorkerCrawler(
        db_path=args.db,
        output_dir=args.output_dir,
        num_workers=args.workers
    )
    
    crawler.run()


if __name__ == "__main__":
    main()