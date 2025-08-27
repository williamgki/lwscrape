#!/usr/bin/env python3
"""
Single-worker crawler prototype for testing extraction pipeline
Starting with Wikipedia as the most reliable domain
"""

import requests
import sqlite3
import json
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import os
import argparse
from bs4 import BeautifulSoup
import re


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
    extract_method: Optional[str] = None  # which extractor worked
    metadata: Optional[Dict] = None


class ContentExtractor:
    """Content extraction pipeline with multiple strategies"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LW-Research-Crawler/1.0; Educational Use)'
        })
    
    def extract_wikipedia(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """Wikipedia-specific extraction"""
        try:
            # Get title
            title_elem = soup.find('h1', {'class': 'firstHeading'}) or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            # Get main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            # Remove unwanted elements
            for elem in content_div.find_all(['script', 'style', 'sup', '.navbox', '.infobox']):
                elem.decompose()
            
            # Extract text from paragraphs
            paragraphs = content_div.find_all('p')
            content_parts = []
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Skip very short paragraphs
                    content_parts.append(text)
            
            content = '\n\n'.join(content_parts)
            
            # Extract some metadata
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
            return CrawledContent(
                url=url,
                title="",
                content="",
                domain=urlparse(url).netloc,
                content_type='unknown',
                status='failed',
                scraped_at=datetime.now().isoformat(),
                content_length=0,
                error_message=str(e),
                extract_method='wikipedia_specific'
            )
    
    def extract_generic(self, soup: BeautifulSoup, url: str) -> Optional[CrawledContent]:
        """Generic content extraction fallback"""
        try:
            domain = urlparse(url).netloc
            
            # Try to get title
            title_elem = (soup.find('title') or 
                         soup.find('h1') or 
                         soup.find('h2'))
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            # Remove unwanted elements
            for elem in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                elem.decompose()
            
            # Try to find main content area
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
            
            # Extract text
            text = content_elem.get_text(separator='\n', strip=True)
            
            # Clean up text
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
            return CrawledContent(
                url=url,
                title="",
                content="",
                domain=urlparse(url).netloc,
                content_type='unknown',
                status='failed',
                scraped_at=datetime.now().isoformat(),
                content_length=0,
                error_message=str(e),
                extract_method='generic'
            )
    
    def fetch_and_extract(self, url: str, timeout: int = 30) -> Optional[CrawledContent]:
        """Fetch URL and extract content using appropriate method"""
        try:
            # Fetch page
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
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
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Choose extraction method based on domain
            domain = urlparse(url).netloc.lower()
            
            if 'wikipedia.org' in domain:
                return self.extract_wikipedia(soup, url)
            else:
                return self.extract_generic(soup, url)
                
        except requests.exceptions.RequestException as e:
            return CrawledContent(
                url=url,
                title="",
                content="",
                domain=urlparse(url).netloc,
                content_type='unknown',
                status='failed',
                scraped_at=datetime.now().isoformat(),
                content_length=0,
                error_message=f"Request failed: {str(e)}"
            )
        except Exception as e:
            return CrawledContent(
                url=url,
                title="",
                content="",
                domain=urlparse(url).netloc,
                content_type='unknown',
                status='failed',
                scraped_at=datetime.now().isoformat(),
                content_length=0,
                error_message=f"Unexpected error: {str(e)}"
            )


class SingleWorkerCrawler:
    """Single-worker crawler for testing"""
    
    def __init__(self, db_path: str, output_dir: str = "data/crawled", worker_id: int = 1):
        self.db_path = db_path
        self.output_dir = output_dir
        self.worker_id = worker_id
        self.output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        self.extractor = ContentExtractor()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'partial': 0,
            'start_time': datetime.now()
        }
    
    def get_database_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_next_urls(self, domain_filter: str = None, limit: int = 10) -> List[tuple]:
        """Get next URLs to crawl"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        # Build query
        where_clause = "status = 'pending' AND attempts < 3"
        params = []
        
        if domain_filter:
            where_clause += " AND domain = ?"
            params.append(domain_filter)
        
        query = f"""
            SELECT id, url, domain, frequency 
            FROM urls 
            WHERE {where_clause}
            ORDER BY frequency DESC, id ASC
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def mark_processing(self, url_id: int):
        """Mark URL as being processed"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE urls 
            SET status = 'processing', 
                worker_id = ?,
                attempts = attempts + 1,
                last_attempt = ?
            WHERE id = ?
        """, (self.worker_id, datetime.now().isoformat(), url_id))
        
        conn.commit()
        conn.close()
    
    def mark_completed(self, url_id: int, result: CrawledContent):
        """Mark URL as completed and save result info"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE urls 
            SET status = ?,
                success_timestamp = ?,
                file_path = ?,
                content_length = ?,
                error_message = ?
            WHERE id = ?
        """, (
            result.status,
            datetime.now().isoformat() if result.status == 'success' else None,
            self.output_file if result.status == 'success' else None,
            result.content_length,
            result.error_message,
            url_id
        ))
        
        conn.commit()
        conn.close()
    
    def save_content(self, content: CrawledContent):
        """Save crawled content to JSONL file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            json.dump(asdict(content), f, ensure_ascii=False)
            f.write('\n')
    
    def get_crawl_delay(self, domain: str) -> float:
        """Get appropriate crawl delay for domain"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT crawl_delay FROM domains WHERE domain = ?", (domain,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 2.0
    
    def update_domain_last_crawled(self, domain: str):
        """Update last crawled timestamp for domain"""
        conn = self.get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE domains 
            SET last_crawled = ? 
            WHERE domain = ?
        """, (datetime.now().isoformat(), domain))
        
        conn.commit()
        conn.close()
    
    def print_progress(self):
        """Print progress statistics"""
        elapsed = datetime.now() - self.stats['start_time']
        rate = self.stats['processed'] / elapsed.total_seconds() * 3600 if elapsed.total_seconds() > 0 else 0
        
        print(f"Progress: {self.stats['processed']} processed | "
              f"Success: {self.stats['successful']} | "
              f"Failed: {self.stats['failed']} | "
              f"Partial: {self.stats['partial']} | "
              f"Rate: {rate:.1f}/hour")
    
    def crawl_batch(self, domain_filter: str = None, batch_size: int = 10, max_urls: int = None):
        """Crawl a batch of URLs"""
        print(f"Starting single-worker crawler (ID: {self.worker_id})")
        if domain_filter:
            print(f"Domain filter: {domain_filter}")
        
        total_processed = 0
        
        while True:
            # Get next batch of URLs
            urls = self.get_next_urls(domain_filter, batch_size)
            if not urls:
                print("No more URLs to process")
                break
            
            print(f"\nProcessing batch of {len(urls)} URLs...")
            
            for url_id, url, domain, frequency in urls:
                # Check if we've hit the limit
                if max_urls and total_processed >= max_urls:
                    print(f"Reached maximum URL limit: {max_urls}")
                    return
                
                print(f"Crawling: {url} (freq: {frequency})")
                
                # Mark as processing
                self.mark_processing(url_id)
                
                # Get crawl delay and wait
                delay = self.get_crawl_delay(domain)
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
                self.mark_completed(url_id, result)
                self.update_domain_last_crawled(domain)
                
                # Print progress every 5 URLs
                if self.stats['processed'] % 5 == 0:
                    self.print_progress()
                
                total_processed += 1
        
        print(f"\nCrawling completed!")
        self.print_progress()
        
        # Final stats
        elapsed = datetime.now() - self.stats['start_time']
        print(f"Total time: {elapsed}")
        print(f"Output file: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Single-worker crawler for testing')
    parser.add_argument('--db', default='data/crawl_urls.db', help='Database path')
    parser.add_argument('--domain', help='Domain filter (e.g., en.wikipedia.org)')
    parser.add_argument('--max-urls', type=int, help='Maximum URLs to crawl')
    parser.add_argument('--worker-id', type=int, default=1, help='Worker ID')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Error: Database {args.db} not found")
        return
    
    crawler = SingleWorkerCrawler(
        db_path=args.db,
        worker_id=args.worker_id
    )
    
    crawler.crawl_batch(
        domain_filter=args.domain,
        batch_size=args.batch_size,
        max_urls=args.max_urls
    )


if __name__ == "__main__":
    main()