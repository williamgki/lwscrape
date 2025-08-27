#!/usr/bin/env python3
"""
Step 1: Extract unique external URLs from scraped LW data and build crawling database
"""

import csv
import sqlite3
import json
from collections import defaultdict, Counter
from urllib.parse import urlparse
import re
from datetime import datetime
import os


class URLDatabaseBuilder:
    def __init__(self, csv_path: str, db_path: str = "data/crawl_urls.db"):
        self.csv_path = csv_path
        self.db_path = db_path
        self.domain_types = {
            'academic': [
                'arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov', 
                'semanticscholar.org', 'researchgate.net', 'jstor.org',
                'springer.com', 'sciencedirect.com', 'nature.com', 'science.org',
                'acm.org', 'ieee.org', 'aaai.org', 'neurips.cc', 'openreview.net'
            ],
            'wikipedia': [
                'wikipedia.org', 'wikidata.org', 'wikimedia.org'
            ],
            'news': [
                'nytimes.com', 'economist.com', 'reuters.com', 'bbc.com', 'cnn.com',
                'wsj.com', 'ft.com', 'guardian.co.uk', 'washingtonpost.com',
                'bloomberg.com', 'npr.org', 'axios.com', 'vox.com'
            ],
            'social': [
                'twitter.com', 'x.com', 'youtube.com', 'reddit.com', 'facebook.com',
                'instagram.com', 'linkedin.com', 'tiktok.com', 'discord.com'
            ],
            'blogs': [
                'substack.com', 'medium.com', 'wordpress.com', 'blogspot.com',
                'tumblr.com', 'ghost.org', 'notion.so'
            ],
            'technical': [
                'github.com', 'gitlab.com', 'stackoverflow.com', 'stackexchange.com',
                'docs.google.com', 'colab.research.google.com', 'kaggle.com',
                'huggingface.co', 'openai.com', 'anthropic.com'
            ],
            'books': [
                'amazon.com', 'goodreads.com', 'books.google.com', 'archive.org',
                'gutenberg.org', 'libgen.rs', 'z-lib.org'
            ]
        }
        
    def extract_urls_from_csv(self):
        """Extract all external URLs with metadata from CSV files"""
        print(f"Extracting URLs from {self.csv_path}")
        
        urls_data = defaultdict(lambda: {
            'frequency': 0,
            'source_posts': set(),
            'contexts': [],
            'link_texts': set()
        })
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i % 50000 == 0:
                        print(f"Processed {i:,} rows...")
                    
                    if row['link_type'] == 'external':
                        url = row['target_url'].strip()
                        
                        # Basic URL cleaning
                        if not url.startswith(('http://', 'https://')):
                            continue
                            
                        urls_data[url]['frequency'] += 1
                        urls_data[url]['source_posts'].add(row['source_post_id'])
                        
                        # Store context (truncate if too long)
                        context = row.get('context', '').strip()
                        if context and len(context) > 10:
                            urls_data[url]['contexts'].append(context[:200])
                            
                        # Store link text
                        link_text = row.get('link_text', '').strip()
                        if link_text:
                            urls_data[url]['link_texts'].add(link_text[:100])
                
        except FileNotFoundError:
            print(f"Error: Could not find {self.csv_path}")
            return {}
        
        print(f"Extracted {len(urls_data):,} unique external URLs")
        return urls_data
    
    def classify_domain(self, url):
        """Classify domain into type category"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check each category
            for category, domains in self.domain_types.items():
                for pattern in domains:
                    if pattern in domain or domain.endswith(pattern):
                        return category, domain
                        
            return 'general', domain
            
        except Exception:
            return 'unknown', 'unknown'
    
    def analyze_domains(self, urls_data):
        """Analyze domain distribution and generate statistics"""
        domain_stats = defaultdict(lambda: {
            'count': 0, 
            'total_frequency': 0,
            'type': 'unknown',
            'urls': []
        })
        
        type_stats = defaultdict(int)
        
        for url, data in urls_data.items():
            domain_type, domain = self.classify_domain(url)
            
            domain_stats[domain]['count'] += 1
            domain_stats[domain]['total_frequency'] += data['frequency']
            domain_stats[domain]['type'] = domain_type
            domain_stats[domain]['urls'].append({
                'url': url,
                'frequency': data['frequency']
            })
            
            type_stats[domain_type] += 1
        
        return dict(domain_stats), dict(type_stats)
    
    def create_database(self):
        """Create SQLite database with proper schema"""
        print(f"Creating database at {self.db_path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create URLs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                domain_type TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                source_posts_count INTEGER NOT NULL,
                link_texts TEXT,  -- JSON array of link texts
                contexts TEXT,    -- JSON array of contexts
                status TEXT DEFAULT 'pending',
                worker_id INTEGER,
                attempts INTEGER DEFAULT 0,
                last_attempt TIMESTAMP,
                success_timestamp TIMESTAMP,
                file_path TEXT,
                error_message TEXT,
                content_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create domains table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domains (
                domain TEXT PRIMARY KEY,
                domain_type TEXT NOT NULL,
                url_count INTEGER NOT NULL,
                total_frequency INTEGER NOT NULL,
                crawl_delay REAL DEFAULT 2.0,
                worker_id INTEGER,
                last_crawled TIMESTAMP,
                robots_txt_checked BOOLEAN DEFAULT FALSE,
                robots_txt_allowed BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_urls_domain ON urls(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_urls_frequency ON urls(frequency DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domains_type ON domains(domain_type)')
        
        conn.commit()
        return conn
    
    def populate_database(self, urls_data, domain_stats):
        """Populate database with extracted URL data"""
        print("Populating database...")
        
        conn = self.create_database()
        cursor = conn.cursor()
        
        # Populate domains table
        domain_rows = []
        for domain, stats in domain_stats.items():
            domain_rows.append((
                domain,
                stats['type'],
                stats['count'],
                stats['total_frequency'],
                self.get_crawl_delay(stats['type'])
            ))
        
        cursor.executemany('''
            INSERT OR REPLACE INTO domains 
            (domain, domain_type, url_count, total_frequency, crawl_delay)
            VALUES (?, ?, ?, ?, ?)
        ''', domain_rows)
        
        # Populate URLs table
        url_rows = []
        for url, data in urls_data.items():
            domain_type, domain = self.classify_domain(url)
            
            url_rows.append((
                url,
                domain,
                domain_type,
                data['frequency'],
                len(data['source_posts']),
                json.dumps(list(data['link_texts'])[:10]),  # Limit to 10 link texts
                json.dumps(data['contexts'][:10])  # Limit to 10 contexts
            ))
        
        cursor.executemany('''
            INSERT OR REPLACE INTO urls 
            (url, domain, domain_type, frequency, source_posts_count, link_texts, contexts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', url_rows)
        
        conn.commit()
        conn.close()
        
        print(f"Populated database with {len(url_rows):,} URLs and {len(domain_rows):,} domains")
    
    def get_crawl_delay(self, domain_type):
        """Get appropriate crawl delay for domain type"""
        delays = {
            'academic': 3.0,    # Be respectful to academic servers
            'wikipedia': 1.0,   # Wikipedia can handle more load
            'news': 2.0,        # News sites moderate load
            'social': 2.5,      # Social platforms moderate load  
            'blogs': 1.5,       # Personal blogs are usually lighter
            'technical': 2.0,   # Technical sites moderate load
            'books': 3.0,       # Book sites often slower
            'general': 2.5,     # Conservative default
            'unknown': 3.0      # Very conservative for unknown
        }
        return delays.get(domain_type, 2.5)
    
    def generate_statistics(self, urls_data, domain_stats, type_stats):
        """Generate comprehensive statistics about the URL dataset"""
        stats = {
            'total_urls': len(urls_data),
            'total_references': sum(data['frequency'] for data in urls_data.values()),
            'domains_count': len(domain_stats),
            'domain_types': type_stats,
            'top_domains': [],
            'top_urls': [],
            'crawl_estimates': {}
        }
        
        # Top domains by URL count
        top_domains = sorted(domain_stats.items(), 
                           key=lambda x: x[1]['count'], reverse=True)[:20]
        stats['top_domains'] = [(domain, data['count'], data['type']) 
                               for domain, data in top_domains]
        
        # Top URLs by frequency  
        top_urls = sorted(urls_data.items(), 
                         key=lambda x: x[1]['frequency'], reverse=True)[:20]
        stats['top_urls'] = [(url, data['frequency']) 
                            for url, data in top_urls]
        
        # Crawl time estimates
        for domain_type, count in type_stats.items():
            delay = self.get_crawl_delay(domain_type)
            estimated_hours = (count * delay) / 3600
            stats['crawl_estimates'][domain_type] = {
                'count': count,
                'delay_seconds': delay,
                'estimated_hours': round(estimated_hours, 1)
            }
        
        return stats
    
    def save_statistics(self, stats):
        """Save statistics to JSON file"""
        stats_path = "data/crawl_statistics.json"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics saved to {stats_path}")
        return stats_path
    
    def print_summary(self, stats):
        """Print summary statistics to console"""
        print("\n" + "="*60)
        print("URL DATABASE SUMMARY")
        print("="*60)
        
        print(f"Total unique URLs: {stats['total_urls']:,}")
        print(f"Total references: {stats['total_references']:,}")
        print(f"Unique domains: {stats['domains_count']:,}")
        
        print(f"\nDomain types:")
        for dtype, count in sorted(stats['domain_types'].items(), 
                                  key=lambda x: x[1], reverse=True):
            est = stats['crawl_estimates'].get(dtype, {})
            hours = est.get('estimated_hours', 0)
            print(f"  {dtype:12}: {count:5,} URLs (~{hours:4.1f}h to crawl)")
        
        print(f"\nTop domains by URL count:")
        for domain, count, dtype in stats['top_domains'][:10]:
            print(f"  {domain:25} {count:4} URLs ({dtype})")
        
        print(f"\nTop URLs by reference frequency:")
        for url, freq in stats['top_urls'][:5]:
            print(f"  {freq:3}x {url[:70]}...")
    
    def run(self):
        """Execute the complete URL database building process"""
        print("Starting URL database build process...")
        
        # Step 1: Extract URLs from CSV
        urls_data = self.extract_urls_from_csv()
        if not urls_data:
            print("No URLs extracted. Exiting.")
            return
        
        # Step 2: Analyze domains
        domain_stats, type_stats = self.analyze_domains(urls_data)
        
        # Step 3: Create and populate database
        self.populate_database(urls_data, domain_stats)
        
        # Step 4: Generate statistics
        stats = self.generate_statistics(urls_data, domain_stats, type_stats)
        
        # Step 5: Save and display results
        self.save_statistics(stats)
        self.print_summary(stats)
        
        print(f"\nDatabase created at: {self.db_path}")
        print("Ready for crawler implementation!")


if __name__ == "__main__":
    # Build database from the incremental links file
    csv_path = "data/links/links_incremental_20250816_093702.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print("Available files:")
        import glob
        for f in glob.glob("data/links/*.csv"):
            print(f"  {f}")
        exit(1)
    
    builder = URLDatabaseBuilder(csv_path)
    builder.run()