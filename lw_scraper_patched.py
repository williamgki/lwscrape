#!/usr/bin/env python3
"""
GreaterWrong/LessWrong Scraper - PATCHED VERSION
Scrapes posts chronologically with efficient link extraction + incremental link saving
"""

import requests
import time
import json
import csv
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Set, Optional
import re
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import os


@dataclass
class Post:
    """Data structure for a LessWrong post"""
    post_id: str
    title: str
    author: str
    date: str
    url: str
    content: str
    points: int
    comment_count: int
    read_time: str
    tags: List[str]
    internal_links: List[str]  # Links to other LW posts
    external_links: List[str]  # Links to external sites
    scraped_at: str


@dataclass
class LinkData:
    """Data structure for extracted links"""
    source_post_id: str
    source_url: str
    target_url: str
    link_text: str
    link_type: str  # 'internal' or 'external'
    context: str  # Surrounding text for context


class LWScraper:
    def __init__(self, base_url: str = "https://www.greaterwrong.com", delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LW Research Scraper 1.0 (Educational/Research Use)'
        })
        
        # Create output directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/posts', exist_ok=True)
        os.makedirs('data/links', exist_ok=True)
        
        # Track scraped posts to avoid duplicates
        self.scraped_posts: Set[str] = set()
        self.all_links: List[LinkData] = []
        
        # NEW: Counter for incremental link saving
        self.links_saved_count = 0
        self.incremental_links_file = f"data/links/links_incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Load previously scraped post IDs to avoid re-processing
        self.load_existing_post_ids()
    
    def load_existing_post_ids(self):
        """Load post IDs from existing JSON files to avoid re-scraping"""
        import glob
        json_files = glob.glob('data/posts/*.json')
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    posts = json.load(f)
                    for post in posts:
                        if isinstance(post, dict) and 'post_id' in post:
                            self.scraped_posts.add(post['post_id'])
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        print(f"Loaded {len(self.scraped_posts)} previously scraped post IDs")
        
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page with rate limiting"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_links_from_content(self, soup: BeautifulSoup, post_id: str, post_url: str) -> tuple[List[str], List[str], List[LinkData]]:
        """Extract all links from post content with detailed metadata"""
        internal_links = []
        external_links = []
        link_data = []
        
        # Find all links in the content area
        content_area = soup.find('div', class_='post-content') or soup.find('div', class_='body') or soup
        
        for link in content_area.find_all('a', href=True):
            href = link.get('href')
            if not href:
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(self.base_url, href)
            link_text = link.get_text(strip=True)
            
            # Get context (surrounding text)
            context = ""
            if link.parent:
                parent_text = link.parent.get_text()
                # Find link position and extract context
                link_pos = parent_text.find(link_text)
                if link_pos >= 0:
                    start = max(0, link_pos - 50)
                    end = min(len(parent_text), link_pos + len(link_text) + 50)
                    context = parent_text[start:end].strip()
            
            # Classify link type
            parsed_url = urlparse(full_url)
            is_internal = (parsed_url.netloc in ['www.greaterwrong.com', 'greaterwrong.com', 
                          'www.lesswrong.com', 'lesswrong.com'] or 
                          parsed_url.netloc == '')
            
            link_type = 'internal' if is_internal else 'external'
            
            # Store link data
            link_data.append(LinkData(
                source_post_id=post_id,
                source_url=post_url,
                target_url=full_url,
                link_text=link_text,
                link_type=link_type,
                context=context
            ))
            
            if is_internal:
                internal_links.append(full_url)
            else:
                external_links.append(full_url)
        
        return internal_links, external_links, link_data
    
    def parse_post_from_archive(self, post_element) -> Optional[str]:
        """Extract post ID from archive listing"""
        link = post_element.find('a')
        if link and link.get('href'):
            href = link.get('href')
            # Extract post ID from URL pattern /posts/{id}/{slug}
            match = re.search(r'/posts/([^/]+)/', href)
            if match:
                return match.group(1)
        return None
    
    def scrape_post(self, post_id: str) -> Optional[Post]:
        """Scrape individual post content"""
        if post_id in self.scraped_posts:
            return None
            
        # Try different URL patterns
        post_urls = [
            f"{self.base_url}/posts/{post_id}",
            f"{self.base_url}/posts/{post_id}/",
        ]
        
        soup = None
        post_url = None
        for url in post_urls:
            soup = self.get_page(url)
            if soup:
                post_url = url
                break
        
        if not soup:
            print(f"Could not fetch post {post_id}")
            return None
        
        try:
            # Extract post metadata
            title = soup.find('h1', class_='post-title')
            title = title.get_text(strip=True) if title else "Unknown Title"
            
            author_elem = soup.find('a', class_='author') or soup.find('.byline a')
            author = author_elem.get_text(strip=True) if author_elem else "Unknown Author"
            
            date_elem = soup.find('time') or soup.find('.date')
            date = date_elem.get('datetime') if date_elem and date_elem.get('datetime') else ""
            if not date and date_elem:
                date = date_elem.get_text(strip=True)
            
            # Extract content
            content_elem = (soup.find('div', class_='post-content') or 
                           soup.find('div', class_='body') or 
                           soup.find('div', class_='post-body'))
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            # Extract metadata
            points_elem = soup.find('.points') or soup.find('.score')
            points = 0
            if points_elem:
                points_text = points_elem.get_text()
                points_match = re.search(r'(\d+)', points_text)
                points = int(points_match.group(1)) if points_match else 0
            
            comment_count_elem = soup.find('.comments-count')
            comment_count = 0
            if comment_count_elem:
                comment_text = comment_count_elem.get_text()
                comment_match = re.search(r'(\d+)', comment_text)
                comment_count = int(comment_match.group(1)) if comment_match else 0
            
            read_time = soup.find('.read-time')
            read_time = read_time.get_text(strip=True) if read_time else ""
            
            # Extract tags
            tags = []
            tag_elements = soup.find_all('.tag') or soup.find_all('[class*="tag"]')
            for tag in tag_elements:
                tag_text = tag.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)
            
            # Extract links
            internal_links, external_links, link_data = self.extract_links_from_content(soup, post_id, post_url)
            self.all_links.extend(link_data)
            
            post = Post(
                post_id=post_id,
                title=title,
                author=author,
                date=date,
                url=post_url,
                content=content,
                points=points,
                comment_count=comment_count,
                read_time=read_time,
                tags=tags,
                internal_links=internal_links,
                external_links=external_links,
                scraped_at=datetime.now().isoformat()
            )
            
            self.scraped_posts.add(post_id)
            
            # NEW: Check if we should save links incrementally
            self.save_links_incrementally()
            
            return post
            
        except Exception as e:
            print(f"Error parsing post {post_id}: {e}")
            return None
    
    def save_links_incrementally(self):
        """NEW: Save links incrementally every 100 new links"""
        links_to_save = len(self.all_links) - self.links_saved_count
        
        if links_to_save >= 100:  # Save every 100 new links
            # Append new links to CSV file
            new_links = self.all_links[self.links_saved_count:]
            
            # Create file with header if it doesn't exist
            file_exists = os.path.exists(self.incremental_links_file)
            
            with open(self.incremental_links_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if new file
                if not file_exists:
                    writer.writerow(['source_post_id', 'source_url', 'target_url', 'link_text', 'link_type', 'context'])
                
                # Write new links
                for link in new_links:
                    writer.writerow([
                        link.source_post_id,
                        link.source_url,
                        link.target_url,
                        link.link_text,
                        link.link_type,
                        link.context
                    ])
            
            print(f"Saved {len(new_links)} links incrementally to {self.incremental_links_file}")
            self.links_saved_count = len(self.all_links)
    
    def scrape_archive_page(self, offset: int = 0) -> List[str]:
        """Scrape post IDs from archive page"""
        url = f"{self.base_url}/archive?offset={offset}"
        soup = self.get_page(url)
        
        if not soup:
            return []
        
        post_ids = []
        
        # Look for post links in various possible structures
        post_links = soup.find_all('a', href=re.compile(r'/posts/[^/]+'))
        
        for link in post_links:
            href = link.get('href')
            match = re.search(r'/posts/([^/]+)', href)
            if match:
                post_id = match.group(1)
                if post_id not in post_ids:
                    post_ids.append(post_id)
        
        return post_ids
    
    def save_posts_to_json(self, posts: List[Post], filename: str = None):
        """Save posts to JSON file"""
        if not filename:
            filename = f"data/posts/posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        posts_data = [asdict(post) for post in posts]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(posts)} posts to {filename}")
    
    def save_links_to_csv(self, filename: str = None):
        """Save extracted links to CSV"""
        if not filename:
            filename = f"data/links/links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.all_links:
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source_post_id', 'source_url', 'target_url', 'link_text', 'link_type', 'context'])
            
            for link in self.all_links:
                writer.writerow([
                    link.source_post_id,
                    link.source_url,
                    link.target_url,
                    link.link_text,
                    link.link_type,
                    link.context
                ])
        
        print(f"Saved {len(self.all_links)} links to {filename}")
    
    def scrape_chronologically(self, max_pages: int = 10, start_offset: int = 0):
        """Main scraping function - goes through archive chronologically"""
        all_posts = []
        
        for page in range(max_pages):
            offset = start_offset + (page * 20)
            print(f"Scraping archive page {page + 1} (offset: {offset})")
            
            post_ids = self.scrape_archive_page(offset)
            
            if not post_ids:
                print("No more posts found")
                break
            
            print(f"Found {len(post_ids)} posts on this page")
            
            # Filter out already scraped posts before processing
            new_post_ids = [pid for pid in post_ids if pid not in self.scraped_posts]
            if len(new_post_ids) < len(post_ids):
                print(f"Skipping {len(post_ids) - len(new_post_ids)} already scraped posts")
            
            # Scrape each new post
            for i, post_id in enumerate(new_post_ids):
                print(f"Scraping post {i+1}/{len(new_post_ids)}: {post_id}")
                post = self.scrape_post(post_id)
                if post:
                    all_posts.append(post)
                    
                    # Save incrementally every 10 posts
                    if len(all_posts) % 10 == 0:
                        self.save_posts_to_json(all_posts[-10:], 
                                               f"data/posts/batch_{len(all_posts)//10}.json")
        
        # Save final results
        if all_posts:
            self.save_posts_to_json(all_posts)
        
        # Final incremental save of any remaining links
        if len(self.all_links) > self.links_saved_count:
            self.save_links_incrementally()
        
        # Also save complete links file
        if self.all_links:
            self.save_links_to_csv()
        
        print(f"\nScraping complete!")
        print(f"Total posts scraped: {len(all_posts)}")
        print(f"Total links extracted: {len(self.all_links)}")
        print(f"Internal links: {sum(1 for link in self.all_links if link.link_type == 'internal')}")
        print(f"External links: {sum(1 for link in self.all_links if link.link_type == 'external')}")


if __name__ == "__main__":
    scraper = LWScraper(delay=1.5)  # Be respectful with 1.5s delay
    
    # Start scraping from the beginning (most recent posts)
    scraper.scrape_chronologically(max_pages=5000, start_offset=0)