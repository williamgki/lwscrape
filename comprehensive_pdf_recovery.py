#!/usr/bin/env python3
"""
Comprehensive PDF Recovery - Extract all 1,889 missed PDFs
Prioritized by reference frequency for DDL experiment
"""

import os
import sys
import json
import time
import requests
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from datetime import datetime
import concurrent.futures
import threading

# PDF processing libraries
import PyPDF2
import fitz  # pymupdf

class HighSpeedPDFRecovery:
    """Multi-threaded PDF recovery system"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.session_pool = []
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': time.time()
        }
        self.stats_lock = threading.Lock()
        
        # Create session pool
        for _ in range(max_workers):
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; LessWrong-Research/2.0)'
            })
            self.session_pool.append(session)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pdf_recovery.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_session(self) -> requests.Session:
        """Get a session from the pool"""
        return self.session_pool[threading.current_thread().ident % len(self.session_pool)]
    
    def download_pdf(self, url: str, timeout: int = 20) -> Optional[bytes]:
        """Download PDF with optimizations"""
        try:
            session = self.get_session()
            response = session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.endswith('.pdf'):
                return None
            
            # Download with 20MB limit
            content = b''
            for chunk in response.iter_content(chunk_size=16384):  # Larger chunks
                content += chunk
                if len(content) > 20 * 1024 * 1024:  # 20MB limit
                    raise ValueError("PDF too large")
            
            return content
            
        except Exception as e:
            self.logger.warning(f"Download failed for {url}: {e}")
            return None
    
    def extract_pdf_fast(self, pdf_content: bytes) -> Optional[str]:
        """Fast PDF extraction with PyMuPDF"""
        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                doc = fitz.open(tmp_file.name)
                text_parts = []
                
                # Limit pages but extract efficiently
                max_pages = min(len(doc), 30)  # Reasonable limit
                for page_num in range(max_pages):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(text)
                
                doc.close()
                return "\\n\\n".join(text_parts) if text_parts else None
                
        except Exception as e:
            # Fallback to PyPDF2
            try:
                with tempfile.NamedTemporaryFile() as tmp_file:
                    tmp_file.write(pdf_content)
                    tmp_file.flush()
                    
                    with open(tmp_file.name, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text_parts = []
                        
                        max_pages = min(len(reader.pages), 30)
                        for page in reader.pages[:max_pages]:
                            try:
                                text = page.extract_text()
                                if text.strip():
                                    text_parts.append(text)
                            except:
                                continue
                        
                        return "\\n\\n".join(text_parts) if text_parts else None
                        
            except Exception as e2:
                self.logger.warning(f"Both extraction methods failed: {e}, {e2}")
                return None
    
    def process_single_pdf(self, pdf_data: tuple) -> Dict[str, Any]:
        """Process a single PDF - optimized for speed"""
        url, domain, frequency = pdf_data
        start_time = time.time()
        
        result = {
            'url': url,
            'domain': domain,
            'frequency': frequency,
            'status': 'failed',
            'content': None,
            'metadata': {
                'file_size': 0,
                'processing_time': 0,
                'timestamp': datetime.now().isoformat()
            },
            'error': None
        }
        
        try:
            # Download
            pdf_content = self.download_pdf(url)
            if not pdf_content:
                result['error'] = "Download failed"
                return result
            
            result['metadata']['file_size'] = len(pdf_content)
            
            # Extract text
            text = self.extract_pdf_fast(pdf_content)
            if text:
                result['content'] = text
                result['status'] = 'success'
                
                with self.stats_lock:
                    self.stats['successful'] += 1
            else:
                result['error'] = "Text extraction failed"
                with self.stats_lock:
                    self.stats['failed'] += 1
            
        except Exception as e:
            result['error'] = str(e)
            with self.stats_lock:
                self.stats['failed'] += 1
        
        result['metadata']['processing_time'] = time.time() - start_time
        
        with self.stats_lock:
            self.stats['processed'] += 1
        
        return result
    
    def run_recovery(self):
        """Run comprehensive PDF recovery"""
        # Load missed PDFs
        pdf_urls = []
        try:
            with open('missed_pdfs_comprehensive.txt', 'r') as f:
                for line in f:
                    parts = line.strip().split('\\t')
                    if len(parts) >= 3:
                        url, domain, frequency = parts[0], parts[1], int(parts[2])
                        pdf_urls.append((url, domain, frequency))
        except Exception as e:
            self.logger.error(f"Failed to load missed PDFs: {e}")
            return
        
        # Sort by frequency (high-priority first)
        pdf_urls.sort(key=lambda x: x[2], reverse=True)
        
        self.logger.info(f"Starting recovery of {len(pdf_urls)} PDFs with {self.max_workers} workers")
        
        # Create output directory
        output_dir = Path('data/pdfs_recovered')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'recovered_pdfs.jsonl'
        
        # Process with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with open(output_file, 'w', encoding='utf-8') as outf:
                # Submit all tasks
                future_to_pdf = {executor.submit(self.process_single_pdf, pdf_data): pdf_data 
                               for pdf_data in pdf_urls}
                
                for future in concurrent.futures.as_completed(future_to_pdf):
                    result = future.result()
                    
                    # Save result
                    outf.write(json.dumps(result) + '\\n')
                    outf.flush()
                    
                    # Progress logging
                    if result['status'] == 'success':
                        content_len = len(result.get('content', ''))
                        self.logger.info(f"✓ {result['domain']}: {content_len:,} chars (freq: {result['frequency']})")
                    
                    # Progress report every 50 PDFs
                    if self.stats['processed'] % 50 == 0:
                        elapsed = time.time() - self.stats['start_time']
                        rate = self.stats['processed'] / elapsed * 3600  # per hour
                        success_rate = self.stats['successful'] / self.stats['processed'] * 100
                        
                        self.logger.info(f"Progress: {self.stats['processed']}/{len(pdf_urls)} "
                                       f"({success_rate:.1f}% success, {rate:.0f}/hour)")
                        
                        # ETA calculation
                        remaining = len(pdf_urls) - self.stats['processed']
                        eta_hours = remaining / (rate / 3600) if rate > 0 else 0
                        self.logger.info(f"ETA: {eta_hours:.1f} hours")
        
        # Final stats
        elapsed = time.time() - self.stats['start_time']
        success_rate = self.stats['successful'] / self.stats['processed'] * 100 if self.stats['processed'] > 0 else 0
        
        self.logger.info(f"Recovery completed: {self.stats['processed']} processed, "
                        f"{self.stats['successful']} successful ({success_rate:.1f}%), "
                        f"Time: {elapsed/3600:.1f} hours")

def main():
    recovery = HighSpeedPDFRecovery(max_workers=6)  # Aggressive recovery
    recovery.run_recovery()

if __name__ == '__main__':
    main()