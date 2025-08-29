#!/usr/bin/env python3
"""
PDF Crawler for LessWrong DDL Experiment
Extracts text content from PDF URLs that failed in main crawler
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

# PDF processing libraries
import PyPDF2
import fitz  # pymupdf

class PDFExtractor:
    """Extract text from PDF files using multiple methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LessWrong-Research/1.0)'
        })
        
    def download_pdf(self, url: str, timeout: int = 30) -> Optional[bytes]:
        """Download PDF content from URL"""
        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.endswith('.pdf'):
                return None
                
            # Download with size limit (50MB)
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 50 * 1024 * 1024:  # 50MB limit
                    raise ValueError("PDF too large")
                    
            return content
            
        except Exception as e:
            logging.warning(f"Download failed for {url}: {e}")
            return None
    
    def extract_with_pymupdf(self, pdf_content: bytes) -> Optional[str]:
        """Extract text using PyMuPDF (fitz) - generally better"""
        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                doc = fitz.open(tmp_file.name)
                text_parts = []
                
                for page_num in range(min(len(doc), 50)):  # Limit to 50 pages
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"Page {page_num + 1}:\n{text}")
                
                doc.close()
                
                if text_parts:
                    return "\n\n".join(text_parts)
                return None
                
        except Exception as e:
            logging.warning(f"PyMuPDF extraction failed: {e}")
            return None
    
    def extract_with_pypdf2(self, pdf_content: bytes) -> Optional[str]:
        """Extract text using PyPDF2 - fallback method"""
        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                with open(tmp_file.name, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    
                    for page_num, page in enumerate(reader.pages[:50]):  # Limit to 50 pages
                        try:
                            text = page.extract_text()
                            if text.strip():
                                text_parts.append(f"Page {page_num + 1}:\n{text}")
                        except Exception as e:
                            logging.warning(f"Failed to extract page {page_num}: {e}")
                            continue
                    
                    if text_parts:
                        return "\n\n".join(text_parts)
                    return None
                    
        except Exception as e:
            logging.warning(f"PyPDF2 extraction failed: {e}")
            return None
    
    def extract_pdf_text(self, url: str) -> Dict[str, Any]:
        """Main extraction method - tries multiple approaches"""
        start_time = time.time()
        
        result = {
            'url': url,
            'status': 'failed',
            'content': None,
            'metadata': {
                'pages': 0,
                'file_size': 0,
                'extraction_method': None,
                'processing_time': 0,
                'timestamp': datetime.now().isoformat()
            },
            'error': None
        }
        
        try:
            # Download PDF
            logging.info(f"Downloading: {url}")
            pdf_content = self.download_pdf(url)
            
            if not pdf_content:
                result['error'] = "Failed to download PDF"
                return result
            
            result['metadata']['file_size'] = len(pdf_content)
            
            # Try PyMuPDF first (generally better)
            text = self.extract_with_pymupdf(pdf_content)
            if text:
                result['content'] = text
                result['status'] = 'success'
                result['metadata']['extraction_method'] = 'pymupdf'
                result['metadata']['pages'] = text.count('Page ')
            else:
                # Fallback to PyPDF2
                text = self.extract_with_pypdf2(pdf_content)
                if text:
                    result['content'] = text
                    result['status'] = 'success' 
                    result['metadata']['extraction_method'] = 'pypdf2'
                    result['metadata']['pages'] = text.count('Page ')
                else:
                    result['error'] = "Text extraction failed with both methods"
            
        except Exception as e:
            result['error'] = str(e)
            logging.error(f"PDF extraction error for {url}: {e}")
        
        result['metadata']['processing_time'] = time.time() - start_time
        return result

def main():
    """Main PDF crawler function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdf_crawler.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load PDF URLs
    pdf_urls = []
    try:
        with open('pdf_urls.txt', 'r') as f:
            for line in f:
                url, domain, frequency = line.strip().split('\t')
                pdf_urls.append((url, domain, int(frequency)))
    except Exception as e:
        logging.error(f"Failed to load pdf_urls.txt: {e}")
        return
    
    # Sort by frequency (most referenced first)
    pdf_urls.sort(key=lambda x: x[2], reverse=True)
    
    logging.info(f"Starting PDF crawler for {len(pdf_urls)} URLs")
    
    # Create output directory
    output_dir = Path('data/pdfs')
    output_dir.mkdir(exist_ok=True)
    
    extractor = PDFExtractor()
    processed = 0
    successful = 0
    
    # Output file for results
    output_file = output_dir / 'extracted_pdfs.jsonl'
    
    try:
        with open(output_file, 'w') as outf:
            for url, domain, frequency in pdf_urls:
                processed += 1
                
                logging.info(f"[{processed}/{len(pdf_urls)}] Processing: {url} (freq: {frequency})")
                
                result = extractor.extract_pdf_text(url)
                
                if result['status'] == 'success':
                    successful += 1
                    pages = result['metadata']['pages']
                    size_kb = result['metadata']['file_size'] / 1024
                    logging.info(f"  ✓ Success: {pages} pages, {size_kb:.1f}KB")
                else:
                    logging.warning(f"  ✗ Failed: {result['error']}")
                
                # Save result
                outf.write(json.dumps(result) + '\n')
                outf.flush()
                
                # Progress report every 10 PDFs
                if processed % 10 == 0:
                    success_rate = (successful / processed * 100) if processed > 0 else 0
                    logging.info(f"Progress: {processed}/{len(pdf_urls)} ({success_rate:.1f}% success)")
                
                # Rate limiting
                time.sleep(1)  # Be respectful
                
    except KeyboardInterrupt:
        logging.info("PDF crawler interrupted by user")
    except Exception as e:
        logging.error(f"PDF crawler error: {e}")
    finally:
        success_rate = (successful / processed * 100) if processed > 0 else 0
        logging.info(f"PDF crawler completed: {processed} processed, {successful} successful ({success_rate:.1f}%)")

if __name__ == '__main__':
    main()