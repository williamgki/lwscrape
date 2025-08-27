#!/usr/bin/env python3
"""
Patch for multi_crawler.py to add live PDF detection
This modifies the DomainSpecificExtractor to queue PDFs when detected
"""

import os
import sys
from urllib.parse import urlparse
from pdf_queue_utils import PDFQueueManager

# Add PDF queue manager to the extractor
class PDFAwareDomainExtractor:
    """Extended extractor that queues PDFs for later processing"""
    
    def __init__(self, db_path: str = "data/crawl_urls.db"):
        self.pdf_queue = PDFQueueManager(db_path)
        # Keep track of PDFs detected
        self.pdfs_detected = 0
    
    def handle_non_html_content(self, url: str, content_type: str, original_error: str):
        """Enhanced non-HTML handler that queues PDFs"""
        
        # Check if this is a PDF
        if self.pdf_queue.is_pdf_content_type(content_type):
            # Add to PDF queue for later processing
            if self.pdf_queue.add_pdf_to_queue(url, content_type):
                self.pdfs_detected += 1
                if self.pdfs_detected % 10 == 0:
                    print(f"Queued {self.pdfs_detected} PDFs for processing")
                
                # Return a special status indicating PDF was queued
                from multi_crawler import CrawledContent
                from datetime import datetime
                return CrawledContent(
                    url=url,
                    title="PDF queued for processing",
                    content="",
                    domain=urlparse(url).netloc,
                    content_type=content_type,
                    status='pdf_queued',  # Special status
                    scraped_at=datetime.now().isoformat(),
                    content_length=0,
                    error_message=f"PDF queued: {content_type}"
                )
        
        # For non-PDF files, return the original error
        from multi_crawler import CrawledContent
        from datetime import datetime
        return CrawledContent(
            url=url,
            title="Non-HTML content",
            content="",
            domain=urlparse(url).netloc,
            content_type=content_type,
            status='failed',
            scraped_at=datetime.now().isoformat(),
            content_length=0,
            error_message=original_error
        )

# Monkey patch the existing extractor
def patch_main_crawler():
    """Apply PDF detection patch to running crawler"""
    import multi_crawler
    
    # Store reference to original method
    original_extract = multi_crawler.DomainSpecificExtractor.extract_content
    pdf_handler = PDFAwareDomainExtractor()
    
    def patched_extract_content(self, url: str):
        """Patched version that handles PDFs"""
        try:
            result = original_extract(self, url)
            
            # If it failed with non-HTML content, check if it's a PDF
            if (result.status == 'failed' and 
                result.error_message and 
                'Non-HTML content type:' in result.error_message):
                
                # Extract content type from error message
                content_type = result.error_message.replace('Non-HTML content type: ', '').strip()
                
                # Use PDF handler
                return pdf_handler.handle_non_html_content(url, content_type, result.error_message)
            
            return result
            
        except Exception as e:
            # Fallback to original behavior
            return original_extract(self, url)
    
    # Apply the patch
    multi_crawler.DomainSpecificExtractor.extract_content = patched_extract_content
    print("Applied PDF detection patch to main crawler")
    return pdf_handler

if __name__ == "__main__":
    patch_main_crawler()
    print("PDF detection patch ready - import this module in your crawler")