#!/usr/bin/env python3
"""
Live PDF Processor - continuously monitors PDF queue and processes new PDFs
Works alongside the main crawler to handle PDFs as they're discovered
"""

import time
import json
import os
import logging
from pathlib import Path
from pdf_crawler import PDFExtractor
from pdf_queue_utils import PDFQueueManager

class LivePDFProcessor:
    """Continuously processes PDFs from the queue"""
    
    def __init__(self, db_path: str = "data/crawl_urls.db", output_dir: str = "data/pdfs", 
                 check_interval: int = 60):
        self.pdf_queue = PDFQueueManager(db_path)
        self.extractor = PDFExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.check_interval = check_interval
        
        # Output file for live results
        self.output_file = self.output_dir / 'live_pdfs.jsonl'
        
        # Stats
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'queue_empty_checks': 0
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_pdf_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, batch_size: int = 10) -> int:
        """Process a batch of pending PDFs"""
        pending_pdfs = self.pdf_queue.get_pending_pdfs(batch_size)
        
        if not pending_pdfs:
            self.stats['queue_empty_checks'] += 1
            return 0
        
        processed_count = 0
        
        with open(self.output_file, 'a', encoding='utf-8') as outf:
            for url, domain, frequency in pending_pdfs:
                self.logger.info(f"Processing PDF: {url} (freq: {frequency})")
                
                try:
                    # Extract PDF content
                    result = self.extractor.extract_pdf_text(url)
                    
                    # Update stats
                    if result['status'] == 'success':
                        self.stats['successful'] += 1
                        pages = result['metadata']['pages']
                        size_kb = result['metadata']['file_size'] / 1024
                        self.logger.info(f"  ✓ Success: {pages} pages, {size_kb:.1f}KB")
                        
                        # Mark as successful
                        self.pdf_queue.mark_pdf_processed(
                            url, True, 
                            file_path="live_pdfs.jsonl",
                            content_length=len(result.get('content', ''))
                        )
                    else:
                        self.stats['failed'] += 1
                        self.logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                        
                        # Mark as failed
                        self.pdf_queue.mark_pdf_processed(
                            url, False, 
                            error_message=result.get('error', 'Unknown error')
                        )
                    
                    # Save result to JSONL
                    outf.write(json.dumps(result) + '\n')
                    outf.flush()
                    
                    self.stats['processed'] += 1
                    processed_count += 1
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing PDF {url}: {e}")
                    self.pdf_queue.mark_pdf_processed(url, False, error_message=str(e))
                    self.stats['failed'] += 1
                    processed_count += 1
        
        return processed_count
    
    def run(self):
        """Main processing loop"""
        self.logger.info("Starting Live PDF Processor")
        
        try:
            while True:
                # Get queue stats
                queue_stats = self.pdf_queue.get_queue_stats()
                pending = queue_stats.get('pending', 0)
                
                if pending > 0:
                    self.logger.info(f"Processing {pending} pending PDFs...")
                    processed = self.process_batch()
                    
                    if processed > 0:
                        self.logger.info(f"Processed {processed} PDFs. "
                                       f"Total: {self.stats['processed']} "
                                       f"(Success: {self.stats['successful']}, "
                                       f"Failed: {self.stats['failed']})")
                else:
                    # No pending PDFs - wait longer
                    if self.stats['queue_empty_checks'] % 10 == 0:
                        self.logger.info("PDF queue empty, waiting for new PDFs...")
                    time.sleep(self.check_interval)
                    continue
                
                # Brief pause between batches
                time.sleep(10)
                
        except KeyboardInterrupt:
            self.logger.info("Live PDF Processor interrupted by user")
        except Exception as e:
            self.logger.error(f"Live PDF Processor error: {e}")
        finally:
            final_stats = self.pdf_queue.get_queue_stats()
            self.logger.info(f"Final stats - Processed: {self.stats['processed']}, "
                           f"Queue: {final_stats}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live PDF Processor')
    parser.add_argument('--db', default='data/crawl_urls.db', help='Database path')
    parser.add_argument('--output-dir', default='data/pdfs', help='Output directory')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Check interval in seconds when queue is empty')
    
    args = parser.parse_args()
    
    processor = LivePDFProcessor(
        db_path=args.db,
        output_dir=args.output_dir,
        check_interval=args.interval
    )
    
    processor.run()

if __name__ == '__main__':
    main()