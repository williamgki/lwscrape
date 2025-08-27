#!/usr/bin/env python3
"""
Parallel PDF Recovery - 4 workers processing remaining 1,853 PDFs
"""

import multiprocessing as mp
import json
import time
import queue
from pathlib import Path
from pdf_crawler import PDFExtractor

def worker_process(worker_id: int, pdf_queue: mp.Queue, results_queue: mp.Queue):
    """PDF processing worker"""
    extractor = PDFExtractor()
    processed = 0
    successful = 0
    
    while True:
        try:
            pdf_data = pdf_queue.get(timeout=5)
            if pdf_data is None:  # Shutdown signal
                break
                
            url, domain, frequency = pdf_data
            result = extractor.extract_pdf_text(url)
            result['worker_id'] = worker_id
            result['frequency'] = frequency
            result['domain'] = domain
            
            results_queue.put(result)
            
            processed += 1
            if result['status'] == 'success':
                successful += 1
                
            if processed % 20 == 0:
                print(f"Worker {worker_id}: {processed} processed, {successful} successful")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            
    print(f"Worker {worker_id} finished: {processed} processed, {successful} successful")

def main():
    """Main parallel recovery function"""
    # Load remaining PDFs (skip first 36 already processed)
    remaining_pdfs = []
    with open('missed_pdfs_comprehensive.txt', 'r') as f:
        lines = f.readlines()[36:]  # Skip priority batch
        
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            url, domain, freq = parts[0], parts[1], int(parts[2])
            remaining_pdfs.append((url, domain, freq))
    
    print(f"Starting parallel recovery of {len(remaining_pdfs)} PDFs with 4 additional workers (5-8)")
    
    # Create queues
    pdf_queue = mp.Queue()
    results_queue = mp.Queue()
    
    # Load work queue
    for pdf_data in remaining_pdfs:
        pdf_queue.put(pdf_data)
    
    # Add shutdown signals
    for _ in range(4):
        pdf_queue.put(None)
    
    # Start workers
    workers = []
    for worker_id in range(5, 9):
        worker = mp.Process(target=worker_process, args=(worker_id, pdf_queue, results_queue))
        worker.start()
        workers.append(worker)
    
    # Collect results
    output_dir = Path('data/pdfs')
    output_file = output_dir / 'parallel_recovery_boost.jsonl'
    
    processed = 0
    successful = 0
    start_time = time.time()
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        while processed < len(remaining_pdfs):
            try:
                result = results_queue.get(timeout=10)
                outf.write(json.dumps(result) + '\n')
                outf.flush()
                
                processed += 1
                if result['status'] == 'success':
                    successful += 1
                    
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed * 3600  # per hour
                    remaining = len(remaining_pdfs) - processed
                    eta = remaining / (rate / 3600) if rate > 0 else 0
                    
                    print(f"Progress: {processed}/{len(remaining_pdfs)} "
                          f"({successful/processed*100:.1f}% success) "
                          f"Rate: {rate:.0f}/hour ETA: {eta:.1f}h")
                    
            except queue.Empty:
                # Check if workers are still alive
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    break
    
    # Wait for workers
    for worker in workers:
        worker.join(timeout=5)
    
    # Final stats
    elapsed = time.time() - start_time
    success_rate = successful / processed * 100 if processed > 0 else 0
    
    print(f"Parallel recovery completed:")
    print(f"  Processed: {processed}/{len(remaining_pdfs)}")
    print(f"  Successful: {successful} ({success_rate:.1f}%)")
    print(f"  Time: {elapsed/3600:.1f} hours")
    print(f"  Rate: {processed/elapsed*3600:.0f} PDFs/hour")

if __name__ == '__main__':
    main()