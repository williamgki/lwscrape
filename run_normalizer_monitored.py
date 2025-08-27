#!/usr/bin/env python3
"""
Monitored corpus normalizer with progress tracking
"""

import sys
import time
from pathlib import Path
sys.path.append('/home/ubuntu/LW_scrape')

from corpus_normalizer import CorpusNormalizer

def run_with_monitoring():
    """Run normalizer with progress monitoring"""
    input_dir = Path("/home/ubuntu/LW_scrape/data")
    output_dir = Path("/home/ubuntu/LW_scrape/normalized_corpus")
    
    print("Starting monitored corpus normalization...")
    start_time = time.time()
    
    try:
        normalizer = CorpusNormalizer(input_dir, output_dir)
        result = normalizer.process_corpus()
        
        elapsed = time.time() - start_time
        print(f"✅ Normalization complete!")
        print(f"📊 Processed {len(result)} unique documents")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed/60:.1f} minutes: {e}")
        
        # Try to save partial progress
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"🔍 Partial output files: {len(files)}")

if __name__ == '__main__':
    run_with_monitoring()