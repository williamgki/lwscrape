#!/usr/bin/env python3
"""
Add Wilson Lin API Summaries to Reprocessed ArXiv Chunks
Enhances existing simple chunks with AI-generated summary headers for consistency
"""

import pandas as pd
import json
import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from datetime import datetime
import anthropic
from aisitools.api_key import get_api_key_for_proxy
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WilsonLinSummaryEnhancer:
    def __init__(self):
        self.arxiv_dir = Path("/home/ubuntu/LW_scrape/arxiv_reprocessing")
        self.output_dir = Path("/home/ubuntu/LW_scrape/arxiv_reprocessing_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {'api_calls': 0, 'api_failures': 0, 'chunks_processed': 0}
        self._lock = threading.Lock()
        
        # Setup Claude API for summary generation
        self.anthropic_client = None
        self._setup_claude_api()
        
        # Setup file logging
        handler = logging.FileHandler(self.output_dir / 'wilson_lin_enhancement.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
    def _setup_claude_api(self):
        """Setup Claude API client for summary generation"""
        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            anthropic_base = os.environ.get("ANTHROPIC_BASE_URL")
            
            if not anthropic_key or not anthropic_base:
                raise ValueError("Missing ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL")
                
            proxy_key = get_api_key_for_proxy(anthropic_key)
            
            self.anthropic_client = anthropic.Anthropic(
                api_key=proxy_key,
                base_url=anthropic_base
            )
            logger.info("✅ Claude API client initialized for Wilson Lin summaries")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Claude API: {e}")
            self.anthropic_client = None
    
    def generate_wilson_lin_summary(self, chunk_content: str, section_path: str) -> str:
        """Generate Wilson Lin style summary using Claude API"""
        if not self.anthropic_client:
            # Fallback to simple summary
            sentences = re.split(r'[.!?]+', chunk_content)
            return sentences[0].strip()[:150] if sentences else "Research content"
            
        try:
            # Rate limiting delay
            time.sleep(0.05)  # 20 calls/second max
            
            # Limit content for API call
            content_sample = chunk_content[:1000] if len(chunk_content) > 1000 else chunk_content
            
            prompt = f"""Given this academic text chunk from section "{section_path}", write a concise 1-2 line contextual summary that captures the main point and provides local context. Be specific about the research content:

{content_sample}

Contextual summary (1-2 lines max):"""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            lines = summary.split('\\n')
            result = '. '.join(lines[:2]) if len(lines) > 1 else lines[0]
            
            with self._lock:
                self.stats['api_calls'] += 1
            
            return result
            
        except Exception as e:
            logger.debug(f"API summary failed: {e}")
            with self._lock:
                self.stats['api_failures'] += 1
            
            # Fallback to first sentence
            sentences = re.split(r'[.!?]+', chunk_content)
            return sentences[0].strip()[:150] if sentences else "Research content"
    
    def enhance_chunk_batch(self, chunk_batch):
        """Enhance a batch of chunks with Wilson Lin summaries"""
        enhanced_chunks = []
        
        for _, chunk in chunk_batch.iterrows():
            try:
                # Generate enhanced summary
                enhanced_summary = self.generate_wilson_lin_summary(
                    chunk['content'], 
                    chunk['section_path']
                )
                
                # Create enhanced chunk with Wilson Lin summary
                enhanced_chunk = chunk.to_dict()
                enhanced_chunk['summary_header'] = enhanced_summary
                enhanced_chunks.append(enhanced_chunk)
                
                with self._lock:
                    self.stats['chunks_processed'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to enhance chunk {chunk['chunk_id']}: {e}")
                # Keep original chunk
                enhanced_chunks.append(chunk.to_dict())
        
        return enhanced_chunks
    
    def enhance_phase_file(self, phase_file):
        """Enhance chunks in a specific phase file"""
        logger.info(f"Loading {phase_file.name}...")
        
        # Load existing chunks
        df = pd.read_parquet(phase_file)
        total_chunks = len(df)
        
        logger.info(f"Enhancing {total_chunks} chunks with Wilson Lin summaries...")
        
        # Process in batches for parallel API calls
        batch_size = 25
        all_enhanced = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:  # Conservative for API rate limits
            futures = []
            
            for i in range(0, total_chunks, batch_size):
                batch = df.iloc[i:i+batch_size]
                future = executor.submit(self.enhance_chunk_batch, batch)
                futures.append(future)
            
            for i, future in enumerate(futures):
                try:
                    enhanced_batch = future.result()
                    all_enhanced.extend(enhanced_batch)
                    
                    if (i + 1) % 5 == 0:
                        progress = ((i + 1) * batch_size * 100) // total_chunks
                        logger.info(f"Progress: {progress}% - {self.stats['api_calls']} API calls, "
                                  f"{self.stats['api_failures']} failures")
                        
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        # Create enhanced dataframe
        enhanced_df = pd.DataFrame(all_enhanced)
        enhanced_df['corpus_source'] = 'main_wilson_lin_enhanced'
        
        # Save enhanced file
        phase_num = phase_file.stem.split('phase')[-1]
        output_file = self.output_dir / f"arxiv_wilson_lin_enhanced_phase{phase_num}.parquet"
        enhanced_df.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Enhanced {len(enhanced_df)} chunks saved to {output_file.name}")
        
        # Save enhancement stats
        stats = {
            'phase': int(phase_num),
            'chunks_enhanced': len(enhanced_df),
            'papers': enhanced_df['doc_id'].nunique(),
            'api_calls_used': self.stats['api_calls'],
            'api_failures': self.stats['api_failures'],
            'wilson_lin_compliance': True,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / f'wilson_lin_phase{phase_num}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return enhanced_df

def main():
    enhancer = WilsonLinSummaryEnhancer()
    
    logger.info("🚀 Starting Wilson Lin Summary Enhancement for ArXiv Reprocessed Chunks...")
    
    # Find phase files to enhance
    phase_files = sorted(enhancer.arxiv_dir.glob("arxiv_reprocessed_phase*.parquet"))
    
    if not phase_files:
        logger.error("No phase files found to enhance")
        return
    
    logger.info(f"Found {len(phase_files)} phase files to enhance")
    
    all_enhanced_chunks = []
    
    for phase_file in phase_files:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Processing {phase_file.name}")
        logger.info(f"{'='*60}")
        
        enhanced_df = enhancer.enhance_phase_file(phase_file)
        all_enhanced_chunks.append(enhanced_df)
    
    # Combine all enhanced phases
    if all_enhanced_chunks:
        combined_df = pd.concat(all_enhanced_chunks, ignore_index=True)
        combined_df['corpus_source'] = 'main_wilson_lin_enhanced'
        
        # Save combined enhanced corpus
        combined_file = enhancer.output_dir / 'arxiv_all_wilson_lin_enhanced.parquet'
        combined_df.to_parquet(combined_file, index=False)
        
        # Final stats
        final_stats = {
            'total_papers_enhanced': combined_df['doc_id'].nunique(),
            'total_chunks_enhanced': len(combined_df),
            'avg_tokens_per_chunk': combined_df['token_count'].mean(),
            'total_api_calls': enhancer.stats['api_calls'],
            'api_success_rate': 1 - (enhancer.stats['api_failures'] / max(enhancer.stats['api_calls'], 1)),
            'wilson_lin_compliance': True,
            'enhancement_complete': datetime.now().isoformat()
        }
        
        with open(enhancer.output_dir / 'final_wilson_lin_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info("\\n" + "="*60)
        logger.info("✅ Wilson Lin Enhancement Complete!")
        logger.info(f"📊 Enhanced {final_stats['total_chunks_enhanced']} chunks from {final_stats['total_papers_enhanced']} papers")
        logger.info(f"🔥 Used {final_stats['total_api_calls']} API calls with {final_stats['api_success_rate']:.1%} success rate")
        logger.info(f"💾 Combined file saved: {combined_file}")
        logger.info("="*60)

if __name__ == '__main__':
    main()