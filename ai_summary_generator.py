#!/usr/bin/env python3
"""
AI Summary Generator for Wilson Lin Chunks
Adds 1-2 line contextual summary headers using Claude API
Following Wilson Lin methodology: "Light summarization header (1–2 lines): 
a localized TL;DR of the section"
"""

import pandas as pd
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import anthropic
from aisitools.api_key import get_api_key_for_proxy
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaudeAPISummarizer:
    """Claude API integration for Wilson Lin summary generation"""
    
    def __init__(self):
        # Setup Claude API using AWS secrets like other files
        self.anthropic_client = None
        self._setup_claude_api()
        
        # Rate limiting
        self._lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 20 requests per second max like existing code
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_summaries': 0,
            'api_errors': 0,
            'rate_limits': 0,
            'retries': 0
        }
    
    def _setup_claude_api(self):
        """Setup Claude API client using AWS secrets like existing files"""
        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            anthropic_base = os.environ.get("ANTHROPIC_BASE_URL")
            
            if not anthropic_key or not anthropic_base:
                raise ValueError("Missing ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL environment variables")
                
            proxy_key = get_api_key_for_proxy(anthropic_key)
            
            self.anthropic_client = anthropic.Anthropic(
                api_key=proxy_key,
                base_url=anthropic_base
            )
            logger.info("✅ Claude API client initialized for Wilson Lin summaries")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Claude API: {e}")
            self.anthropic_client = None
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def generate_wilson_lin_summary(self, chunk_content: str, section_path: str, 
                                   prev_context: str = "", max_retries: int = 3) -> str:
        """Generate Wilson Lin style summary header using Claude API"""
        
        if not self.anthropic_client:
            # Fallback to simple summary
            return f"Content from {section_path.split(' › ')[-1]}"
        
        # Build context-aware prompt
        prompt = self._build_summary_prompt(chunk_content, section_path, prev_context)
        
        for attempt in range(max_retries + 1):
            try:
                self._enforce_rate_limit()
                
                with self._lock:
                    self.stats['total_requests'] += 1
                
                # Make API call using Anthropic client
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-latest",  # Fast model like existing code
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                summary = response.content[0].text.strip()
                
                with self._lock:
                    self.stats['successful_summaries'] += 1
                
                return self._clean_summary(summary)
            
            except Exception as e:
                logger.debug(f"API summary failed on attempt {attempt + 1}: {e}")
                
                with self._lock:
                    self.stats['api_errors'] += 1
                
                if attempt < max_retries:
                    with self._lock:
                        self.stats['retries'] += 1
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return f"Content from {section_path.split(' › ')[-1]}"  # Fallback
        
        # Should not reach here
        return f"Content from {section_path.split(' › ')[-1]}"
    
    def _build_summary_prompt(self, content: str, section_path: str, prev_context: str) -> str:
        """Build Wilson Lin style summary prompt"""
        
        prompt = f"""Create a 1-2 line contextual summary header for this text chunk. This is for Wilson Lin's contextual chunking methodology where each chunk needs a localized TL;DR that serves as both a label and context.

Section Context: {section_path}
{f"Previous Context: {prev_context[:200]}..." if prev_context else ""}

Chunk Content:
{content[:1500]}...

Requirements:
- Exactly 1-2 lines maximum
- Focus on what THIS specific chunk covers
- Use the section context to provide specificity  
- Make it useful for retrieval and understanding
- Don't just repeat the section name

Example format:
"Explains the mesa-optimization threat model where learned optimizers pursue goals different from base objectives. Focuses on inner alignment failures in advanced AI systems."

Summary Header:"""
        
        return prompt
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and validate AI-generated summary"""
        # Remove quotes and extra whitespace
        summary = summary.strip().strip('"\'')
        
        # Ensure it's not too long (Wilson Lin wants 1-2 lines)
        lines = summary.split('\n')
        if len(lines) > 2:
            summary = '\n'.join(lines[:2])
        
        # Limit character length
        if len(summary) > 300:
            summary = summary[:300] + "..."
        
        return summary

class WilsonLinSummaryGenerator:
    """Generate AI summaries for Wilson Lin chunks"""
    
    def __init__(self):
        self.wilson_lin_dir = Path("/home/ubuntu/LW_scrape/wilson_lin_chunks")
        self.output_dir = Path("/home/ubuntu/LW_scrape/wilson_lin_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Claude API
        try:
            self.summarizer = ClaudeAPISummarizer()
            logger.info("✅ Claude API initialized for summary generation")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Claude API: {e}")
            logger.info("💡 Please set ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL environment variables")
            raise
        
        # Threading
        self._lock = threading.Lock()
        self.batch_stats = {
            'chunks_processed': 0,
            'summaries_generated': 0,
            'fallback_summaries': 0,
            'failed_chunks': 0
        }
        
        # Setup logging
        handler = logging.FileHandler(self.output_dir / 'summary_generation.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        logger.info("🤖 Wilson Lin AI Summary Generator initialized")
    
    def load_wilson_lin_chunks(self) -> pd.DataFrame:
        """Load Wilson Lin chunks for summary generation"""
        chunks_file = self.wilson_lin_dir / 'wilson_lin_chunks.parquet'
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Wilson Lin chunks not found: {chunks_file}")
        
        df = pd.read_parquet(chunks_file)
        logger.info(f"📚 Loaded {len(df):,} Wilson Lin chunks for summary generation")
        
        return df
    
    def generate_summary_for_chunk(self, chunk_row: dict) -> dict:
        """Generate AI summary for a single chunk"""
        try:
            chunk_id = chunk_row['chunk_id']
            content = chunk_row['content']
            section_path = chunk_row['section_path']
            prev_context = chunk_row.get('prev_context', '')
            
            # Generate AI summary
            summary_header = self.summarizer.generate_wilson_lin_summary(
                chunk_content=content,
                section_path=section_path,
                prev_context=prev_context
            )
            
            # Update chunk with summary
            enhanced_chunk = dict(chunk_row)
            enhanced_chunk['summary_header'] = summary_header
            
            # Check if it's a fallback summary
            is_fallback = summary_header.startswith("Summary: ")
            
            with self._lock:
                self.batch_stats['chunks_processed'] += 1
                if is_fallback:
                    self.batch_stats['fallback_summaries'] += 1
                else:
                    self.batch_stats['summaries_generated'] += 1
            
            return enhanced_chunk
            
        except Exception as e:
            logger.error(f"Error generating summary for chunk {chunk_row.get('chunk_id', 'unknown')}: {e}")
            
            # Return chunk with fallback summary
            enhanced_chunk = dict(chunk_row)
            enhanced_chunk['summary_header'] = f"Content from {chunk_row.get('section_path', 'document').split(' › ')[-1]}"
            
            with self._lock:
                self.batch_stats['failed_chunks'] += 1
            
            return enhanced_chunk
    
    def generate_all_summaries(self, max_workers: int = 3, batch_size: int = 100):
        """Generate AI summaries for all Wilson Lin chunks"""
        logger.info("🤖 Starting AI summary generation for Wilson Lin chunks...")
        
        # Load chunks
        df = self.load_wilson_lin_chunks()
        chunks = df.to_dict('records')
        
        logger.info(f"📝 Generating summaries for {len(chunks):,} chunks...")
        logger.info(f"⚡ Using {max_workers} workers with batch size {batch_size}")
        
        # Process in batches to manage API rate limits
        enhanced_chunks = []
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            logger.info(f"📦 Processing batch {batch_start//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} "
                       f"({len(batch_chunks)} chunks)")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch tasks
                future_to_chunk = {executor.submit(self.generate_summary_for_chunk, chunk): chunk 
                                 for chunk in batch_chunks}
                
                # Collect results
                batch_results = []
                for future in as_completed(future_to_chunk):
                    result = future.result()
                    batch_results.append(result)
                
                enhanced_chunks.extend(batch_results)
            
            # Log batch progress
            progress = (batch_end / len(chunks)) * 100
            logger.info(f"📈 Progress: {progress:.1f}% - Generated: {self.batch_stats['summaries_generated']:,}, "
                       f"Fallbacks: {self.batch_stats['fallback_summaries']:,}")
            
            # Brief pause between batches to respect rate limits
            if batch_end < len(chunks):
                time.sleep(2)
        
        # Save enhanced chunks
        if enhanced_chunks:
            logger.info(f"💾 Saving {len(enhanced_chunks):,} enhanced Wilson Lin chunks...")
            self._save_enhanced_chunks(enhanced_chunks)
        
        # Log final statistics
        self._log_final_stats(len(enhanced_chunks))
        
        return len(enhanced_chunks)
    
    def _save_enhanced_chunks(self, chunks: List[dict]):
        """Save Wilson Lin chunks with AI summaries"""
        df = pd.DataFrame(chunks)
        
        # Ensure columns are in correct order
        column_order = [
            'chunk_id', 'doc_id', 'content', 'prev_context', 'section_path', 
            'summary_header', 'token_count', 'char_count', 'chunk_index',
            'page_start', 'page_end', 'title', 'url', 'domain', 
            'content_type', 'language', 'structure_type'
        ]
        
        # Reorder columns
        df = df[column_order]
        
        # Save to parquet
        output_file = self.output_dir / 'wilson_lin_enhanced_chunks.parquet'
        df.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Saved enhanced Wilson Lin chunks to {output_file}")
        
        return output_file
    
    def _log_final_stats(self, chunk_count: int):
        """Log comprehensive final statistics"""
        api_stats = self.summarizer.stats
        
        logger.info("\n" + "="*60)
        logger.info("🤖 WILSON LIN AI SUMMARY GENERATION COMPLETE!")
        logger.info("="*60)
        logger.info(f"📊 SUMMARY STATISTICS:")
        logger.info(f"  Chunks processed: {self.batch_stats['chunks_processed']:,}")
        logger.info(f"  AI summaries generated: {self.batch_stats['summaries_generated']:,}")
        logger.info(f"  Fallback summaries: {self.batch_stats['fallback_summaries']:,}")
        logger.info(f"  Failed chunks: {self.batch_stats['failed_chunks']:,}")
        
        logger.info(f"\n🔌 API STATISTICS:")
        logger.info(f"  Total API requests: {api_stats['total_requests']:,}")
        logger.info(f"  Successful requests: {api_stats['successful_summaries']:,}")
        logger.info(f"  API errors: {api_stats['api_errors']:,}")
        logger.info(f"  Rate limits: {api_stats['rate_limits']:,}")
        logger.info(f"  Retries: {api_stats['retries']:,}")
        
        success_rate = (self.batch_stats['summaries_generated'] / max(chunk_count, 1)) * 100
        api_success_rate = (api_stats['successful_summaries'] / max(api_stats['total_requests'], 1)) * 100
        
        logger.info(f"\n📈 SUCCESS RATES:")
        logger.info(f"  Overall success rate: {success_rate:.1f}%")
        logger.info(f"  API success rate: {api_success_rate:.1f}%")
        logger.info("="*60)
        
        # Save comprehensive stats
        all_stats = {
            **self.batch_stats,
            **{f"api_{k}": v for k, v in api_stats.items()},
            'summary_generation_complete': datetime.now().isoformat(),
            'success_rate': success_rate,
            'api_success_rate': api_success_rate,
            'total_enhanced_chunks': chunk_count
        }
        
        with open(self.output_dir / 'summary_generation_stats.json', 'w') as f:
            json.dump(all_stats, f, indent=2)

def main():
    # Check for required environment variables
    import os
    if 'ANTHROPIC_API_KEY' not in os.environ or 'ANTHROPIC_BASE_URL' not in os.environ:
        print("❌ Missing required environment variables")
        print("💡 Please set:")
        print("   export ANTHROPIC_API_KEY='aws-secretsmanager://your-secret-arn'")
        print("   export ANTHROPIC_BASE_URL='your-base-url'")
        return
    
    generator = WilsonLinSummaryGenerator()
    
    # Increased workers to speed up processing
    max_workers = 6  # Higher concurrency while respecting rate limits
    batch_size = 50  # Small batches
    
    logger.info(f"🤖 Starting Wilson Lin AI summary generation...")
    logger.info(f"⚡ Settings: {max_workers} workers, batch size {batch_size}")
    
    enhanced_count = generator.generate_all_summaries(
        max_workers=max_workers, 
        batch_size=batch_size
    )
    
    logger.info(f"🎯 Wilson Lin enhancement complete: {enhanced_count:,} chunks with AI summaries")

if __name__ == '__main__':
    main()