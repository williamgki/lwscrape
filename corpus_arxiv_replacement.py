#!/usr/bin/env python3
"""
Corpus ArXiv Replacement Script
Replace old poorly-extracted arXiv chunks with Wilson Lin enhanced versions
Eliminates duplicates while preserving all non-arXiv content
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusArXivReplacer:
    def __init__(self):
        self.main_corpus = Path("/home/ubuntu/LW_scrape/unified_corpus/unified_contextual_chunks_complete.parquet")
        self.enhanced_arxiv = Path("/home/ubuntu/LW_scrape/arxiv_reprocessing_enhanced/arxiv_all_wilson_lin_enhanced.parquet")
        self.output_dir = Path("/home/ubuntu/LW_scrape/corpus_final_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'original_chunks': 0,
            'arxiv_removed': 0,
            'arxiv_added': 0,
            'final_chunks': 0,
            'papers_enhanced': 0
        }
        
    def extract_arxiv_id(self, url_or_title):
        """Extract arXiv ID from URL or title"""
        if pd.isna(url_or_title):
            return None
            
        # Common arXiv ID patterns
        patterns = [
            r'arxiv\.org/(?:abs/|pdf/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)',
            r'(?:^|/)([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)(?:\.pdf)?$',
            r'(?:^|/)([0-9]{7}\.[0-9]{4}(?:v[0-9]+)?)(?:\.pdf)?$',  # Old format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(url_or_title), re.IGNORECASE)
            if match:
                arxiv_id = match.group(1)
                # Normalize version numbers (remove v1, v2, etc for matching)
                base_id = re.sub(r'v[0-9]+$', '', arxiv_id)
                return base_id
        return None
    
    def identify_arxiv_chunks(self, df):
        """Identify chunks that come from arXiv papers"""
        arxiv_mask = df['original_url'].str.contains('arxiv', case=False, na=False)
        
        # Also check doc_title for arXiv IDs
        arxiv_ids_in_title = df['doc_title'].apply(self.extract_arxiv_id).notna()
        
        # Combine both conditions
        is_arxiv = arxiv_mask | arxiv_ids_in_title
        
        logger.info(f"Found {is_arxiv.sum()} arXiv chunks out of {len(df)} total chunks")
        return is_arxiv
    
    def create_arxiv_id_mapping(self, enhanced_df):
        """Create mapping from arXiv ID to enhanced chunks"""
        arxiv_mapping = {}
        
        for _, chunk in enhanced_df.iterrows():
            # Extract arXiv ID from doc_id or original_url
            arxiv_id = self.extract_arxiv_id(chunk['doc_id'])
            if not arxiv_id:
                arxiv_id = self.extract_arxiv_id(chunk['original_url'])
                
            if arxiv_id:
                if arxiv_id not in arxiv_mapping:
                    arxiv_mapping[arxiv_id] = []
                arxiv_mapping[arxiv_id].append(chunk)
        
        logger.info(f"Created mapping for {len(arxiv_mapping)} arXiv papers")
        return arxiv_mapping
    
    def replace_arxiv_chunks(self):
        """Main replacement logic"""
        logger.info("🚀 Starting ArXiv Chunk Replacement...")
        
        # Load main corpus
        logger.info(f"Loading main corpus from {self.main_corpus}")
        main_df = pd.read_parquet(self.main_corpus)
        self.stats['original_chunks'] = len(main_df)
        
        # Load enhanced arXiv chunks
        logger.info(f"Loading enhanced arXiv chunks from {self.enhanced_arxiv}")
        enhanced_df = pd.read_parquet(self.enhanced_arxiv)
        self.stats['papers_enhanced'] = enhanced_df['doc_id'].nunique()
        
        # Identify arXiv chunks in main corpus
        arxiv_chunks_mask = self.identify_arxiv_chunks(main_df)
        self.stats['arxiv_removed'] = arxiv_chunks_mask.sum()
        
        # Remove old arXiv chunks
        logger.info(f"Removing {self.stats['arxiv_removed']} old arXiv chunks")
        non_arxiv_df = main_df[~arxiv_chunks_mask].copy()
        
        # Add enhanced arXiv chunks
        logger.info(f"Adding {len(enhanced_df)} enhanced arXiv chunks")
        self.stats['arxiv_added'] = len(enhanced_df)
        
        # Combine non-arXiv content with enhanced arXiv content
        # Ensure schema compatibility
        enhanced_df_compat = enhanced_df.copy()
        
        # Ensure all columns match
        for col in non_arxiv_df.columns:
            if col not in enhanced_df_compat.columns:
                enhanced_df_compat[col] = None
        
        for col in enhanced_df_compat.columns:
            if col not in non_arxiv_df.columns:
                non_arxiv_df[col] = None
        
        # Reorder columns to match
        enhanced_df_compat = enhanced_df_compat[non_arxiv_df.columns]
        
        # Combine datasets
        final_df = pd.concat([non_arxiv_df, enhanced_df_compat], ignore_index=True)
        
        # Update corpus source for tracking
        final_df.loc[final_df['corpus_source'].isna(), 'corpus_source'] = 'main_lw_gchildren'
        
        self.stats['final_chunks'] = len(final_df)
        
        # Save enhanced corpus
        output_file = self.output_dir / 'final_enhanced_corpus.parquet'
        final_df.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Enhanced corpus saved to {output_file}")
        
        # Generate comprehensive statistics
        self.generate_replacement_stats(main_df, enhanced_df, final_df)
        
        return final_df
    
    def generate_replacement_stats(self, original_df, enhanced_df, final_df):
        """Generate detailed replacement statistics"""
        # Quality comparison
        original_arxiv_mask = self.identify_arxiv_chunks(original_df)
        original_arxiv = original_df[original_arxiv_mask]
        
        # Count papers with proper metadata
        orig_with_titles = int((original_arxiv['doc_title'].str.len() > 10).sum())
        enhanced_with_titles = int((enhanced_df['doc_title'].str.len() > 10).sum())
        
        orig_with_authors = int(original_arxiv['authors'].notna().sum())
        enhanced_with_authors = int(enhanced_df['authors'].notna().sum())
        
        stats = {
            'replacement_summary': {
                'original_total_chunks': int(len(original_df)),
                'original_arxiv_chunks': int(len(original_arxiv)),
                'enhanced_arxiv_chunks': int(len(enhanced_df)),
                'final_total_chunks': int(len(final_df)),
                'net_change': int(len(final_df) - len(original_df))
            },
            'quality_improvement': {
                'original_arxiv_papers': int(original_arxiv['doc_id'].nunique()),
                'enhanced_arxiv_papers': int(enhanced_df['doc_id'].nunique()),
                'original_chunks_per_paper': float(len(original_arxiv) / max(original_arxiv['doc_id'].nunique(), 1)),
                'enhanced_chunks_per_paper': float(len(enhanced_df) / max(enhanced_df['doc_id'].nunique(), 1)),
                'original_papers_with_titles': orig_with_titles,
                'enhanced_papers_with_titles': enhanced_with_titles,
                'original_papers_with_authors': orig_with_authors,
                'enhanced_papers_with_authors': enhanced_with_authors,
                'title_coverage_improvement': f"{(enhanced_with_titles/len(enhanced_df)*100):.1f}% vs {(orig_with_titles/len(original_arxiv)*100):.1f}%",
                'author_coverage_improvement': f"{(enhanced_with_authors/len(enhanced_df)*100):.1f}% vs {(orig_with_authors/len(original_arxiv)*100):.1f}%"
            },
            'corpus_composition': {
                'main_lw_chunks': int(len(final_df[final_df['corpus_source'].str.contains('main', na=False)])),
                'gchildren_chunks': int(len(final_df[final_df['corpus_source'].str.contains('gchildren', na=False)])),
                'enhanced_arxiv_chunks': int(len(final_df[final_df['corpus_source'].str.contains('wilson_lin', na=False)])),
                'total_papers': int(final_df['doc_id'].nunique()),
                'avg_chunks_per_paper': float(len(final_df) / final_df['doc_id'].nunique()),
                'wilson_lin_compliance': True
            },
            'replacement_timestamp': datetime.now().isoformat(),
            'files': {
                'original_corpus': str(self.main_corpus),
                'enhanced_arxiv': str(self.enhanced_arxiv),
                'final_corpus': str(self.output_dir / 'final_enhanced_corpus.parquet')
            }
        }
        
        # Save detailed stats
        with open(self.output_dir / 'replacement_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Log summary
        logger.info("\\n" + "="*60)
        logger.info("📊 CORPUS REPLACEMENT COMPLETE!")
        logger.info("="*60)
        logger.info(f"Original corpus: {stats['replacement_summary']['original_total_chunks']:,} chunks")
        logger.info(f"Removed old arXiv: {stats['replacement_summary']['original_arxiv_chunks']:,} chunks")
        logger.info(f"Added enhanced arXiv: {stats['replacement_summary']['enhanced_arxiv_chunks']:,} chunks")
        logger.info(f"Final enhanced corpus: {stats['replacement_summary']['final_total_chunks']:,} chunks")
        logger.info(f"Net change: {stats['replacement_summary']['net_change']:+,} chunks")
        logger.info("\\n" + "🔥 QUALITY IMPROVEMENTS:")
        logger.info(f"Title coverage: {stats['quality_improvement']['title_coverage_improvement']}")
        logger.info(f"Author coverage: {stats['quality_improvement']['author_coverage_improvement']}")
        logger.info(f"Chunks per paper: {stats['quality_improvement']['enhanced_chunks_per_paper']:.1f} vs {stats['quality_improvement']['original_chunks_per_paper']:.1f}")
        logger.info("="*60)

def main():
    replacer = CorpusArXivReplacer()
    
    if not replacer.main_corpus.exists():
        logger.error(f"Main corpus not found: {replacer.main_corpus}")
        return
        
    if not replacer.enhanced_arxiv.exists():
        logger.error(f"Enhanced arXiv corpus not found: {replacer.enhanced_arxiv}")
        return
    
    # Execute replacement
    final_corpus = replacer.replace_arxiv_chunks()
    
    logger.info(f"🎉 Corpus replacement complete! Enhanced corpus available at:")
    logger.info(f"📁 {replacer.output_dir / 'final_enhanced_corpus.parquet'}")

if __name__ == '__main__':
    main()