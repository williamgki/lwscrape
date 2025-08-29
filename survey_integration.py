#!/usr/bin/env python3
"""
Survey Corpus Integration
Combines survey contextual chunks with existing enhanced corpus
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SurveyCorpusIntegrator:
    """Integrate survey chunks with existing enhanced corpus"""
    
    def __init__(self):
        self.enhanced_corpus_path = Path("/home/ubuntu/LW_scrape/corpus_final_enhanced/final_enhanced_corpus.parquet")
        self.survey_chunks_path = Path("/home/ubuntu/LW_scrape/survey_chunked/survey_contextual_chunks.parquet")
        self.output_dir = Path("/home/ubuntu/LW_scrape/corpus_with_survey")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🔗 Survey Corpus Integrator initialized")
    
    def load_existing_corpus(self) -> pd.DataFrame:
        """Load existing enhanced corpus"""
        logger.info("📚 Loading existing enhanced corpus...")
        df = pd.read_parquet(self.enhanced_corpus_path)
        logger.info(f"  Loaded {len(df):,} existing chunks")
        return df
    
    def load_survey_chunks(self) -> pd.DataFrame:
        """Load survey contextual chunks"""
        logger.info("🗂️  Loading survey contextual chunks...")
        df = pd.read_parquet(self.survey_chunks_path)
        logger.info(f"  Loaded {len(df):,} survey chunks")
        return df
    
    def harmonize_schemas(self, existing_df: pd.DataFrame, survey_df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize schemas between existing and survey chunks"""
        logger.info("⚙️  Harmonizing chunk schemas...")
        
        # Get target schema from existing corpus
        target_columns = existing_df.columns.tolist()
        logger.info(f"  Target schema has {len(target_columns)} columns")
        
        # Map survey columns to target schema
        survey_harmonized = pd.DataFrame()
        
        # Core mapping
        survey_harmonized['chunk_id'] = survey_df['chunk_id']
        survey_harmonized['doc_id'] = survey_df['doc_id']
        survey_harmonized['content'] = survey_df['content']
        
        # Enhanced corpus specific mapping
        if 'doc_title' in target_columns:
            survey_harmonized['doc_title'] = survey_df['title']
        
        if 'authors' in target_columns:
            survey_harmonized['authors'] = ''  # Survey chunks don't have authors consistently
        
        if 'original_url' in target_columns:
            survey_harmonized['original_url'] = survey_df['url']
        
        # Content metadata
        if 'content_type' in target_columns:
            survey_harmonized['content_type'] = survey_df['content_type']
        
        if 'domain' in target_columns:
            survey_harmonized['domain'] = survey_df['domain']
            
        # Chunking metadata
        if 'chunk_index' in target_columns:
            survey_harmonized['chunk_index'] = survey_df['chunk_index']
        
        if 'token_count' in target_columns:
            survey_harmonized['token_count'] = survey_df['token_count']
        
        # Wilson Lin specific fields
        if 'section_path' in target_columns:
            survey_harmonized['section_path'] = survey_df['section_path']
        
        if 'prev_context' in target_columns:
            survey_harmonized['prev_context'] = survey_df['previous_context']
        
        # Enhanced corpus specific fields
        if 'summary_header' in target_columns:
            survey_harmonized['summary_header'] = ''  # Survey chunks don't have AI summaries yet
        
        if 'page_refs' in target_columns:
            survey_harmonized['page_refs'] = ''
        
        if 'structure_type' in target_columns:
            survey_harmonized['structure_type'] = survey_df['chunk_type']
        
        if 'heading_level' in target_columns:
            survey_harmonized['heading_level'] = 0
        
        if 'content_hash' in target_columns:
            # Generate content hash for survey chunks
            survey_harmonized['content_hash'] = survey_df['content'].apply(
                lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()[:16] if x else ''
            )
        
        # Publication dates
        if 'pub_date' in target_columns:
            survey_harmonized['pub_date'] = None
        
        # Fill any missing columns with appropriate defaults
        for col in target_columns:
            if col not in survey_harmonized.columns:
                if col in ['ref_frequency', 'duplicate_count']:
                    survey_harmonized[col] = 1
                elif col in ['domain_reputation']:
                    survey_harmonized[col] = 0.7  # Default reputation for survey content
                elif col in ['minhash_signature']:
                    survey_harmonized[col] = b''  # Empty bytes for minhash
                else:
                    survey_harmonized[col] = ''  # String default
        
        # Reorder columns to match target schema
        survey_harmonized = survey_harmonized[target_columns]
        
        logger.info(f"  ✅ Harmonized {len(survey_harmonized):,} survey chunks to target schema")
        
        return survey_harmonized
    
    def add_corpus_source_tags(self, existing_df: pd.DataFrame, survey_df: pd.DataFrame) -> tuple:
        """Add corpus source tags to differentiate content"""
        logger.info("🏷️  Adding corpus source tags...")
        
        # Add source column to existing corpus
        existing_df = existing_df.copy()
        existing_df['corpus_source'] = 'enhanced_lw'
        
        # Add source column to survey corpus  
        survey_df = survey_df.copy()
        survey_df['corpus_source'] = 'survey'
        
        logger.info(f"  Tagged {len(existing_df):,} enhanced LW chunks")
        logger.info(f"  Tagged {len(survey_df):,} survey chunks")
        
        return existing_df, survey_df
    
    def combine_corpora(self, existing_df: pd.DataFrame, survey_df: pd.DataFrame) -> pd.DataFrame:
        """Combine the two corpora"""
        logger.info("🔗 Combining corpora...")
        
        # Combine dataframes
        combined_df = pd.concat([existing_df, survey_df], ignore_index=True)
        
        # Sort by corpus source then chunk_id for consistency
        combined_df = combined_df.sort_values(['corpus_source', 'chunk_id']).reset_index(drop=True)
        
        logger.info(f"  ✅ Combined corpus: {len(combined_df):,} total chunks")
        logger.info(f"    - Enhanced LW: {len(existing_df):,} chunks")
        logger.info(f"    - Survey: {len(survey_df):,} chunks")
        
        return combined_df
    
    def validate_integration(self, combined_df: pd.DataFrame) -> dict:
        """Validate the integrated corpus"""
        logger.info("✅ Validating integrated corpus...")
        
        validation = {
            'total_chunks': len(combined_df),
            'unique_chunk_ids': combined_df['chunk_id'].nunique(),
            'unique_doc_ids': combined_df['doc_id'].nunique(),
            'corpus_sources': combined_df['corpus_source'].value_counts().to_dict(),
            'content_types': combined_df['content_type'].value_counts().to_dict(),
            'structure_types': combined_df['structure_type'].value_counts().to_dict() if 'structure_type' in combined_df.columns else {},
            'total_content_chars': combined_df['content'].str.len().sum(),
            'avg_chunk_size': combined_df['content'].str.len().mean(),
            'schema_columns': len(combined_df.columns),
            'has_duplicates': combined_df['chunk_id'].duplicated().any()
        }
        
        logger.info("📊 INTEGRATION VALIDATION:")
        logger.info(f"  Total chunks: {validation['total_chunks']:,}")
        logger.info(f"  Unique chunk IDs: {validation['unique_chunk_ids']:,}")
        logger.info(f"  Unique documents: {validation['unique_doc_ids']:,}")
        logger.info(f"  Content sources: {validation['corpus_sources']}")
        logger.info(f"  Total content: {validation['total_content_chars']:,} characters")
        logger.info(f"  Average chunk size: {validation['avg_chunk_size']:.0f} characters")
        logger.info(f"  Schema columns: {validation['schema_columns']}")
        logger.info(f"  Has duplicates: {validation['has_duplicates']}")
        
        if validation['has_duplicates']:
            logger.warning("⚠️  Duplicate chunk IDs detected!")
        else:
            logger.info("✅ No duplicate chunk IDs found")
        
        return validation
    
    def save_integrated_corpus(self, combined_df: pd.DataFrame):
        """Save the integrated corpus"""
        logger.info("💾 Saving integrated corpus...")
        
        output_file = self.output_dir / 'integrated_corpus_with_survey.parquet'
        combined_df.to_parquet(output_file, index=False)
        
        logger.info(f"✅ Saved integrated corpus to {output_file}")
        logger.info(f"   File size: {output_file.stat().st_size / 1024**2:.1f} MB")
        
        return output_file
    
    def run_integration(self) -> tuple:
        """Run complete integration pipeline"""
        logger.info("🚀 Starting survey corpus integration...")
        
        # Load both corpora
        existing_df = self.load_existing_corpus()
        survey_df = self.load_survey_chunks()
        
        # Harmonize schemas
        survey_harmonized = self.harmonize_schemas(existing_df, survey_df)
        
        # Add source tags
        existing_tagged, survey_tagged = self.add_corpus_source_tags(existing_df, survey_harmonized)
        
        # Combine corpora
        combined_df = self.combine_corpora(existing_tagged, survey_tagged)
        
        # Validate integration
        validation = self.validate_integration(combined_df)
        
        # Save integrated corpus
        output_file = self.save_integrated_corpus(combined_df)
        
        # Save validation stats
        stats = {
            **validation,
            'integration_timestamp': datetime.now().isoformat(),
            'enhanced_lw_chunks': len(existing_df),
            'survey_chunks': len(survey_harmonized),
            'output_file': str(output_file)
        }
        
        stats_file = self.output_dir / 'integration_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info("🎉 SURVEY INTEGRATION COMPLETE!")
        logger.info(f"📈 Final corpus: {len(combined_df):,} chunks from {validation['unique_doc_ids']:,} documents")
        logger.info(f"📊 Enhanced LW: {len(existing_df):,} chunks + Survey: {len(survey_harmonized):,} chunks")
        logger.info(f"💾 Output: {output_file}")
        logger.info(f"📋 Stats: {stats_file}")
        
        return combined_df, stats

def main():
    integrator = SurveyCorpusIntegrator()
    combined_df, stats = integrator.run_integration()
    
    print(f"\n🎯 Survey corpus integration completed!")
    print(f"  - Total chunks: {len(combined_df):,}")
    print(f"  - Enhanced LW: {stats['enhanced_lw_chunks']:,}")
    print(f"  - Survey: {stats['survey_chunks']:,}")
    print(f"  - Ready for hybrid retrieval")

if __name__ == '__main__':
    main()