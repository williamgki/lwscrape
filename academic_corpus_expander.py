#!/usr/bin/env python3
"""
Academic Corpus Expander
Integrates manifest builder with existing Wilson Lin chunking pipeline
"""

import sqlite3
import pandas as pd
from pathlib import Path
import requests
import logging
from typing import List, Dict
import hashlib
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AcademicCorpusExpander:
    def __init__(self, manifest_path: str = "data/raw_sources.manifest"):
        self.manifest_path = manifest_path
        self.output_dir = Path("/home/ubuntu/LW_scrape/academic_expansion")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_manifest(self) -> pd.DataFrame:
        """Load papers from manifest database"""
        if not Path(self.manifest_path).exists():
            logger.error(f"Manifest not found: {self.manifest_path}")
            return pd.DataFrame()
            
        conn = sqlite3.connect(self.manifest_path)
        df = pd.read_sql_query("""
            SELECT * FROM manifest 
            WHERE s3_path IS NOT NULL 
            ORDER BY cited_by_count DESC NULLS LAST
        """, conn)
        conn.close()
        
        logger.info(f"📚 Loaded {len(df)} papers from manifest")
        return df
    
    def filter_high_relevance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for papers likely to be AI alignment relevant"""
        
        # Alignment-specific keywords for enhanced filtering
        alignment_keywords = [
            'alignment', 'interpretability', 'safety', 'robustness',
            'reward hacking', 'mesa-optimizer', 'inner alignment', 'outer alignment',
            'deception', 'oversight', 'amplification', 'debate', 'rlhf', 'rlaif'
        ]
        
        # Filter by title/abstract keywords
        mask = df['title'].str.lower().str.contains('|'.join(alignment_keywords), na=False)
        
        # Prioritize highly cited papers
        high_cited = df['cited_by_count'].fillna(0) >= 10
        
        # Recent papers (last 3 years) 
        recent = df['publication_year'].fillna(0) >= 2022
        
        filtered = df[mask | (high_cited & recent)]
        logger.info(f"🔍 Filtered to {len(filtered)} high-relevance papers")
        
        return filtered.head(500)  # Limit for processing
    
    def check_against_existing_corpus(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove papers already in our corpus"""
        
        existing_corpus_path = "/home/ubuntu/LW_scrape/scored_corpus/ai_relevance_scored_corpus.parquet"
        if not Path(existing_corpus_path).exists():
            return df
            
        existing_df = pd.read_parquet(existing_corpus_path)
        existing_titles = set(existing_df['title'].str.lower().unique())
        existing_dois = set([doi.lower() for doi in existing_df.get('doi', []) if doi])
        
        # Filter out existing papers
        new_mask = ~(
            df['title'].str.lower().isin(existing_titles) |
            df['doi'].str.lower().isin(existing_dois)
        )
        
        new_papers = df[new_mask]
        logger.info(f"🆕 Found {len(new_papers)} new papers not in existing corpus")
        
        return new_papers
    
    def download_and_process_papers(self, df: pd.DataFrame) -> List[Dict]:
        """Download PDFs and prepare for Wilson Lin chunking"""
        
        processed_papers = []
        
        for idx, paper in df.iterrows():
            logger.info(f"📄 Processing: {paper['title'][:50]}...")
            
            # Download PDF if not already cached
            pdf_path = self.output_dir / f"{paper.name}.pdf"
            
            if not pdf_path.exists() and paper.get('pdf_url'):
                try:
                    response = requests.get(paper['pdf_url'], timeout=30)
                    response.raise_for_status()
                    
                    with open(pdf_path, 'wb') as f:
                        f.write(response.content)
                        
                    logger.info(f"✅ Downloaded: {pdf_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Download failed: {e}")
                    continue
            
            # Prepare metadata for chunking
            paper_metadata = {
                'doc_id': hashlib.sha256(paper['title'].encode()).hexdigest(),
                'title': paper['title'],
                'url': paper.get('pdf_url', ''),
                'domain': 'academic',
                'content_type': 'pdf',
                'authors': paper.get('first_author', ''),
                'publication_year': paper.get('publication_year'),
                'cited_by_count': paper.get('cited_by_count', 0),
                'source_apis': paper.get('sources', 'unknown'),
                'openalex_id': paper.get('openalex_id'),
                'arxiv_id': paper.get('arxiv_id'),
                'doi': paper.get('doi'),
                'pdf_path': str(pdf_path) if pdf_path.exists() else None
            }
            
            processed_papers.append(paper_metadata)
            
        return processed_papers
    
    def integrate_with_wilson_lin(self, papers: List[Dict]) -> str:
        """Prepare papers for Wilson Lin contextual chunking"""
        
        # Create input format compatible with Wilson Lin chunker
        documents_for_chunking = []
        
        for paper in papers:
            if not paper.get('pdf_path'):
                continue
                
            doc_record = {
                'doc_id': paper['doc_id'],
                'title': paper['title'], 
                'url': paper['url'],
                'domain': 'academic_expansion',
                'content_type': 'pdf',
                'language': 'en',
                'structure_type': 'academic',
                'metadata': {
                    'authors': paper.get('authors'),
                    'year': paper.get('publication_year'),
                    'citations': paper.get('cited_by_count', 0),
                    'source_apis': paper.get('source_apis'),
                    'identifiers': {
                        'openalex_id': paper.get('openalex_id'),
                        'arxiv_id': paper.get('arxiv_id'),
                        'doi': paper.get('doi')
                    }
                },
                'file_path': paper['pdf_path']
            }
            
            documents_for_chunking.append(doc_record)
        
        # Save documents list for chunking pipeline
        output_file = self.output_dir / 'academic_papers_for_chunking.json'
        with open(output_file, 'w') as f:
            json.dump(documents_for_chunking, f, indent=2, default=str)
            
        logger.info(f"📋 Prepared {len(documents_for_chunking)} papers for chunking: {output_file}")
        return str(output_file)

def main():
    expander = AcademicCorpusExpander()
    
    # Load and filter papers from manifest
    manifest_df = expander.load_manifest()
    if manifest_df.empty:
        logger.error("No manifest data found")
        return
        
    high_relevance_papers = expander.filter_high_relevance(manifest_df)
    new_papers = expander.check_against_existing_corpus(high_relevance_papers)
    
    if new_papers.empty:
        logger.info("🎯 No new papers to process")
        return
    
    # Process papers
    processed_papers = expander.download_and_process_papers(new_papers)
    documents_file = expander.integrate_with_wilson_lin(processed_papers)
    
    logger.info("="*60)
    logger.info("🎯 ACADEMIC CORPUS EXPANSION COMPLETE")
    logger.info("="*60)
    logger.info(f"📚 New papers identified: {len(new_papers)}")
    logger.info(f"📄 Papers processed: {len(processed_papers)}")
    logger.info(f"📋 Documents ready for chunking: {documents_file}")
    logger.info("="*60)
    logger.info("🔄 Next steps:")
    logger.info("1. Run Wilson Lin chunker on the prepared documents")
    logger.info("2. Add AI summaries to new chunks")
    logger.info("3. Integrate with existing scored corpus")

if __name__ == "__main__":
    main()