#!/usr/bin/env python3
"""
Multi-Source to Wilson Lin Pipeline
Processes collected papers through Wilson Lin contextual chunking with AI summaries
"""

import json
import sqlite3
import requests
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import argparse

from directory_config import load_config

# Import our existing Wilson Lin chunker
import sys
sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = load_config()


class MultiSourceProcessor:
    def __init__(self,
                 corpus_dir: str = str(config.corpus_dir),
                 output_dir: str = str(config.output_dir),
                 temp_dir: str = str(config.temp_dir)):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.manifest_db = self.corpus_dir / "multi_source_manifest.db"
        
        # Output directories
        self.papers_dir = self.corpus_dir / "downloaded_papers"
        self.papers_dir.mkdir(exist_ok=True)
        
        self.normalized_dir = self.corpus_dir / "normalized_documents"  
        self.normalized_dir.mkdir(exist_ok=True)
        
        self.chunks_dir = self.corpus_dir / "wilson_lin_chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
    def load_deduplicated_papers(self) -> List[Dict]:
        """Load deduplicated papers from database"""
        if not self.manifest_db.exists():
            logger.error(f"Manifest database not found: {self.manifest_db}")
            return []
            
        conn = sqlite3.connect(self.manifest_db)
        
        # Get papers with PDF URLs
        query = """
            SELECT * FROM papers 
            WHERE pdf_url IS NOT NULL 
            AND relevance_score >= 0.3
            ORDER BY relevance_score DESC, cited_by_count DESC
            LIMIT 100
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        papers = df.to_dict('records')
        logger.info(f"📚 Loaded {len(papers)} papers with PDFs for processing")
        
        return papers
    
    def download_paper_pdfs(self, papers: List[Dict]) -> List[Dict]:
        """Download PDFs for papers"""
        downloaded_papers = []
        
        for i, paper in enumerate(papers):
            logger.info(f"📄 Processing paper {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # Generate paper filename
            paper_id = paper.get('paper_id') or f"paper_{paper['id']}"
            pdf_filename = f"{paper_id}.pdf"
            pdf_path = self.papers_dir / pdf_filename
            
            # Skip if already downloaded
            if pdf_path.exists():
                logger.info(f"✅ Already downloaded: {pdf_filename}")
                paper_info = self.create_paper_metadata(paper, pdf_path)
                downloaded_papers.append(paper_info)
                continue
            
            # Download PDF
            pdf_url = paper.get('pdf_url')
            if not pdf_url:
                logger.warning(f"⚠️ No PDF URL for paper: {paper['title'][:50]}")
                continue
            
            try:
                headers = {'User-Agent': 'multi-source-processor/1.0'}
                response = requests.get(pdf_url, timeout=30, headers=headers)
                response.raise_for_status()
                
                # Save PDF
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✅ Downloaded: {pdf_filename} ({len(response.content)} bytes)")
                
                # Create paper metadata
                paper_info = self.create_paper_metadata(paper, pdf_path)
                downloaded_papers.append(paper_info)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to download {paper['title'][:50]}: {e}")
                continue
        
        logger.info(f"📁 Downloaded {len(downloaded_papers)} papers successfully")
        return downloaded_papers
    
    def create_paper_metadata(self, paper: Dict, pdf_path: Path) -> Dict:
        """Create metadata structure compatible with Wilson Lin chunker"""
        
        # Generate document ID
        title = paper.get('title', '')
        doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
        
        # Parse authors
        authors = paper.get('authors', '')
        if authors:
            author_list = [name.strip() for name in authors.split(';')][:3]
        else:
            author_list = []
        
        # Parse concepts/keywords
        concepts = []
        if paper.get('concepts'):
            try:
                concepts = json.loads(paper['concepts'])
            except:
                pass
        
        keywords = paper.get('keywords', '').split(',') if paper.get('keywords') else []
        
        paper_metadata = {
            'doc_id': doc_id,
            'title': title,
            'url': paper.get('pdf_url', ''),
            'domain': 'academic_multisource',
            'content_type': 'pdf',
            'language': 'en',
            'structure_type': 'academic',
            'file_path': str(pdf_path),
            'source_metadata': {
                'primary_source': paper.get('primary_source'),
                'sources': json.loads(paper.get('sources', '[]')),
                'authors': author_list,
                'publication_year': paper.get('publication_year'),
                'publication_date': paper.get('publication_date'),
                'venue': paper.get('venue'),
                'abstract': paper.get('abstract', ''),
                'cited_by_count': paper.get('cited_by_count', 0),
                'concepts': concepts,
                'keywords': keywords,
                'relevance_score': paper.get('relevance_score', 0.0),
                'identifiers': {
                    'openalex_id': paper.get('openalex_id'),
                    'arxiv_id': paper.get('arxiv_id'),
                    'openreview_id': paper.get('openreview_id'),
                    'doi': paper.get('doi')
                },
                'is_open_access': paper.get('is_open_access', False)
            }
        }
        
        return paper_metadata
    
    def create_wilson_lin_input(self, papers: List[Dict]) -> str:
        """Create input file for Wilson Lin chunker"""
        
        # Convert to format expected by Wilson Lin chunker
        documents = []
        
        for paper in papers:
            doc_record = {
                'doc_id': paper['doc_id'],
                'title': paper['title'],
                'url': paper['url'],
                'domain': paper['domain'],
                'content_type': paper['content_type'],
                'language': paper['language'],
                'structure_type': paper['structure_type'],
                'file_path': paper['file_path'],
                'metadata': paper['source_metadata']
            }
            documents.append(doc_record)
        
        # Save as JSON for Wilson Lin chunker
        input_file = self.corpus_dir / "papers_for_chunking.json"
        with open(input_file, 'w') as f:
            json.dump(documents, f, indent=2, default=str)
        
        logger.info(f"📋 Created Wilson Lin input file: {input_file}")
        logger.info(f"📊 {len(documents)} papers ready for chunking")
        
        return str(input_file)
    
    def run_wilson_lin_chunking(self, input_file: str) -> str:
        """Run Wilson Lin contextual chunking on the papers"""
        
        logger.info("🔄 Starting Wilson Lin contextual chunking...")
        
        # Import and run Wilson Lin chunker
        try:
            from wilson_lin_chunker import WilsonLinContextualChunker
            
            chunker = WilsonLinContextualChunker()
            chunker.output_dir = self.chunks_dir
            
            # Load documents
            with open(input_file, 'r') as f:
                documents = json.load(f)
            
            logger.info(f"📚 Processing {len(documents)} documents...")
            
            # Process documents
            chunks = []
            for i, doc in enumerate(documents):
                logger.info(f"📄 Chunking document {i+1}/{len(documents)}: {doc['title'][:50]}...")
                
                try:
                    doc_chunks = chunker.process_document(doc)
                    chunks.extend(doc_chunks)
                    logger.info(f"✅ Generated {len(doc_chunks)} chunks")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to chunk document {doc['title'][:50]}: {e}")
                    continue
            
            # Save chunks
            output_file = self.chunks_dir / "multisource_chunks.parquet"
            
            if chunks:
                chunks_df = pd.DataFrame(chunks)
                chunks_df.to_parquet(output_file, index=False)
                
                logger.info(f"💾 Saved {len(chunks)} chunks to {output_file}")
                
                # Generate stats
                stats = {
                    'total_documents': len(documents),
                    'total_chunks': len(chunks),
                    'avg_chunks_per_doc': len(chunks) / len(documents) if documents else 0,
                    'processing_timestamp': datetime.now().isoformat(),
                    'chunk_schema': list(chunks_df.columns) if chunks else [],
                    'source_breakdown': chunks_df['domain'].value_counts().to_dict() if chunks else {}
                }
                
                stats_file = self.chunks_dir / "chunking_stats.json"
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                
                return str(output_file)
            else:
                logger.error("❌ No chunks generated")
                return ""
                
        except ImportError as e:
            logger.error(f"❌ Failed to import Wilson Lin chunker: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Wilson Lin chunking failed: {e}")
            return ""
    
    def integrate_with_existing_corpus(self, chunks_file: str):
        """Integrate new chunks with existing scored corpus"""
        
        if not chunks_file or not Path(chunks_file).exists():
            logger.error("No chunks file to integrate")
            return
        
        logger.info("🔗 Integrating with existing scored corpus...")
        
        # Load new chunks
        new_chunks = pd.read_parquet(chunks_file)
        
        # Load existing scored corpus
        existing_corpus_path = self.output_dir / "ai_relevance_scored_corpus.parquet"

        if existing_corpus_path.exists():
            existing_corpus = pd.read_parquet(existing_corpus_path)
            
            # Combine corpora
            combined_corpus = pd.concat([existing_corpus, new_chunks], ignore_index=True)
            
            # Remove duplicates based on chunk_id
            initial_count = len(combined_corpus)
            combined_corpus = combined_corpus.drop_duplicates(subset=['chunk_id'], keep='first')
            final_count = len(combined_corpus)
            
            # Save expanded corpus
            expanded_output = self.corpus_dir / "expanded_corpus_with_multisource.parquet"
            combined_corpus.to_parquet(expanded_output, index=False)
            
            logger.info(f"🎯 Expanded corpus created:")
            logger.info(f"   Original corpus: {len(existing_corpus):,} chunks")
            logger.info(f"   New chunks: {len(new_chunks):,} chunks")
            logger.info(f"   Combined total: {final_count:,} chunks")
            logger.info(f"   Duplicates removed: {initial_count - final_count}")
            logger.info(f"   Output: {expanded_output}")
            
        else:
            logger.info(f"📊 New corpus only (no existing corpus to merge): {len(new_chunks)} chunks")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process multi-source papers through Wilson Lin chunking")
    parser.add_argument("--corpus-dir", default=str(config.corpus_dir))
    parser.add_argument("--output-dir", default=str(config.output_dir))
    parser.add_argument("--temp-dir", default=str(config.temp_dir))
    parser.add_argument("--skip-download", action="store_true", help="Skip PDF download")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip Wilson Lin chunking")

    args = parser.parse_args()

    processor = MultiSourceProcessor(args.corpus_dir, args.output_dir, args.temp_dir)
    
    logger.info("🚀 Starting multi-source to Wilson Lin pipeline...")
    
    # 1. Load papers
    papers = processor.load_deduplicated_papers()
    if not papers:
        logger.error("No papers to process")
        return
    
    # 2. Download PDFs
    if not args.skip_download:
        downloaded_papers = processor.download_paper_pdfs(papers)
    else:
        # Create metadata for existing PDFs
        downloaded_papers = []
        for paper in papers:
            paper_id = paper.get('paper_id') or f"paper_{paper['id']}"
            pdf_path = processor.papers_dir / f"{paper_id}.pdf"
            if pdf_path.exists():
                paper_info = processor.create_paper_metadata(paper, pdf_path)
                downloaded_papers.append(paper_info)
        logger.info(f"📁 Found {len(downloaded_papers)} existing papers")
    
    if not downloaded_papers:
        logger.error("No papers available for chunking")
        return
    
    # 3. Create Wilson Lin input
    input_file = processor.create_wilson_lin_input(downloaded_papers)
    
    # 4. Run Wilson Lin chunking
    if not args.skip_chunking:
        chunks_file = processor.run_wilson_lin_chunking(input_file)
        
        # 5. Integrate with existing corpus
        if chunks_file:
            processor.integrate_with_existing_corpus(chunks_file)
    
    logger.info("="*60)
    logger.info("🎯 MULTI-SOURCE TO WILSON LIN COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()