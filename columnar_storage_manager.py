#!/usr/bin/env python3
"""
Columnar Storage Manager
Manages Parquet files with DuckDB for efficient querying and S3 persistence
"""

import duckdb
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColumnarStorageManager:
    def __init__(self, 
                 local_storage_path: str = "/home/ubuntu/LW_scrape/columnar_storage",
                 s3_bucket: str = "alignment-corpus",
                 s3_prefix: str = "processed/"):
        
        self.local_path = Path(local_storage_path)
        self.local_path.mkdir(exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        # Initialize DuckDB connection
        self.db_path = self.local_path / "alignment_corpus.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        
        # Initialize S3 client (optional)
        try:
            self.s3_client = boto3.client('s3')
            self.s3_available = True
        except Exception as e:
            logger.warning(f"S3 not available: {e}")
            self.s3_client = None
            self.s3_available = False
        
        # Table schemas
        self.init_table_schemas()
    
    def init_table_schemas(self):
        """Initialize DuckDB table schemas"""
        
        # Papers table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                doc_id VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                abstract TEXT,
                authors JSON,
                publication_year INTEGER,
                publication_date DATE,
                venue VARCHAR,
                doi VARCHAR,
                openalex_id VARCHAR,
                arxiv_id VARCHAR,
                openreview_id VARCHAR,
                pdf_url VARCHAR,
                cited_by_count INTEGER DEFAULT 0,
                concepts JSON,
                keywords VARCHAR,
                relevance_score DOUBLE DEFAULT 0.0,
                primary_source VARCHAR,
                sources JSON,
                is_open_access BOOLEAN DEFAULT false,
                processing_timestamp TIMESTAMP,
                file_path VARCHAR
            )
        """)
        
        # Chunks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                content TEXT NOT NULL,
                prev_context TEXT,
                section_path VARCHAR,
                summary_header TEXT,
                token_count INTEGER,
                char_count INTEGER,
                chunk_index INTEGER,
                section_type VARCHAR,
                hierarchical_level INTEGER,
                contains_figures JSON,
                citations JSON,
                structural_metadata JSON,
                ai_relevance_score DOUBLE DEFAULT 0.0,
                title VARCHAR,
                url VARCHAR,
                domain VARCHAR,
                content_type VARCHAR,
                language VARCHAR,
                structure_type VARCHAR,
                processing_timestamp TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES papers(doc_id)
            )
        """)
        
        # References table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS references (
                ref_id VARCHAR PRIMARY KEY,
                source_doc_id VARCHAR NOT NULL,
                raw_text TEXT,
                original_title VARCHAR,
                original_authors JSON,
                original_year INTEGER,
                original_venue VARCHAR,
                original_doi VARCHAR,
                crossref_doi VARCHAR,
                openalex_id VARCHAR,
                crossref_score DOUBLE DEFAULT 0.0,
                normalization_method VARCHAR,
                canonical_title VARCHAR,
                canonical_authors JSON,
                canonical_venue VARCHAR,
                citation_count INTEGER DEFAULT 0,
                referenced_works JSON,
                citing_works JSON,
                concepts JSON,
                processing_timestamp TIMESTAMP,
                FOREIGN KEY (source_doc_id) REFERENCES papers(doc_id)
            )
        """)
        
        # Citations table (relationships between papers)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY,
                citing_doc_id VARCHAR NOT NULL,
                cited_doc_id VARCHAR,
                cited_openalex_id VARCHAR,
                citation_context TEXT,
                confidence_score DOUBLE DEFAULT 0.0,
                citation_type VARCHAR DEFAULT 'reference',
                processing_timestamp TIMESTAMP,
                FOREIGN KEY (citing_doc_id) REFERENCES papers(doc_id),
                FOREIGN KEY (cited_doc_id) REFERENCES papers(doc_id)
            )
        """)
        
        # Figures table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS figures (
                figure_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                figure_name VARCHAR,
                caption TEXT,
                page_number INTEGER,
                bbox JSON,
                figure_type VARCHAR DEFAULT 'Figure',
                image_path VARCHAR,
                processing_timestamp TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES papers(doc_id)
            )
        """)
        
        # Create indexes for performance
        self.create_indexes()
    
    def create_indexes(self):
        """Create indexes for efficient querying"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_papers_relevance ON papers(relevance_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(publication_year DESC)",
            "CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(primary_source)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_relevance ON chunks(ai_relevance_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_section_type ON chunks(section_type)",
            "CREATE INDEX IF NOT EXISTS idx_references_source ON references(source_doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_references_openalex ON references(openalex_id)",
            "CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_doc_id)"
        ]
        
        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except Exception as e:
                logger.debug(f"Index creation warning: {e}")
    
    def ingest_papers(self, papers_file: str) -> int:
        """Ingest papers from parquet file"""
        
        if not Path(papers_file).exists():
            logger.error(f"Papers file not found: {papers_file}")
            return 0
        
        logger.info(f"📚 Ingesting papers from {papers_file}")
        
        # Read parquet file
        df = pd.read_parquet(papers_file)
        
        # Prepare data for insertion
        df['processing_timestamp'] = datetime.now()
        
        # Handle JSON columns
        json_columns = ['authors', 'concepts', 'sources']
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if pd.notna(x) else None)
        
        # Insert into DuckDB
        self.conn.execute("DELETE FROM papers")  # Clear existing data
        self.conn.register('papers_df', df)
        
        self.conn.execute("""
            INSERT INTO papers SELECT * FROM papers_df
        """)
        
        count = len(df)
        logger.info(f"✅ Ingested {count} papers into DuckDB")
        
        # Save to parquet for backup
        local_parquet = self.local_path / "papers.parquet" 
        df.to_parquet(local_parquet, index=False)
        
        # Upload to S3 if available
        if self.s3_available:
            self.upload_to_s3(local_parquet, "papers.parquet")
        
        return count
    
    def ingest_chunks(self, chunks_file: str) -> int:
        """Ingest chunks from parquet file"""
        
        if not Path(chunks_file).exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            return 0
        
        logger.info(f"📄 Ingesting chunks from {chunks_file}")
        
        # Read parquet file
        df = pd.read_parquet(chunks_file)
        
        # Prepare data
        df['processing_timestamp'] = datetime.now()
        
        # Handle JSON columns
        json_columns = ['contains_figures', 'citations', 'structural_metadata']
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, str) else json.dumps(x) if pd.notna(x) else None)
        
        # Insert into DuckDB
        self.conn.execute("DELETE FROM chunks")  # Clear existing data
        self.conn.register('chunks_df', df)
        
        self.conn.execute("""
            INSERT INTO chunks SELECT * FROM chunks_df
        """)
        
        count = len(df)
        logger.info(f"✅ Ingested {count} chunks into DuckDB")
        
        # Save to parquet
        local_parquet = self.local_path / "chunks.parquet"
        df.to_parquet(local_parquet, index=False)
        
        # Upload to S3 if available
        if self.s3_available:
            self.upload_to_s3(local_parquet, "chunks.parquet")
        
        return count
    
    def ingest_references(self, references_file: str) -> int:
        """Ingest normalized references"""
        
        if not Path(references_file).exists():
            logger.error(f"References file not found: {references_file}")
            return 0
        
        logger.info(f"📖 Ingesting references from {references_file}")
        
        # Read parquet file
        df = pd.read_parquet(references_file)
        
        # Prepare data
        df['processing_timestamp'] = datetime.now()
        
        # Extract source_doc_id from ref_id (assumes format: doc_id_ref_###)
        df['source_doc_id'] = df['ref_id'].str.extract(r'(.+)_ref_\d+')[0]
        
        # Handle JSON columns
        json_columns = ['original_authors', 'canonical_authors', 'referenced_works', 'citing_works', 'concepts']
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, str) else json.dumps(x) if pd.notna(x) else None)
        
        # Insert into DuckDB
        self.conn.execute("DELETE FROM references")
        self.conn.register('references_df', df)
        
        self.conn.execute("""
            INSERT INTO references SELECT * FROM references_df
        """)
        
        count = len(df)
        logger.info(f"✅ Ingested {count} references into DuckDB")
        
        # Save to parquet
        local_parquet = self.local_path / "references.parquet"
        df.to_parquet(local_parquet, index=False)
        
        # Upload to S3 if available
        if self.s3_available:
            self.upload_to_s3(local_parquet, "references.parquet")
        
        return count
    
    def build_citation_graph(self) -> int:
        """Build citation relationships between papers"""
        
        logger.info("🔗 Building citation graph from references...")
        
        # Clear existing citations
        self.conn.execute("DELETE FROM citations")
        
        # Insert citations based on normalized references
        citation_query = """
            INSERT INTO citations (citing_doc_id, cited_openalex_id, confidence_score, processing_timestamp)
            SELECT DISTINCT 
                r.source_doc_id as citing_doc_id,
                r.openalex_id as cited_openalex_id,
                r.crossref_score as confidence_score,
                CURRENT_TIMESTAMP as processing_timestamp
            FROM references r
            WHERE r.openalex_id IS NOT NULL 
            AND r.crossref_score > 0.5
        """
        
        self.conn.execute(citation_query)
        
        # Count citations created
        result = self.conn.execute("SELECT COUNT(*) FROM citations").fetchone()
        citation_count = result[0] if result else 0
        
        logger.info(f"✅ Built citation graph with {citation_count} citation edges")
        
        return citation_count
    
    def upload_to_s3(self, local_file: Path, s3_key: str) -> bool:
        """Upload file to S3"""
        
        if not self.s3_available:
            return False
        
        try:
            full_s3_key = f"{self.s3_prefix}{s3_key}"
            
            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                full_s3_key
            )
            
            logger.info(f"☁️ Uploaded to s3://{self.s3_bucket}/{full_s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 upload failed for {s3_key}: {e}")
            return False
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        
        try:
            result = self.conn.execute(sql).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()
    
    def get_corpus_stats(self) -> Dict:
        """Get comprehensive corpus statistics"""
        
        stats = {}
        
        # Paper statistics
        papers_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_papers,
                COUNT(CASE WHEN relevance_score > 0.7 THEN 1 END) as high_relevance_papers,
                AVG(relevance_score) as avg_relevance_score,
                COUNT(DISTINCT primary_source) as unique_sources,
                MIN(publication_year) as earliest_year,
                MAX(publication_year) as latest_year,
                SUM(cited_by_count) as total_citations
            FROM papers
        """).fetchone()
        
        if papers_stats:
            stats['papers'] = {
                'total': papers_stats[0],
                'high_relevance': papers_stats[1], 
                'avg_relevance_score': round(papers_stats[2] or 0, 3),
                'unique_sources': papers_stats[3],
                'year_range': f"{papers_stats[4]}-{papers_stats[5]}" if papers_stats[4] else "N/A",
                'total_citations': papers_stats[6] or 0
            }
        
        # Chunk statistics
        chunks_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                AVG(token_count) as avg_tokens,
                AVG(ai_relevance_score) as avg_ai_score,
                COUNT(DISTINCT section_type) as unique_section_types
            FROM chunks
        """).fetchone()
        
        if chunks_stats:
            stats['chunks'] = {
                'total': chunks_stats[0],
                'avg_tokens': round(chunks_stats[1] or 0, 1),
                'avg_ai_score': round(chunks_stats[2] or 0, 3),
                'unique_section_types': chunks_stats[3]
            }
        
        # Reference statistics
        ref_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_references,
                COUNT(CASE WHEN openalex_id IS NOT NULL THEN 1 END) as normalized_references,
                AVG(crossref_score) as avg_crossref_score
            FROM references
        """).fetchone()
        
        if ref_stats:
            stats['references'] = {
                'total': ref_stats[0],
                'normalized': ref_stats[1],
                'normalization_rate': round((ref_stats[1] / ref_stats[0] * 100) if ref_stats[0] else 0, 1),
                'avg_crossref_score': round(ref_stats[2] or 0, 3)
            }
        
        # Citation graph statistics
        citation_stats = self.conn.execute("""
            SELECT COUNT(*) as total_edges FROM citations
        """).fetchone()
        
        if citation_stats:
            stats['citation_graph'] = {
                'total_edges': citation_stats[0]
            }
        
        return stats
    
    def export_for_analysis(self, output_dir: str = None) -> Dict[str, str]:
        """Export optimized parquet files for analysis"""
        
        if output_dir is None:
            output_dir = self.local_path / "export"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        exports = {}
        
        # Export high-relevance papers
        high_rel_papers = self.query("""
            SELECT * FROM papers 
            WHERE relevance_score > 0.5 
            ORDER BY relevance_score DESC
        """)
        
        if not high_rel_papers.empty:
            export_file = output_dir / "high_relevance_papers.parquet"
            high_rel_papers.to_parquet(export_file, index=False)
            exports['high_relevance_papers'] = str(export_file)
        
        # Export high-scoring chunks
        top_chunks = self.query("""
            SELECT * FROM chunks 
            WHERE ai_relevance_score > 0.6
            ORDER BY ai_relevance_score DESC
        """)
        
        if not top_chunks.empty:
            export_file = output_dir / "top_ai_chunks.parquet"
            top_chunks.to_parquet(export_file, index=False)
            exports['top_ai_chunks'] = str(export_file)
        
        # Export citation graph
        citations = self.query("SELECT * FROM citations")
        if not citations.empty:
            export_file = output_dir / "citation_graph.parquet"
            citations.to_parquet(export_file, index=False)
            exports['citation_graph'] = str(export_file)
        
        # Export normalized references
        norm_refs = self.query("""
            SELECT * FROM references 
            WHERE normalization_method != 'failed'
            ORDER BY crossref_score DESC
        """)
        
        if not norm_refs.empty:
            export_file = output_dir / "normalized_references.parquet"
            norm_refs.to_parquet(export_file, index=False)
            exports['normalized_references'] = str(export_file)
        
        logger.info(f"📊 Exported {len(exports)} analysis datasets to {output_dir}")
        return exports
    
    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage columnar storage with DuckDB")
    parser.add_argument("--papers-file", help="Parquet file with papers")
    parser.add_argument("--chunks-file", help="Parquet file with chunks")  
    parser.add_argument("--references-file", help="Parquet file with references")
    parser.add_argument("--build-citations", action="store_true", help="Build citation graph")
    parser.add_argument("--export", action="store_true", help="Export analysis datasets")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    parser.add_argument("--query", help="Execute SQL query")
    
    args = parser.parse_args()
    
    storage = ColumnarStorageManager()
    
    try:
        # Ingest data
        if args.papers_file:
            storage.ingest_papers(args.papers_file)
        
        if args.chunks_file:
            storage.ingest_chunks(args.chunks_file)
        
        if args.references_file:
            storage.ingest_references(args.references_file)
        
        # Build citation graph
        if args.build_citations:
            storage.build_citation_graph()
        
        # Show statistics
        if args.stats:
            stats = storage.get_corpus_stats()
            print("\n📊 CORPUS STATISTICS:")
            print("="*50)
            print(json.dumps(stats, indent=2))
        
        # Execute query
        if args.query:
            result = storage.query(args.query)
            print(f"\n🔍 QUERY RESULTS:")
            print(result.to_string(index=False))
        
        # Export datasets
        if args.export:
            exports = storage.export_for_analysis()
            print(f"\n📤 EXPORTED DATASETS:")
            for name, path in exports.items():
                print(f"  {name}: {path}")
    
    finally:
        storage.close()

if __name__ == "__main__":
    main()