#!/usr/bin/env python3

import duckdb
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import re
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class QueryResult:
    para_id: str
    paper_id: str
    section_id: str
    page: int
    type: str
    text: str
    score: float
    paper_title: Optional[str] = None
    paper_authors: Optional[List[str]] = None
    figure_meta: Optional[Dict] = None
    refs: Optional[List[str]] = None


class DocUnitsQueryEngine:
    """Advanced query engine for DocUnits with semantic search capabilities"""
    
    def __init__(self, db_path: str = "docunits.db", parquet_dir: Optional[Path] = None):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # Load from Parquet if database is empty
        if parquet_dir and Path(parquet_dir).exists():
            self._load_from_parquet(Path(parquet_dir))
        
        # Create full-text search index
        self._setup_full_text_search()
    
    def _load_from_parquet(self, parquet_dir: Path):
        """Load data from Parquet files if database is empty"""
        try:
            # Check if tables are empty
            result = self.conn.execute("SELECT COUNT(*) FROM docunits").fetchone()
            if result[0] == 0:
                docunits_file = parquet_dir / "DocUnits.parquet"
                if docunits_file.exists():
                    self.conn.execute(f"INSERT INTO docunits SELECT * FROM '{docunits_file}'")
                
                edges_file = parquet_dir / "DocRefEdges.parquet"
                if edges_file.exists():
                    self.conn.execute(f"INSERT INTO docrefedges SELECT * FROM '{edges_file}'")
                
                print(f"Loaded data from {parquet_dir}")
        except Exception as e:
            print(f"Error loading from Parquet: {e}")
    
    def _setup_full_text_search(self):
        """Setup full-text search using DuckDB FTS extension"""
        try:
            # Install and load FTS extension
            self.conn.execute("INSTALL fts")
            self.conn.execute("LOAD fts")
            
            # Create FTS index on text content
            self.conn.execute("""
                PRAGMA create_fts_index(
                    'docunits', 
                    'para_id', 
                    'text',
                    stemmer = 'english',
                    stopwords = 'english',
                    ignore = '(\\.|[^a-z])+',
                    strip_accents = 1,
                    lower = 1,
                    overwrite = 1
                )
            """)
        except Exception as e:
            print(f"FTS setup warning: {e}")
    
    def search_text(self, 
                   query: str,
                   limit: int = 20,
                   min_score: float = 0.1,
                   section_filter: Optional[str] = None,
                   type_filter: Optional[str] = None,
                   paper_filter: Optional[str] = None) -> List[QueryResult]:
        """Full-text search across DocUnits"""
        
        # Build SQL query
        sql = """
        SELECT 
            d.para_id,
            d.paper_id,
            d.section_id,
            d.page,
            d.type,
            d.text,
            fts_main_docunits.score as score,
            d.figure_meta,
            d.refs
        FROM docunits d
        JOIN (
            SELECT para_id, score 
            FROM fts_main_docunits($1)
            WHERE score >= $2
        ) fts ON d.para_id = fts.para_id
        """
        
        params = [query, min_score]
        param_idx = 3
        
        # Add filters
        conditions = []
        if section_filter:
            conditions.append(f"d.section_id ILIKE ${param_idx}")
            params.append(f"%{section_filter}%")
            param_idx += 1
            
        if type_filter:
            conditions.append(f"d.type = ${param_idx}")
            params.append(type_filter)
            param_idx += 1
            
        if paper_filter:
            conditions.append(f"d.paper_id ILIKE ${param_idx}")
            params.append(f"%{paper_filter}%")
            param_idx += 1
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += f" ORDER BY fts.score DESC LIMIT ${param_idx}"
        params.append(limit)
        
        try:
            results = self.conn.execute(sql, params).fetchall()
            return [self._row_to_result(row) for row in results]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def search_by_citations(self, 
                          target_paper_id: str,
                          edge_type: str = "cites",
                          include_content: bool = True,
                          limit: int = 50) -> List[QueryResult]:
        """Find DocUnits that cite or are cited by a target paper"""
        
        sql = """
        SELECT DISTINCT
            d.para_id,
            d.paper_id,
            d.section_id,
            d.page,
            d.type,
            d.text,
            1.0 as score,
            d.figure_meta,
            d.refs
        FROM docunits d
        JOIN docrefedges e ON d.paper_id = e.src_paper_id
        WHERE e.dst_paper_id = $1 AND e.edge_type = $2
        """
        
        if include_content:
            sql += " AND d.type IN ('text', 'caption')"
        
        sql += " ORDER BY d.page, d.para_id LIMIT $3"
        
        try:
            results = self.conn.execute(sql, [target_paper_id, edge_type, limit]).fetchall()
            return [self._row_to_result(row) for row in results]
        except Exception as e:
            print(f"Citation search error: {e}")
            return []
    
    def get_paper_structure(self, paper_id: str) -> Dict[str, List[QueryResult]]:
        """Get structured view of a paper's DocUnits organized by section"""
        
        sql = """
        SELECT 
            para_id, paper_id, section_id, page, type, text,
            1.0 as score, figure_meta, refs
        FROM docunits 
        WHERE paper_id = $1 
        ORDER BY page, section_id, para_id
        """
        
        try:
            results = self.conn.execute(sql, [paper_id]).fetchall()
            
            # Group by section
            sections = {}
            for row in results:
                result = self._row_to_result(row)
                section = result.section_id
                if section not in sections:
                    sections[section] = []
                sections[section].append(result)
            
            return sections
        except Exception as e:
            print(f"Structure query error: {e}")
            return {}
    
    def find_similar_content(self, 
                           para_id: str,
                           similarity_threshold: float = 0.7,
                           limit: int = 10) -> List[QueryResult]:
        """Find similar DocUnits using MinHash similarity"""
        
        # Get source unit's MinHash
        sql = """
        SELECT text, hashes.minhash_signature
        FROM docunits 
        WHERE para_id = $1
        """
        
        try:
            source = self.conn.execute(sql, [para_id]).fetchone()
            if not source:
                return []
            
            # This would implement MinHash similarity search
            # For now, use simple text similarity as placeholder
            return self.search_text(source[0][:100], limit=limit)
            
        except Exception as e:
            print(f"Similarity search error: {e}")
            return []
    
    def get_figure_references(self, paper_id: str) -> List[QueryResult]:
        """Get all figures and their references for a paper"""
        
        sql = """
        SELECT 
            para_id, paper_id, section_id, page, type, text,
            1.0 as score, figure_meta, refs
        FROM docunits 
        WHERE paper_id = $1 AND type IN ('figure', 'table', 'caption')
        ORDER BY page, figure_meta.figure_id
        """
        
        try:
            results = self.conn.execute(sql, [paper_id]).fetchall()
            return [self._row_to_result(row) for row in results]
        except Exception as e:
            print(f"Figure query error: {e}")
            return []
    
    def analyze_citation_patterns(self, limit: int = 100) -> Dict[str, Any]:
        """Analyze citation patterns in the corpus"""
        
        try:
            # Most cited papers
            most_cited = self.conn.execute("""
                SELECT dst_paper_id, COUNT(*) as citation_count
                FROM docrefedges 
                WHERE edge_type = 'is_cited_by'
                GROUP BY dst_paper_id
                ORDER BY citation_count DESC
                LIMIT $1
            """, [limit]).fetchall()
            
            # Most citing papers
            most_citing = self.conn.execute("""
                SELECT src_paper_id, COUNT(*) as citing_count
                FROM docrefedges 
                WHERE edge_type = 'cites'
                GROUP BY src_paper_id
                ORDER BY citing_count DESC
                LIMIT $1
            """, [limit]).fetchall()
            
            # Citation network stats
            network_stats = self.conn.execute("""
                SELECT 
                    COUNT(DISTINCT src_paper_id) as total_citing_papers,
                    COUNT(DISTINCT dst_paper_id) as total_cited_papers,
                    COUNT(*) as total_citations,
                    AVG(citation_count) as avg_citations_per_paper
                FROM (
                    SELECT dst_paper_id, COUNT(*) as citation_count
                    FROM docrefedges 
                    WHERE edge_type = 'is_cited_by'
                    GROUP BY dst_paper_id
                ) sub
            """).fetchone()
            
            return {
                'most_cited': most_cited,
                'most_citing': most_citing,
                'network_stats': {
                    'total_citing_papers': network_stats[0],
                    'total_cited_papers': network_stats[1],
                    'total_citations': network_stats[2],
                    'avg_citations_per_paper': network_stats[3]
                }
            }
            
        except Exception as e:
            print(f"Citation analysis error: {e}")
            return {}
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get comprehensive corpus statistics"""
        
        try:
            stats = {}
            
            # Basic counts
            stats['docunits'] = self.conn.execute("SELECT COUNT(*) FROM docunits").fetchone()[0]
            stats['papers'] = self.conn.execute("SELECT COUNT(DISTINCT paper_id) FROM docunits").fetchone()[0]
            stats['citations'] = self.conn.execute("SELECT COUNT(*) FROM docrefedges").fetchone()[0]
            
            # Content type distribution
            type_dist = self.conn.execute("""
                SELECT type, COUNT(*) as count
                FROM docunits 
                GROUP BY type 
                ORDER BY count DESC
            """).fetchall()
            stats['content_types'] = dict(type_dist)
            
            # Section distribution
            section_dist = self.conn.execute("""
                SELECT section_id, COUNT(*) as count
                FROM docunits 
                GROUP BY section_id 
                ORDER BY count DESC
                LIMIT 20
            """).fetchall()
            stats['top_sections'] = dict(section_dist)
            
            # Text length statistics
            text_stats = self.conn.execute("""
                SELECT 
                    AVG(LENGTH(text)) as avg_length,
                    MIN(LENGTH(text)) as min_length,
                    MAX(LENGTH(text)) as max_length,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(text)) as median_length
                FROM docunits 
                WHERE type = 'text'
            """).fetchone()
            
            stats['text_statistics'] = {
                'avg_length': text_stats[0],
                'min_length': text_stats[1],
                'max_length': text_stats[2],
                'median_length': text_stats[3]
            }
            
            return stats
            
        except Exception as e:
            print(f"Stats query error: {e}")
            return {}
    
    def _row_to_result(self, row) -> QueryResult:
        """Convert database row to QueryResult"""
        return QueryResult(
            para_id=row[0],
            paper_id=row[1],
            section_id=row[2],
            page=row[3],
            type=row[4],
            text=row[5],
            score=row[6],
            figure_meta=row[7] if len(row) > 7 else None,
            refs=row[8] if len(row) > 8 else None
        )
    
    def export_search_results(self, 
                            results: List[QueryResult],
                            output_file: Path,
                            format: str = 'json'):
        """Export search results to file"""
        
        if format == 'json':
            data = [
                {
                    'para_id': r.para_id,
                    'paper_id': r.paper_id,
                    'section_id': r.section_id,
                    'page': r.page,
                    'type': r.type,
                    'text': r.text,
                    'score': r.score,
                    'figure_meta': r.figure_meta,
                    'refs': r.refs
                }
                for r in results
            ]
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            data = []
            for r in results:
                data.append({
                    'para_id': r.para_id,
                    'paper_id': r.paper_id,
                    'section_id': r.section_id,
                    'page': r.page,
                    'type': r.type,
                    'text': r.text,
                    'score': r.score,
                    'has_figure': r.figure_meta is not None,
                    'ref_count': len(r.refs) if r.refs else 0
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Demo of DocUnits query capabilities"""
    
    # Initialize query engine
    engine = DocUnitsQueryEngine("docunits.db", Path("./docunits_output"))
    
    try:
        # Get corpus statistics
        print("=== Corpus Statistics ===")
        stats = engine.get_corpus_stats()
        print(json.dumps(stats, indent=2))
        
        # Example searches
        print("\n=== Sample Text Search ===")
        results = engine.search_text("artificial intelligence safety alignment", limit=5)
        for r in results:
            print(f"Score: {r.score:.3f} | {r.paper_id} | {r.section_id}")
            print(f"Text: {r.text[:200]}...")
            print()
        
        # Citation analysis
        print("=== Citation Analysis ===")
        citation_stats = engine.analyze_citation_patterns(limit=10)
        print("Most cited papers:")
        for paper_id, count in citation_stats.get('most_cited', [])[:5]:
            print(f"  {paper_id}: {count} citations")
        
    finally:
        engine.close()


if __name__ == "__main__":
    main()