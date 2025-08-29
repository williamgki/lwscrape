#!/usr/bin/env python3

"""
Production Scrape Configuration
Copy-paste configs & commands for the big corpus scrape
"""

import json
from pathlib import Path

# =============================================================================
# OPENALEX API CONFIGURATIONS
# =============================================================================

OPENALEX_CONFIGS = {
    "base_url": "https://api.openalex.org",
    
    # Sample filters for AI alignment/safety corpus
    "search_queries": {
        "broad_alignment": "/works?search=alignment%20OR%20interpretability%20OR%20\"reward%20modeling\"&per_page=200",
        
        "deceptive_alignment": "/works?filter=from_publication_date:2016-01-01,to_publication_date:2025-12-31&search=\"deceptive%20alignment\"",
        
        "ai_safety_concepts": "/works?search=\"AI%20safety\"%20OR%20\"artificial%20intelligence%20safety\"%20OR%20\"AI%20alignment\"&per_page=200",
        
        "interpretability": "/works?search=interpretability%20OR%20explainability%20OR%20\"mechanistic%20interpretability\"&per_page=200",
        
        "rlhf_concepts": "/works?search=RLHF%20OR%20\"reinforcement%20learning%20human%20feedback\"%20OR%20\"reward%20modeling\"&per_page=200"
    },
    
    # Optimized field selection to reduce payload
    "select_fields": "&select=id,title,doi,abstract_inverted_index,primary_location,open_access,referenced_works,authorships,publication_year,cited_by_count,concepts",
    
    # Date filters
    "date_filters": {
        "recent": "&filter=from_publication_date:2020-01-01,to_publication_date:2025-12-31",
        "modern_era": "&filter=from_publication_date:2016-01-01,to_publication_date:2025-12-31",
        "comprehensive": "&filter=from_publication_date:2010-01-01,to_publication_date:2025-12-31"
    },
    
    "rate_limit_delay": 0.1,
    "max_retries": 3,
    "per_page": 200
}

# =============================================================================
# ARXIV CONFIGURATION
# =============================================================================

ARXIV_CONFIG = {
    "search_query": '(cs.AI OR cs.LG) AND (alignment OR interpretability OR "reward modeling" OR RLHF OR RLAIF OR ELK OR debate)',
    "max_results": 5000,
    "sort_by": "arxiv.SortCriterion.SubmittedDate",
    "batch_size": 100,
    
    # Code template
    "search_template": '''
import arxiv
search = arxiv.Search(
    query='(cs.AI OR cs.LG) AND (alignment OR interpretability OR "reward modeling" OR RLHF OR RLAIF OR ELK OR debate)',
    max_results=1000,
    sort_by=arxiv.SortCriterion.SubmittedDate
)
for result in search.results():
    print(f"Title: {result.title}")
    print(f"arXiv ID: {result.entry_id}")
    print(f"PDF URL: {result.pdf_url}")
    '''
}

# =============================================================================
# OPENREVIEW V2 CONFIGURATION  
# =============================================================================

OPENREVIEW_CONFIG = {
    "base_url": "https://api2.openreview.net",
    "venues": [
        "ICLR 2025", "ICLR 2024", "ICLR 2023",
        "NeurIPS 2024", "NeurIPS 2023", "NeurIPS 2022",
        "ICML 2024", "ICML 2023", "ICML 2022"
    ],
    
    # Code template
    "client_template": '''
import openreview
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username=username,
    password=password
)
recs = client.get_notes(
    content={'venue': 'ICLR 2025'}, 
    details='replies'
)
    ''',
    
    "search_keywords": [
        "alignment", "interpretability", "reward modeling", 
        "RLHF", "AI safety", "mechanistic interpretability",
        "oversight", "deceptive alignment", "ELK"
    ]
}

# =============================================================================
# GROBID SERVICE CONFIGURATION
# =============================================================================

GROBID_CONFIG = {
    "docker_service": {
        "image": "lfoppiano/grobid:0.8.0",
        "port": "8070",
        "start_command": "docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0"
    },
    
    "api_endpoints": {
        "base_url": "http://localhost:8070/api",
        "process_fulltext": "/processFulltextDocument",
        "process_header": "/processHeaderDocument",
        "process_citations": "/processCitations"
    },
    
    "processing_options": {
        "consolidate_header": "1",
        "consolidate_citations": "1", 
        "coordinates": "true",
        "format": "tei"
    },
    
    "curl_example": '''
curl -v --form input=@paper.pdf \
     --form consolidateHeader=1 \
     --form consolidateCitations=1 \
     --form coordinates=true \
     http://localhost:8070/api/processFulltextDocument
    '''
}

# =============================================================================
# PDFFIGURES2 CONFIGURATION
# =============================================================================

PDFFIGURES2_CONFIG = {
    "installation": {
        "repo": "https://github.com/allenai/pdffigures2",
        "build_command": "sbt assembly",
        "jar_location": "target/scala-*/pdffigures2-assembly-*.jar"
    },
    
    "cli_options": {
        "basic": "java -jar pdffigures2-assembly.jar input.pdf",
        "with_text": "java -jar pdffigures2-assembly.jar -g input.pdf",
        "output_format": "java -jar pdffigures2-assembly.jar -m /output/dir -d /data/dir input.pdf"
    },
    
    "processing_flags": {
        "-g": "adds section titles and text dump partitioned by headings",
        "-m": "specifies output directory for figure images", 
        "-d": "specifies data directory for JSON metadata",
        "-q": "quiet mode",
        "-t": "specify timeout in seconds"
    }
}

# =============================================================================
# PARQUET/DUCKDB CONFIGURATION
# =============================================================================

PARQUET_DUCKDB_CONFIG = {
    "write_parquet": '''
# Write from pandas/pyarrow
df.to_parquet('DocUnits/chunks.parquet')

# Write from DuckDB
COPY table_name TO 'output.parquet' (FORMAT PARQUET);
    ''',
    
    "read_queries": {
        "basic_read": "SELECT * FROM read_parquet('DocUnits/*.parquet');",
        "filtered_read": "SELECT * FROM read_parquet('DocUnits/*.parquet') WHERE paper_id LIKE '%alignment%';",
        "aggregated": "SELECT paper_id, COUNT(*) as chunk_count FROM read_parquet('DocUnits/*.parquet') GROUP BY paper_id;",
        "join_example": """
        SELECT p.title, c.text 
        FROM read_parquet('papers.parquet') p 
        JOIN read_parquet('chunks.parquet') c ON p.paper_id = c.paper_id;
        """
    },
    
    "performance_tips": [
        "Filters push down automatically",
        "Use columnar selection: SELECT specific_cols",  
        "Partition by paper_id for better query performance",
        "Use ZSTD compression for better storage efficiency"
    ]
}

# =============================================================================
# COLBERTV2 + BGE RERANKER CONFIGURATION
# =============================================================================

COLBERT_BGE_CONFIG = {
    "colbert_model": {
        "name": "colbert-ir/colbertv2.0",
        "hf_link": "https://huggingface.co/colbert-ir/colbertv2.0",
        "paper": "https://aclanthology.org/2022.naacl-main.272/",
        "usage": '''
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

with Run().context(RunConfig(nranks=1)):
    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0")
    indexer.index(name="academic_papers", collection="chunks.tsv")
        '''
    },
    
    "bge_reranker": {
        "name": "BAAI/bge-reranker-v2-m3",
        "hf_link": "https://huggingface.co/BAAI/bge-reranker-v2-m3", 
        "features": ["long-input support (8192 tokens)", "multilingual", "strong on science text"],
        "usage": '''
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=8192)
scores = reranker.predict([(query, doc) for doc in candidates])
        '''
    }
}

# =============================================================================
# BM25/SPLADE CONFIGURATION  
# =============================================================================

BM25_SPLADE_CONFIG = {
    "pyserini_bm25": {
        "repo": "https://github.com/castorini/pyserini",
        "docs": "https://pyserini.readthedocs.io/",
        "installation": "pip install pyserini",
        "usage": '''
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('path/to/index')
hits = searcher.search('query', k=100)
        '''
    },
    
    "splade": {
        "paper": "https://arxiv.org/abs/2109.10086",
        "model": "naver/splade-cocondenser-ensembledistil",
        "docs": "https://github.com/naver/splade",
        "key_concept": "Sparse lexical retrieval with learned sparse representations"
    }
}

# =============================================================================
# RANK FUSION CONFIGURATION
# =============================================================================

RANK_FUSION_CONFIG = {
    "rrf_formula": "score = Σ 1/(k + rank_i) over systems",
    "recommended_k": 60,
    "reference": "https://cormack.uwaterloo.ca/",
    
    "implementation": '''
def reciprocal_rank_fusion(rankings_list, k=60):
    """Combine multiple rankings using RRF"""
    scores = {}
    for rankings in rankings_list:
        for rank, item_id in enumerate(rankings, 1):
            if item_id not in scores:
                scores[item_id] = 0
            scores[item_id] += 1.0 / (k + rank)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    '''
}

# =============================================================================
# QUALITY & OPS GUARDRAILS
# =============================================================================

QUALITY_GUARDRAILS = {
    "deduplication": {
        "method": "SimHash/MinHash on title+abstract+intro",
        "libraries": ["simhash (PyPI)", "datasketch for MinHash"],
        "threshold": "Merge papers with >0.85 similarity",
        "priority": "arXiv vs journal - prefer journal version"
    },
    
    "graph_sanity": {
        "rule": "Guarantee at least N outgoing refs for survey/tutorial papers", 
        "threshold": "Survey papers must have >=20 references",
        "action": "Mark as low-info if below threshold"
    },
    
    "evidence_density": {
        "metric": "(#quotes with page anchors) / (#claims)",
        "failing_threshold": "<0.8 for paper reviews",
        "action": "Flag papers with poor evidence extraction"
    },
    
    "reproducibility": {
        "requirement": "Save every API response in raw_metadata",
        "include": ["ETags", "timestamps", "API version", "request parameters"],
        "storage": "One JSON file per paper with all API responses"
    },
    
    "monitoring": {
        "api_rate_limits": "Track requests/minute for each API",
        "error_tracking": "Log all API failures with retry attempts", 
        "progress_metrics": "Papers processed, extraction success rate, validation pass rate",
        "disk_usage": "Monitor PDF storage, Parquet files, index sizes"
    }
}

# =============================================================================
# PRODUCTION PIPELINE COMMANDS
# =============================================================================

PRODUCTION_COMMANDS = {
    "docker_services": {
        "grobid": "docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0",
        "duckdb": "docker run --rm -v $(pwd):/data duckdb/duckdb",
    },
    
    "build_commands": {
        "pdffigures2": "cd pdffigures2 && sbt assembly",
        "colbert_index": "python colbert_indexer.py --collection chunks.tsv --index-name production_papers",
        "parquet_export": "python -c \"import duckdb; duckdb.execute('COPY docunits TO chunks.parquet (FORMAT PARQUET)')\""
    },
    
    "scrape_sequence": [
        "1. python multi_source_corpus_builder.py --openalex-pages 50 --arxiv-results 5000",
        "2. python structured_pdf_parser.py --batch-process --pdf-dir ./pdfs",
        "3. python deterministic_paper_pipeline.py --batch-size 10",
        "4. python typed_object_linking.py --evidence-search",
        "5. python citation_graph_builder.py --build-ego-graphs",
        "6. python production_chunking_config.py --chunk-all-papers",
        "7. python fusion_retrieval_pipeline.py --build-indexes"
    ]
}

# =============================================================================
# SAVE CONFIGURATION TO FILE
# =============================================================================

def save_production_config():
    """Save all configurations to JSON file"""
    
    config = {
        "openalex": OPENALEX_CONFIGS,
        "arxiv": ARXIV_CONFIG,
        "openreview": OPENREVIEW_CONFIG,
        "grobid": GROBID_CONFIG,
        "pdffigures2": PDFFIGURES2_CONFIG,
        "parquet_duckdb": PARQUET_DUCKDB_CONFIG,
        "colbert_bge": COLBERT_BGE_CONFIG,
        "bm25_splade": BM25_SPLADE_CONFIG,
        "rank_fusion": RANK_FUSION_CONFIG,
        "quality_guardrails": QUALITY_GUARDRAILS,
        "production_commands": PRODUCTION_COMMANDS
    }
    
    config_file = Path("production_scrape_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Production configuration saved to: {config_file}")
    print(f"📁 Total configuration sections: {len(config)}")
    
    # Print quick start commands
    print("\n🚀 Quick Start Commands:")
    for i, command in enumerate(PRODUCTION_COMMANDS["scrape_sequence"], 1):
        print(f"   {command}")


if __name__ == "__main__":
    save_production_config()
    
    print("\n" + "="*80)
    print("🎯 PRODUCTION SCRAPE CONFIGURATION READY")
    print("="*80)
    
    print("\n📊 API Endpoints Configured:")
    print(f"   • OpenAlex: {len(OPENALEX_CONFIGS['search_queries'])} search queries")
    print(f"   • arXiv: {ARXIV_CONFIG['max_results']} max results")
    print(f"   • OpenReview: {len(OPENREVIEW_CONFIG['venues'])} venues")
    
    print("\n🔧 Processing Tools Ready:")
    print("   • GROBID: Docker service + API endpoints") 
    print("   • pdffigures2: CLI with -g text extraction")
    print("   • ColBERTv2: Late-interaction indexing")
    print("   • BGE Reranker: v2-m3 long-input model")
    
    print("\n🛡️ Quality Guardrails:")
    print("   • SimHash/MinHash deduplication")
    print("   • Evidence density thresholds") 
    print("   • Graph sanity checks")
    print("   • Full API response logging")
    
    print("\n🗄️ Storage Format:")
    print("   • DocUnits: Parquet with DuckDB queries")
    print("   • Citations: Graph with OpenAlex metadata")
    print("   • Evidence: JSON ledgers per paper")
    print("   • Indexes: ColBERT + BM25/SPLADE")
    
    print(f"\n✨ Ready for big corpus scrape!")