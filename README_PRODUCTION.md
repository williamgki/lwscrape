# Academic AI Safety Corpus Builder

A comprehensive production system for building, indexing, and querying academic corpora focused on AI safety, alignment, and interpretability research.

## 🎯 System Overview

This repository contains a complete pipeline for:

1. **Multi-source paper discovery** (OpenAlex, arXiv, OpenReview)
2. **Structured PDF parsing** (GROBID + pdffigures2)
3. **Schema-first typed extraction** (claims, mechanisms, experiments)
4. **Production chunking** (512-token semantic windows)
5. **Hybrid retrieval** (ColBERT + BM25 + SPLADE → RRF → BGE reranker)
6. **Citation graph analysis** (OpenAlex ego-graphs)
7. **Evidence linking** (typed objects → retrieval corpus)

## 🚀 Quick Start

### Prerequisites
```bash
# Required services
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

# Python dependencies
pip install -r requirements_multisource.txt
```

### Production Pipeline (7 Steps)
```bash
# 1. Multi-source discovery
python multi_source_corpus_builder.py --openalex-pages 50 --arxiv-results 5000

# 2. PDF processing
python structured_pdf_parser.py --batch-process --pdf-dir ./pdfs

# 3. Typed extraction
python deterministic_paper_pipeline.py --batch-size 10

# 4. Evidence linking
python typed_object_linking.py --evidence-search

# 5. Citation graphs
python citation_graph_builder.py --build-ego-graphs

# 6. Production chunking
python production_chunking_config.py --chunk-all-papers

# 7. Index building
python fusion_retrieval_pipeline.py --build-indexes
```

## 📋 Core Components

### A. Multi-Source Discovery
- **OpenAlex**: `/works?search=alignment%20OR%20interpretability&per_page=200`
- **arXiv**: `(cs.AI OR cs.LG) AND (alignment OR interpretability OR RLHF)`
- **OpenReview**: ICLR, NeurIPS, ICML 2022-2025

### B. Structured Processing
- **GROBID**: TEI XML extraction with section hierarchy
- **pdffigures2**: Figure/table extraction with captions and bounding boxes
- **Smart deduplication**: DOI → arXiv ID → MinHash similarity

### C. Schema-First Extraction
```json
{
  "claims": [{"id":"C1", "text":"", "type":"empirical", "evidence_spans":[...]}],
  "assumptions": [{"id":"A1", "text":"", "evidence_spans":[...]}],
  "mechanisms": [{"id":"M1", "text":"", "evidence_spans":[...]}],
  "metrics": [{"id":"X1", "name":"", "definition":"", "evidence_spans":[...]}],
  "experiments": [{"id":"E1", "setup":"", "dataset":"", "evidence_spans":[...]}],
  "threat_models": [{"id":"T1", "text":"", "evidence_spans":[...]}]
}
```

### D. Production Chunking
- **512 tokens** with **256 token stride** within sections
- **Header path prepending**: "Title > Section > Subsection"
- **Caption attachment**: Automatic figure citation → caption linking
- **Reference preservation**: `[@Author2023]`, `[1]` inline markers

### E. Hybrid Retrieval System
1. **ColBERT (k=400)**: Late-interaction dense retrieval
2. **BM25 (k=400)**: Sparse lexical matching
3. **SPLADE (k=400)**: Learned sparse representations
4. **RRF Fusion (k=300)**: `score = Σ 1/(60 + rank_i)`
5. **BGE Reranker (k=50)**: BAAI/bge-reranker-v2-m3 precision

## 🗃️ Data Models

### DocUnits (Parquet)
```sql
SELECT chunk_id, paper_id, page_from, page_to, text, headers, 
       figure_ids, colbert_vectors_path, bm25_terms, length_tokens
FROM read_parquet('DocUnits/*.parquet')
WHERE paper_id LIKE '%alignment%';
```

### Citation Graph (NetworkX → JSON)
- **Nodes**: paper metadata (year, venue, organization)
- **Edges**: citation relationships with temporal analysis
- **Ego-graphs**: 1-2 hop expansion around target papers

### Evidence Ledger (JSON)
```json
{
  "claim_id": "C1",
  "intent_type": "supporting|contradictory|cross_domain",
  "search_results": [...],
  "retrieval_scores": {"avg_reranker_score": 0.89}
}
```

## 🛡️ Quality Guardrails

- **Deduplication**: SimHash/MinHash with 0.85 threshold
- **Evidence validation**: Page cross-checking, quote verification
- **Graph sanity**: Survey papers require ≥20 references
- **Reproducibility**: Full API response logging with timestamps

## 📁 Key Files

### Core Pipeline
- `multi_source_corpus_builder.py` - Multi-API paper discovery
- `structured_pdf_parser.py` - GROBID + pdffigures2 integration
- `typed_paper_extractor.py` - Schema-first extraction with validation
- `production_chunking_config.py` - 512-token semantic chunking
- `fusion_retrieval_pipeline.py` - Complete retrieval system

### Advanced Features
- `citation_graph_builder.py` - OpenAlex ego-graph construction
- `typed_object_linking.py` - Evidence search for extracted claims
- `bge_cross_encoder_reranker.py` - BGE precision reranking
- `colbert_indexer.py` - Late-interaction dense indexing

### Configuration
- `production_scrape_config.py` - All API endpoints and parameters
- `requirements_multisource.txt` - Production dependencies

## 🔧 Development Setup

### Docker Services
```bash
# GROBID PDF processing
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

# DuckDB for analytics
docker run --rm -v $(pwd):/data duckdb/duckdb
```

### Build Tools
```bash
# pdffigures2 (requires SBT)
cd pdffigures2 && sbt assembly

# ColBERT indexing
python colbert_indexer.py --collection chunks.tsv --index production_papers
```

## 📊 Expected Outputs

- **Papers discovered**: ~10,000+ from multi-source APIs
- **DocUnits created**: ~500K chunks (512 tokens each)
- **Typed objects**: ~50K claims, mechanisms, experiments
- **Citation graph**: ~100K nodes, ~500K edges
- **Retrieval latency**: <2s for hybrid search + reranking

## 🎛️ Configuration Examples

### OpenAlex Search
```python
"/works?search=alignment%20OR%20interpretability&select=id,title,doi,abstract_inverted_index,primary_location&per_page=200"
```

### ColBERT + BGE Pipeline
```python
# Retrieve candidates
colbert_results = retriever.search(query, k=400)
bm25_results = bm25.search(query, k=400)

# RRF fusion
fused = rrf_fusion([colbert_results, bm25_results], k=60)

# BGE reranking
final = bge_reranker.rerank(query, fused[:200], return_top_k=50)
```

## 📈 Monitoring & Analytics

- **API rate limits**: Tracked per endpoint
- **Processing metrics**: Papers/hour, extraction success rates
- **Quality metrics**: Evidence density, validation pass rates
- **Storage usage**: PDF cache, Parquet files, index sizes

## 🔗 References

- [OpenAlex API](https://docs.openalex.org/)
- [ColBERTv2 Paper](https://aclanthology.org/2022.naacl-main.272/)
- [BGE Reranker Model](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [GROBID Documentation](https://grobid.readthedocs.io/)
- [RRF Paper](https://cormack.uwaterloo.ca/)

---

**Status**: Production-ready system for comprehensive academic corpus building and analysis.