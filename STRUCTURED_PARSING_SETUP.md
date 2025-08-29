# Structured PDF Parsing Setup Guide

## 🎯 Overview
This guide sets up GROBID and pdffigures2 for extracting structured content from academic PDFs, enabling enhanced Wilson Lin chunking with document structure awareness.

## 📋 Prerequisites

### Docker (for GROBID)
```bash
# Install Docker if not available
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### Java & SBT (for pdffigures2)
```bash
# Install Java
sudo apt update
sudo apt install -y openjdk-8-jdk

# Install SBT
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
sudo apt update
sudo apt install -y sbt
```

## 🚀 Service Setup

### 1. GROBID Service
```bash
# Pull and run GROBID Docker container
docker pull grobid/grobid:latest
docker run --rm --init -d -p 8070:8070 --name grobid-service grobid/grobid:latest

# Test service
curl http://localhost:8070/api/isalive
```

### 2. pdffigures2 Setup
```bash
# Clone and build pdffigures2
git clone https://github.com/allenai/pdffigures2
cd pdffigures2
sbt assembly

# Test build
ls target/scala-2.12/pdffigures2-assembly.jar
```

## 🔄 Processing Pipeline

### Step 1: Multi-Source Paper Collection
```bash
# Collect papers from APIs
python multi_source_corpus_builder.py --openalex-pages 5 --arxiv-results 500

# Deduplicate
python corpus_deduplicator.py

# Download PDFs
python multisource_to_wilson_lin.py --skip-chunking
```

### Step 2: Structured Parsing
```bash
# Process PDFs with GROBID + pdffigures2
python structured_pdf_parser.py

# Expected output:
# - grobid_tei/*.xml (structured TEI XML)
# - pdffigures2_json/*.json (figures/tables/captions)
# - structured_papers.parquet (combined results)
```

### Step 3: Structure-Aware Wilson Lin Chunking
```bash
# Apply enhanced chunking with structure awareness
python structure_aware_wilson_lin.py \
  --structured-papers /path/to/structured_papers.parquet

# Expected output:
# - structure_aware_wilson_lin_chunks.parquet
# - Enhanced chunks with section hierarchy, citations, figures
```

## 📊 Expected Benefits

### Enhanced Chunk Quality
- **Section Hierarchy**: Proper academic structure (Introduction › Methods › Results)
- **Citation Context**: Chunks maintain citation relationships
- **Figure References**: Chunks linked to relevant figures/tables
- **Section Classification**: Methodology, results, conclusions properly categorized

### Improved Retrieval
- **Structural Filtering**: Query specific section types
- **Citation Following**: Navigate related papers via references
- **Figure Context**: Access visual content context
- **Academic Flow**: Preserve logical document progression

## 🧪 Test Run (Simplified)

If Docker/SBT unavailable, you can test with fallback text extraction:

```bash
# Test with existing PDF text extraction
python structure_aware_wilson_lin.py \
  --structured-papers fallback_text_papers.json \
  --min-tokens 700 --max-tokens 1200
```

## 📈 Integration with Existing System

The structured chunks integrate seamlessly with your existing AI relevance scoring:

```python
# Load structure-aware chunks
structured_chunks = pd.read_parquet('structure_aware_wilson_lin_chunks.parquet')

# Apply AI relevance scoring
scored_chunks = corpus_relevance_scorer.score_corpus(structured_chunks)

# Enhanced retrieval with structure filters
high_relevance_methods = scored_chunks[
    (scored_chunks['ai_relevance_score'] > 0.7) & 
    (scored_chunks['section_type'] == 'methodology')
]
```

## 🎯 Production Workflow

1. **Collect** → Multi-source academic papers via APIs
2. **Parse** → GROBID + pdffigures2 structure extraction  
3. **Chunk** → Structure-aware Wilson Lin with 700-1200 tokens
4. **Enhance** → AI summaries + relevance scoring
5. **Index** → Hybrid BM25 + dense retrieval with structure filters

This creates a next-generation academic corpus with both content relevance AND structural intelligence for precise information retrieval.