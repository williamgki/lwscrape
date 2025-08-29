# GChildren Corpus Integration Guide

## Overview
This guide explains how to integrate the GChildren academic corpus with the main LessWrong corpus using the provided tools and data.

## What's Included in the Tarball

### 1. GChildren Contextual Chunks
- **Location**: `gchildren/chunked_corpus/`
- **Files**: 10 parquet batch files (`gchildren_chunks_batch_0000.parquet` to `gchildren_chunks_batch_0009.parquet`)
- **Total chunks**: 9,762 contextual chunks
- **Schema**: Identical 17-column schema to main LW corpus for seamless integration

### 2. Source Documents  
- **Location**: `gchildren/normalized_corpus/`
- **File**: `gchildren_document_store.parquet` (normalized source documents)
- **Location**: `gchildren/pdf_full_collection/pdfs/` 
- **Files**: 572 academic PDF papers from arXiv and other sources
- **Location**: `gchildren/aisi_alignment_crawl/scraped_content/`
- **Files**: AISI government research content in JSONL format

### 3. Integration Tools
- **File**: `corpus_merger.py` - Complete merger script with deduplication
- **File**: `GCHILDREN_INTEGRATION_GUIDE.md` - This guide

### 4. Statistics and Logs
- **File**: `gchildren/chunked_corpus/gchildren_chunking_stats.json` - Processing statistics
- **File**: `gchildren/chunking_progress.log` - Complete processing log

## GChildren Corpus Statistics

- **Documents processed**: 824/828 (99.5% success rate)
- **Total chunks**: 9,762 contextual chunks  
- **Average tokens per chunk**: 1,113 tokens
- **Content breakdown**: 98.4% PDF academic papers, 1.6% HTML content
- **Domains**: Primarily arXiv.org with government/research content
- **Chunking approach**: Wilson Lin's contextual chunking with AI-generated summaries

## Integration Process

### Step 1: Validate Prerequisites
Ensure you have the main LW corpus available with these characteristics:
- **Total chunks**: 66,716 chunks
- **Schema**: 17 columns (chunk_id, doc_id, content, prev_context, section_path, summary_header, token_count, chunk_index, page_refs, content_type, structure_type, heading_level, doc_title, domain, original_url, authors, pub_date)
- **Location**: Expected at `/chunked_corpus/` with files named `chunks_batch_*.parquet`

### Step 2: Run the Corpus Merger
```bash
python corpus_merger.py
```

**What it does:**
1. **Schema validation**: Ensures both corpora have identical column structure
2. **Loading**: Reads all chunks from both main LW and gchildren corpora
3. **Deduplication**: Uses content hashing to detect and remove duplicates
4. **Merging**: Combines corpora with source tagging
5. **Saving**: Outputs unified corpus in batch files + complete file

**Expected output:**
- **Combined corpus**: ~76,500 chunks (14.6% expansion)
- **Academic enhancement**: Massive increase in academic PDF content
- **Deduplication**: <3% overlap expected between corpora
- **Output location**: `/unified_corpus/` directory

### Step 3: Verify Integration
The merger will generate:
- `unified_chunks_batch_*.parquet` - Batched chunk files
- `unified_contextual_chunks_complete.parquet` - Complete corpus file  
- `merger_statistics.json` - Integration statistics
- `unified_corpus_summary.json` - Corpus composition summary

## Schema Compatibility

Both corpora use identical 17-column schema:

**Core Chunk Data:**
- `chunk_id` (string): Unique identifier
- `doc_id` (string): Source document ID
- `content` (string): Chunk text content
- `prev_context` (string): Previous contextual information
- `section_path` (string): Hierarchical document structure
- `summary_header` (string): AI-generated chunk summary

**Metadata:**
- `token_count` (int64): Token count
- `chunk_index` (int64): Index within document
- `page_refs` (string): Page references for PDFs
- `content_type` (string): "pdf" or "html"
- `structure_type` (string): "paragraph" or "heading"
- `heading_level` (float64): Heading level if applicable

**Document Info:**
- `doc_title` (string): Document title
- `domain` (string): Source domain
- `original_url` (string): Original URL
- `authors` (string): Authors
- `pub_date` (string): Publication date

## Key Features of GChildren Corpus

### 1. Academic Focus
- **572 full academic PDFs** with complete text extraction
- **arXiv papers**: Latest AI safety and alignment research
- **Government research**: AISI Alignment Project content
- **High academic density**: 11.8 chunks per document vs 5.8 for main corpus

### 2. Contextual Chunking (Wilson Lin's Approach)
- **Semantic boundaries**: Preserves meaning across chunk boundaries
- **Previous context**: 1-3 sentences from previous chunk
- **AI summaries**: Claude-generated 1-2 line summaries for each chunk
- **Section paths**: Hierarchical document structure preservation
- **Optimal size**: 700-1200 tokens per chunk for retrieval performance

### 3. Quality Assurance
- **Cross-corpus deduplication**: Prevents content overlap
- **Schema validation**: Ensures perfect compatibility
- **Error handling**: Robust processing with fallbacks
- **Comprehensive logging**: Full audit trail of processing

## Retrieval System Integration

After merging, update your retrieval system:

### 1. BM25 Index
- Rebuild BM25 index with unified corpus content
- Update document frequencies and term statistics
- Maintain existing preprocessing pipeline

### 2. Dense Embeddings (BGE-M3)
- Generate embeddings for new gchildren chunks only
- Combine with existing main corpus embeddings
- Update FAISS index with expanded embedding set

### 3. Hybrid Retrieval
- No changes needed to reciprocal rank fusion logic
- System will automatically benefit from expanded corpus
- Test with academic queries to verify gchildren content retrieval

## Expected Performance Improvements

- **Academic queries**: Significantly improved retrieval for AI safety research
- **Government content**: Enhanced access to official alignment research
- **PDF content**: 4.5x increase in academic paper availability
- **Research depth**: Access to full paper content vs abstract-only
- **Citation networks**: Rich interconnected academic content

## Troubleshooting

### Schema Mismatch
If schema validation fails:
1. Compare column names and types between corpora
2. Check for encoding differences
3. Verify data type consistency (especially int64 vs int32)

### Memory Issues
For large-scale processing:
1. Process in smaller batches
2. Use chunked reading/writing
3. Monitor memory usage during merger

### Deduplication Issues
If unexpected duplicate rates:
1. Check content hashing function
2. Verify normalization consistency
3. Review chunk size thresholds

## Cost Information

**GChildren Processing Cost**: ~$28.42
- Input tokens: 30.1M tokens @ $0.80/MTok = $24.11
- Output tokens: 1.1M tokens @ $4.00/MTok = $4.31
- Model: Claude Haiku 3.5 for summary generation

## Support

For integration issues:
1. Check logs in `corpus_merger.log`
2. Verify file paths and permissions
3. Confirm Python dependencies (pandas, numpy, pyarrow)
4. Test on small sample before full corpus merger

---

**Generated**: August 18, 2025  
**GChildren Corpus Version**: 1.0  
**Total Academic Expansion**: 9,762 contextual chunks  
**Integration Ready**: ✅