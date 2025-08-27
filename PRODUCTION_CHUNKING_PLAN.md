# Production Contextual Chunking Plan
## Implementing Wilson Lin's Approach at Scale

### Overview
Process 17,746 documents (217MB) into contextual chunks following Wilson Lin's semantic preservation principles. Target: ~50K-70K chunks with rich contextual metadata.

### Architecture Requirements

#### 1. Structure-First Segmentation
**HTML Documents (16,662 docs):**
- Parse `<h1>-<h6>` headings with hierarchical structure
- Extract `<li>` list items as semantic boundaries
- Identify `<table>` captions and structured content
- Detect `<section>`, `<article>`, `<div>` with semantic classes

**PDF Documents (1,084 docs):**
- Use page boundaries as primary structure (page_start, page_end)
- Detect heading patterns: ALL CAPS, numbered sections, Title: format
- Identify paragraph breaks and section transitions
- Preserve citation-friendly page references

#### 2. Token Boundary Management
- **Target Range:** 700-1,200 tokens per chunk
- **Subdivision Strategy:** Split long sections at paragraph boundaries
- **Minimum Threshold:** 400 tokens (avoid tiny chunks)
- **Token Counter:** tiktoken cl100k_base encoding

#### 3. Contextual Enrichment (Wilson Lin Requirements)

**A. Previous Context Window:**
- Extract 1-3 sentences from previous chunk
- Handle document boundaries gracefully
- Preserve pronouns and abbreviation context

**B. Section Path Generation:**
```
Format: "Document Title › Section › Subsection › Topic"
Examples:
- "Alignment › Mesa-Optimization › Threat Models"
- "Goodhart's Law › Applications › Metrics Gaming"
- "Page 15 › Abstract › Conclusion" (PDFs)
```

**C. Summarization Headers:**
- Use Claude-3-5-haiku-latest for 1-2 line summaries
- Localized TL;DR capturing chunk's main point
- Rate limit: 10 requests/second to avoid throttling
- Fallback: First sentence truncated to 200 chars

#### 4. Chunk Store Schema
```python
{
    'chunk_id': 'doc_hash_chunk_001',
    'doc_id': 'original_document_hash',
    'content': 'main_chunk_text',
    'prev_context': '1-3 sentences from previous chunk',
    'section_path': 'hierarchical path string',
    'summary_header': 'Claude-generated 1-2 line summary',
    'token_count': 850,
    'chunk_index': 0,
    'page_refs': 'pp. 15-16' (PDFs only),
    'structure_type': 'heading|paragraph|list|table',
    'heading_level': 2 (if applicable),
    'metadata': {
        'doc_title': 'source document title',
        'domain': 'arxiv.org',
        'content_type': 'pdf|html',
        'original_url': 'source URL',
        'authors': 'document authors',
        'pub_date': 'publication date'
    }
}
```

### Production Pipeline

#### Stage 1: Enhanced Structure Parsing
1. **HTML Parser Improvements:**
   - Detect nested heading hierarchies
   - Extract list structures (`<ul>`, `<ol>`, `<li>`)
   - Parse table captions and headers
   - Handle markdown-style content

2. **PDF Parser Enhancements:**
   - Page-aware chunking with boundaries
   - Improved heading detection patterns
   - Table of contents extraction
   - Reference section handling

#### Stage 2: Robust Chunking Logic
1. **Semantic Boundary Detection:**
   - Structure-first approach (headings > paragraphs > sentences)
   - Respect token boundaries within semantic units
   - Handle edge cases (very long/short sections)

2. **Context Preservation:**
   - Build hierarchical section paths
   - Maintain previous chunk context
   - Handle document transitions

#### Stage 3: Multi-Worker Processing
1. **Worker Architecture:**
   - 4-6 parallel workers for chunking
   - Separate API worker pool for summarization
   - Rate limiting: 10 Claude API calls/second
   - Progress tracking with parquet incremental saves

2. **Error Handling:**
   - API timeout fallbacks
   - Malformed content recovery
   - Progress resumption capability

#### Stage 4: Quality Assurance
1. **Validation Checks:**
   - Token count accuracy
   - Context preservation verification
   - Summary quality sampling
   - Chunk coherence testing

### Implementation Phases

#### Phase 1: Enhanced Parsers (30 mins)
- Upgrade HTML structure parser
- Improve PDF boundary detection
- Add table/list extraction

#### Phase 2: Section Path Algorithm (20 mins)
- Hierarchical path building
- PDF page reference generation
- Path normalization and truncation

#### Phase 3: Production Pipeline (45 mins)
- Multi-worker chunking system
- API rate limiting and batch processing
- Incremental parquet output

#### Phase 4: Full Corpus Processing (3-4 hours)
- Process all 17,746 documents
- Generate ~50K-70K contextual chunks
- Quality validation and stats

### Expected Outcomes

**Chunk Statistics:**
- ~50,000-70,000 total chunks
- Average 800-900 tokens per chunk
- 95%+ chunks with meaningful section paths
- 90%+ chunks with quality summaries

**Storage:**
- Chunk store: ~300-400MB parquet
- Rich metadata for DDL experiments
- Citation-ready PDF page references
- Contextual retrieval optimization

**Quality Metrics:**
- Context preservation score
- Summary coherence rating
- Retrieval performance benchmarks
- DDL-ready semantic structure

This plan implements Wilson Lin's complete specification with production-ready scalability and error handling.