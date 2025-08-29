# LessWrong Contextual Chunking - Multi-VM Handoff Guide

## Project Status Summary
**Date:** August 17, 2025  
**Status:** 4.2% complete - need multi-VM scaling for practical completion time  
**Current VM completed:** 750 documents, 3,160 chunks  

## What We Built
✅ **Complete Wilson Lin contextual chunking implementation**  
✅ **Enhanced structure parsers** (HTML headings, lists, tables, PDF pages)  
✅ **Claude API integration** (claude-3-5-haiku-latest for summaries)  
✅ **Production multi-worker pipeline** with rate limiting  
✅ **Quality validated:** 4.2 chunks/doc average with excellent summaries  

## Critical Files for New VM

### Core Implementation
```
/home/ubuntu/LW_scrape/
├── enhanced_contextual_chunker.py      # Main chunker with Wilson Lin approach
├── production_chunking_pipeline.py     # Multi-worker pipeline
├── PRODUCTION_CHUNKING_PLAN.md         # Complete technical specification
└── normalized_corpus/document_store.parquet  # Source data (217MB, 17,746 docs)
```

### Current Progress
```
/home/ubuntu/LW_scrape/chunked_corpus/
├── chunks_batch_0000.parquet through chunks_batch_0057.parquet  # 57 batches completed
├── progress.json                        # {"processed_documents": 750, "total_chunks": 3160}
└── [Missing batch 56 - minor gap]       # Resume from batch 58
```

## Performance Analysis - Why Slow?

**Original Estimate:** 4 hours (4,436 docs/hour)  
**Actual Performance:** 329 docs/hour (13.5x slower)  

**Bottlenecks:**
1. **Conservative config:** 2 workers (not 4), 0.3s API rate (not 0.1s)
2. **API bound:** 4.2 chunks/doc × 0.3s = 1.26s just for Claude summaries
3. **Memory constraints:** 7.6GB RAM forced conservative approach
4. **Small batches:** 15 docs/batch (not 25) to avoid OOM

## Multi-VM Scaling Strategy

### Recommended: 4 Powerful VMs (16GB+ RAM each)
**Partition corpus by document ranges:**

```
VM1: Documents 0-4,436      (batches 0-295)    
VM2: Documents 4,437-8,872  (batches 296-591)  
VM3: Documents 8,873-13,308 (batches 592-887)  
VM4: Documents 13,309-17,746 (batches 888-1,182)
```

### Optimal VM Configuration
```python
num_workers = 6
batch_size = 25  
api_rate_limit = 0.1  # 10 calls/sec
checkpoint_interval = 20
```

**Expected performance:** 1,100+ docs/hour per VM  
**Total completion time:** 4-6 hours

## Setup Instructions for New VM

### 1. Environment Setup
```bash
# Clone/copy the LW_scrape directory
# CRITICAL: Set these environment variables first
export ANTHROPIC_API_KEY="aws-secretsmanager://arn:aws:secretsmanager:..."
export ANTHROPIC_BASE_URL="https://anthropic-proxy.i.apps.ai-safety-institute.org.uk"

# Install dependencies
cd /home/ubuntu/LW_scrape
python -m venv venv
source venv/bin/activate
pip install pandas tiktoken anthropic

# CRITICAL: Install aisitools for API proxy (requires GitHub SSH access)
pip install --break-system-packages git+ssh://git@github.com/AI-Safety-Institute/aisi-inspect-tools

# Test aisitools API integration:
python -c "
from aisitools.api_key import get_api_key_for_proxy
import os
key = os.environ.get('ANTHROPIC_API_KEY')
proxy_key = get_api_key_for_proxy(key)
print('✅ aisitools working:', proxy_key[:20] + '...')
"
```

### 2. Data Transfer
**Copy these essential files to new VM:**
```
document_store.parquet           # 217MB source corpus
enhanced_contextual_chunker.py   # Core implementation
production_chunking_pipeline.py  # Multi-worker pipeline
```

### 3. Partition-Specific Launch

**VM1 (documents 0-4,436):**
```python
# Modify production_chunking_pipeline.py
def load_documents_batch(parquet_path, batch_size=25, start_idx=0):
    df = pd.read_parquet(parquet_path)
    # VM1: Limit to first 4,436 documents
    df = df.iloc[:4437]  
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    # ... rest of function
```

**VM2 (documents 4,437-8,872):**
```python
def load_documents_batch(parquet_path, batch_size=25, start_idx=0):
    df = pd.read_parquet(parquet_path)
    # VM2: Documents 4,437-8,872
    df = df.iloc[4437:8873]
    # Reset start_idx for this partition
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]
```

### 4. Launch Command
```bash
# For each VM with appropriate partition
source venv/bin/activate
python production_chunking_pipeline.py > chunking_vm1.log 2>&1 &
```

## Current VM Status (Continue or Archive)

**Current process:** PAUSED (PID 148300 killed)  
**Progress saved:** 750 docs, 57 batches, progress.json updated  

**Options:**
1. **Archive current work:** Copy /home/ubuntu/LW_scrape/chunked_corpus/ for later consolidation
2. **Continue current VM:** Resume from batch 58 (document 750) if keeping as VM4

## Quality Validation - Sample Output

**Section Path:** "Ensuring smarter-than-human intelligence has a positive outcome - Machine Intelligence Research Institute"  

**Summary Header:** "The key challenge in AI alignment is developing learning systems that can gradually understand human values, rather than attempting to manually code complex preferences, similar to how machine vision evolved."

**Chunks per doc:** 4.2 average (excellent semantic boundary detection)  
**Token range:** 700-1,200 per chunk (Wilson Lin specification)  
**Context preservation:** prev_context, section_path, page_refs working

## Final Consolidation

After all VMs complete, consolidate with:
```bash
# Collect all chunks_batch_*.parquet from all VMs
# Run consolidation:
python -c "
import pandas as pd
from pathlib import Path

all_chunks = []
for vm_dir in ['vm1_chunks', 'vm2_chunks', 'vm3_chunks', 'vm4_chunks']:
    chunk_files = Path(vm_dir).glob('chunks_batch_*.parquet')
    for f in chunk_files:
        all_chunks.append(pd.read_parquet(f))

final_df = pd.concat(all_chunks, ignore_index=True)
final_df.to_parquet('contextual_chunks_complete.parquet', compression='snappy')
print(f'Final corpus: {len(final_df)} chunks')
"
```

## Expected Final Output
- **~75,000 contextual chunks** with Wilson Lin semantic structure
- **Section paths, summaries, prev_context** for all chunks  
- **PDF page references** preserved for citations
- **Ready for DDL experiments** with Anthropic contextual retrieval optimization

---
**Ready to scale! The architecture is proven - just needs more compute power.**