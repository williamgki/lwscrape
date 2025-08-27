#!/usr/bin/env python3
"""
Quick test of contextual chunking using raw JSONL files
"""

import json
import os
from contextual_chunker import ContextualChunker, load_sample_documents

def load_sample_from_jsonl(jsonl_path: str, sample_size: int = 3):
    """Load sample documents directly from JSONL"""
    documents = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                doc = json.loads(line.strip())
                if doc.get('content'):  # Only docs with content
                    documents.append({
                        'doc_id': doc.get('url_hash', f'doc_{i}'),
                        'content': doc['content'],
                        'metadata': {
                            'content_type': 'html',
                            'original_url': doc.get('url', ''),
                            'title': doc.get('title', ''),
                            'domain': doc.get('url', '').split('/')[2] if doc.get('url') else '',
                            'content_length': len(doc.get('content', ''))
                        }
                    })
            except Exception as e:
                print(f"Error loading doc {i}: {e}")
                continue
                
    return documents

def main():
    """Test chunking on raw JSONL sample"""
    print("=== QUICK CONTEXTUAL CHUNKING TEST ===")
    
    # Find a JSONL file with content
    jsonl_files = [
        '/home/ubuntu/LW_scrape/data/crawled/worker_1.jsonl',
        '/home/ubuntu/LW_scrape/data/crawled/worker_2.jsonl',
        '/home/ubuntu/LW_scrape/data/crawled/worker_3.jsonl'
    ]
    
    sample_docs = []
    for jsonl_path in jsonl_files:
        if os.path.exists(jsonl_path):
            print(f"Loading from {jsonl_path}")
            docs = load_sample_from_jsonl(jsonl_path, 2)
            sample_docs.extend(docs)
            if len(sample_docs) >= 3:
                break
    
    if not sample_docs:
        print("❌ No documents with content found")
        return
        
    print(f"Loaded {len(sample_docs)} sample documents")
    
    # Initialize chunker
    try:
        chunker = ContextualChunker(min_tokens=300, max_tokens=800)  # Smaller for testing
        print("✅ Contextual chunker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize chunker: {e}")
        return
    
    # Process samples
    all_chunks = []
    for doc in sample_docs[:2]:  # Just 2 docs for quick test
        print(f"\nProcessing: {doc['metadata'].get('title', 'Untitled')[:50]}...")
        print(f"Content length: {len(doc['content'])} chars")
        
        try:
            chunks = chunker.process_document(doc)
            all_chunks.extend(chunks)
            print(f"✅ Generated {len(chunks)} chunks")
            
            # Show first chunk details
            if chunks:
                chunk = chunks[0]
                print(f"Sample chunk: {chunk.summary_header}")
                
        except Exception as e:
            print(f"❌ Failed to process document: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_chunks:
        print(f"\n🎉 SUCCESS: Generated {len(all_chunks)} contextual chunks!")
        
        # Show detailed sample
        chunk = all_chunks[0]
        print("\n=== DETAILED SAMPLE CHUNK ===")
        print(f"Doc ID: {chunk.metadata.doc_id}")
        print(f"Chunk Index: {chunk.metadata.chunk_index}")
        print(f"Section Path: {chunk.section_path}")
        print(f"Token Count: {chunk.metadata.token_count}")
        print(f"Summary Header: {chunk.summary_header}")
        print(f"Previous Context: {chunk.prev_context[:100]}...")
        print(f"Content: {chunk.content[:200]}...")
        
    else:
        print("❌ No chunks generated")

if __name__ == "__main__":
    main()