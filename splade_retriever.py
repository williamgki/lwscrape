#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from pyserini.search import SimpleSearcher
    from pyserini.encode import SpladeDocumentEncoder, SpladeQueryEncoder
except Exception as e:  # pragma: no cover - optional dependency
    SimpleSearcher = None
    SpladeDocumentEncoder = None
    SpladeQueryEncoder = None
    logger.warning(f"Pyserini not available: {e}")


class SPLADERetriever:
    """SPLADE sparse retrieval implementation.

    Handles encoding of text to sparse vectors, caching during chunking,
    and query-time retrieval against a SPLADE index.
    """

    def __init__(self,
                 index_dir: str = "./splade_index",
                 model_name: str = "naver/splade-cocondenser-ensembledistil",
                 cache_dir: str = "./splade_cache"):
        self.index_dir = index_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.searcher = None
        self.doc_encoder = None
        self.query_encoder = None

        if SimpleSearcher and Path(index_dir).exists():
            try:
                self.searcher = SimpleSearcher(index_dir)
                logger.info(f"Loaded SPLADE index: {index_dir}")
            except Exception as e:
                logger.warning(f"Failed to load SPLADE index: {e}")

        if SpladeDocumentEncoder:
            try:
                self.doc_encoder = SpladeDocumentEncoder(model_name)
                self.query_encoder = SpladeQueryEncoder(model_name)
                logger.info(f"SPLADE encoders initialized: {model_name}")
            except Exception as e:
                logger.warning(f"SPLADE encoders unavailable: {e}")

    # ------------------------------------------------------------------
    # Encoding utilities
    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> Dict[str, float]:
        """Encode text into a sparse term-weight dict."""
        if not self.doc_encoder:
            logger.warning("Document encoder not initialized; returning empty vector")
            return {}
        try:
            return self.doc_encoder.encode(text)
        except Exception as e:
            logger.warning(f"SPLADE encoding failed: {e}")
            return {}

    def cache_chunk(self, chunk: Any) -> None:
        """Construct and cache SPLADE vector for a chunk.

        Args:
            chunk: Object with ``chunk_id`` and ``text`` attributes. The
                resulting sparse vector is stored in ``chunk.splade_sparse``.
        """
        if getattr(chunk, 'splade_sparse', None):
            return

        vector = self.encode_text(chunk.text)
        chunk.splade_sparse = vector

        if vector:
            cache_file = self.cache_dir / f"{chunk.chunk_id}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(vector, f)
            except Exception as e:
                logger.warning(f"Failed to cache SPLADE vector for {chunk.chunk_id}: {e}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 400) -> List[Dict[str, Any]]:
        """Search SPLADE index with encoded query."""
        if not self.searcher or not self.query_encoder:
            logger.warning("SPLADE searcher not available")
            return []

        try:
            query_vec = self.query_encoder.encode(query)
            hits = self.searcher.search(query_vec, k)
        except Exception as e:
            logger.warning(f"SPLADE search failed: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            results.append({
                'chunk_id': hit.docid,
                'score': hit.score,
                'method': 'splade',
                'rank': rank
            })
        return results
