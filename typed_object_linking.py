#!/usr/bin/env python3

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime

from typed_paper_extractor import TypedPaperExtraction, Claim
from fusion_retrieval_pipeline import ProductionFusionPipeline, FusedResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchIntent:
    """Search query intent for a claim"""
    intent_type: str  # "supporting", "contradictory", "cross_domain"
    query: str
    description: str


@dataclass
class EvidenceLedgerEntry:
    """Evidence entry for a claim"""
    claim_id: str
    paper_id: str
    intent_type: str  # "supporting", "contradictory", "cross_domain"
    query: str
    search_results: List[FusedResult]
    retrieval_timestamp: str
    retrieval_scores: Dict[str, float]


class ClaimSearchIntentGenerator:
    """Generate search intents for claims"""
    
    def __init__(self, anthropic_client=None, model_name: str = "claude-3-5-sonnet-20241022"):
        self.anthropic_client = anthropic_client
        self.model_name = model_name
        
        # Intent generation prompt
        self.intent_prompt = """Generate search queries to find evidence for this claim. Create exactly:
- 4-6 supporting queries: Find evidence that supports/confirms the claim
- 4-6 contradictory/edge case queries: Find evidence that challenges or limits the claim  
- 2-3 cross-domain queries: Find related concepts from different AI domains

Claim: "{claim_text}"
Paper context: "{paper_title}" - Section: "{section}"

Output format:
{
  "supporting": ["query1", "query2", "query3", "query4"],
  "contradictory": ["query1", "query2", "query3", "query4"],
  "cross_domain": ["query1", "query2"]
}

Make queries specific and searchable. Focus on technical terms and concepts."""
    
    def generate_search_intents(self, 
                              claim: Claim, 
                              paper_title: str) -> List[SearchIntent]:
        """Generate search intents for a claim"""
        
        if not self.anthropic_client:
            # Fallback to simple intent generation
            return self._generate_simple_intents(claim)
        
        try:
            prompt = self.intent_prompt.format(
                claim_text=claim.text,
                paper_title=paper_title,
                section=claim.section
            )
            
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.3,  # Some creativity for diverse queries
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            response_text = response.content[0].text
            intent_data = json.loads(response_text)
            
            search_intents = []
            
            # Convert to SearchIntent objects
            for intent_type in ["supporting", "contradictory", "cross_domain"]:
                queries = intent_data.get(intent_type, [])
                for query in queries:
                    intent = SearchIntent(
                        intent_type=intent_type,
                        query=query,
                        description=f"{intent_type.title()} evidence for: {claim.text[:100]}..."
                    )
                    search_intents.append(intent)
            
            logger.info(f"Generated {len(search_intents)} search intents for claim {claim.id}")
            return search_intents
            
        except Exception as e:
            logger.error(f"Failed to generate intents for claim {claim.id}: {e}")
            return self._generate_simple_intents(claim)
    
    def _generate_simple_intents(self, claim: Claim) -> List[SearchIntent]:
        """Fallback simple intent generation"""
        
        # Extract key terms from claim
        claim_lower = claim.text.lower()
        
        # Simple supporting queries
        supporting = [
            claim.text,  # Exact claim
            f"{claim.text} evidence",
            f"{claim.text} results",
            f"{claim.text} validation"
        ]
        
        # Simple contradictory queries
        contradictory = [
            f"{claim.text} limitations",
            f"{claim.text} challenges",
            f"{claim.text} edge cases",
            f"{claim.text} failure modes"
        ]
        
        # Simple cross-domain (based on claim type)
        cross_domain = [
            f"{claim.text} interpretability",
            f"{claim.text} scalability"
        ]
        
        intents = []
        
        for intent_type, queries in [
            ("supporting", supporting),
            ("contradictory", contradictory), 
            ("cross_domain", cross_domain)
        ]:
            for query in queries:
                intent = SearchIntent(
                    intent_type=intent_type,
                    query=query,
                    description=f"Simple {intent_type} search"
                )
                intents.append(intent)
        
        return intents


class EvidenceLedger:
    """Store and manage evidence for typed objects"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Evidence entries by claim ID
        self.entries: Dict[str, List[EvidenceLedgerEntry]] = {}
        
        # Load existing entries
        self._load_existing_entries()
    
    def _load_existing_entries(self):
        """Load existing evidence entries"""
        evidence_files = list(self.storage_path.glob("*_evidence.json"))
        
        for evidence_file in evidence_files:
            try:
                with open(evidence_file, 'r') as f:
                    data = json.load(f)
                
                for entry_data in data.get('entries', []):
                    entry = EvidenceLedgerEntry(
                        claim_id=entry_data['claim_id'],
                        paper_id=entry_data['paper_id'],
                        intent_type=entry_data['intent_type'],
                        query=entry_data['query'],
                        search_results=[],  # Results stored separately
                        retrieval_timestamp=entry_data['retrieval_timestamp'],
                        retrieval_scores=entry_data.get('retrieval_scores', {})
                    )
                    
                    claim_id = entry.claim_id
                    if claim_id not in self.entries:
                        self.entries[claim_id] = []
                    self.entries[claim_id].append(entry)
                    
            except Exception as e:
                logger.warning(f"Failed to load evidence file {evidence_file}: {e}")
    
    def add_evidence_entry(self, 
                          claim_id: str,
                          paper_id: str,
                          intent: SearchIntent,
                          search_results: List[FusedResult]):
        """Add evidence entry to ledger"""
        
        # Calculate aggregate scores
        if search_results:
            scores = {
                'avg_colbert_score': sum(r.colbert_score for r in search_results) / len(search_results),
                'avg_bm25_score': sum(r.bm25_score for r in search_results) / len(search_results),
                'avg_reranker_score': sum(r.reranker_score for r in search_results) / len(search_results),
                'max_final_score': max(r.final_score for r in search_results),
                'result_count': len(search_results)
            }
        else:
            scores = {'result_count': 0}
        
        entry = EvidenceLedgerEntry(
            claim_id=claim_id,
            paper_id=paper_id,
            intent_type=intent.intent_type,
            query=intent.query,
            search_results=search_results,
            retrieval_timestamp=datetime.now().isoformat(),
            retrieval_scores=scores
        )
        
        if claim_id not in self.entries:
            self.entries[claim_id] = []
        
        self.entries[claim_id].append(entry)
        
        logger.info(f"Added evidence entry for {claim_id}: {intent.intent_type} - {len(search_results)} results")
    
    def get_evidence_summary(self, claim_id: str) -> Dict[str, Any]:
        """Get summary of evidence for a claim"""
        
        if claim_id not in self.entries:
            return {"claim_id": claim_id, "total_entries": 0}
        
        entries = self.entries[claim_id]
        
        summary = {
            'claim_id': claim_id,
            'total_entries': len(entries),
            'by_intent_type': {},
            'total_results': 0,
            'best_results': []
        }
        
        # Group by intent type
        for entry in entries:
            intent_type = entry.intent_type
            if intent_type not in summary['by_intent_type']:
                summary['by_intent_type'][intent_type] = {
                    'queries': 0,
                    'total_results': 0,
                    'avg_score': 0.0
                }
            
            summary['by_intent_type'][intent_type]['queries'] += 1
            summary['by_intent_type'][intent_type]['total_results'] += len(entry.search_results)
            summary['total_results'] += len(entry.search_results)
        
        # Get best results across all intents
        all_results = []
        for entry in entries:
            all_results.extend(entry.search_results)
        
        # Sort by final score and take top 10
        all_results.sort(key=lambda x: x.final_score, reverse=True)
        summary['best_results'] = all_results[:10]
        
        return summary
    
    def save_ledger(self, paper_id: str):
        """Save evidence ledger for a paper"""
        
        # Filter entries for this paper
        paper_entries = []
        for claim_id, entries in self.entries.items():
            paper_entries.extend([e for e in entries if e.paper_id == paper_id])
        
        if not paper_entries:
            return
        
        # Prepare export data
        export_data = {
            'paper_id': paper_id,
            'export_timestamp': datetime.now().isoformat(),
            'total_entries': len(paper_entries),
            'entries': []
        }
        
        for entry in paper_entries:
            entry_data = {
                'claim_id': entry.claim_id,
                'paper_id': entry.paper_id,
                'intent_type': entry.intent_type,
                'query': entry.query,
                'retrieval_timestamp': entry.retrieval_timestamp,
                'retrieval_scores': entry.retrieval_scores,
                'search_results': [
                    {
                        'chunk_id': r.chunk_id,
                        'paper_id': r.paper_id,
                        'page_from': r.page_from,
                        'page_to': r.page_to,
                        'text_preview': r.text[:200] + "..." if len(r.text) > 200 else r.text,
                        'headers': r.headers,
                        'final_score': r.final_score,
                        'final_rank': r.final_rank
                    }
                    for r in entry.search_results[:20]  # Limit results saved
                ]
            }
            export_data['entries'].append(entry_data)
        
        # Save to file
        output_file = self.storage_path / f"{paper_id}_evidence.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evidence ledger for {paper_id}: {len(paper_entries)} entries")


class TypedObjectLinker:
    """Link typed objects to retrieval evidence"""
    
    def __init__(self,
                 fusion_pipeline: ProductionFusionPipeline,
                 anthropic_client=None,
                 evidence_storage_path: str = "./evidence_ledgers"):
        
        self.fusion_pipeline = fusion_pipeline
        self.intent_generator = ClaimSearchIntentGenerator(anthropic_client)
        self.evidence_ledger = EvidenceLedger(Path(evidence_storage_path))
    
    async def link_paper_claims(self, 
                              extraction: TypedPaperExtraction,
                              paper_title: str) -> Dict[str, Any]:
        """Link all claims in a paper to retrieval evidence"""
        
        logger.info(f"Linking claims for paper: {extraction.paper_id}")
        
        linking_stats = {
            'paper_id': extraction.paper_id,
            'total_claims': len(extraction.claims),
            'processed_claims': 0,
            'total_intents': 0,
            'total_evidence_entries': 0
        }
        
        for claim in extraction.claims:
            # Skip speculative claims for now
            if getattr(claim, 'status', 'VALID') == 'SPECULATIVE':
                logger.info(f"Skipping speculative claim {claim.id}")
                continue
            
            await self._link_single_claim(claim, extraction.paper_id, paper_title)
            linking_stats['processed_claims'] += 1
        
        # Save evidence ledger
        self.evidence_ledger.save_ledger(extraction.paper_id)
        
        # Update stats
        for claim_id in [c.id for c in extraction.claims]:
            if claim_id in self.evidence_ledger.entries:
                linking_stats['total_evidence_entries'] += len(self.evidence_ledger.entries[claim_id])
        
        logger.info(f"Linking completed for {extraction.paper_id}: {linking_stats}")
        return linking_stats
    
    async def _link_single_claim(self, 
                               claim: Claim, 
                               paper_id: str, 
                               paper_title: str):
        """Link a single claim to evidence"""
        
        logger.info(f"Linking claim {claim.id}: {claim.text[:100]}...")
        
        # Generate search intents
        search_intents = self.intent_generator.generate_search_intents(claim, paper_title)
        
        # Execute searches for each intent
        for intent in search_intents:
            try:
                # Perform fusion search
                search_results = await self.fusion_pipeline.full_fusion_search(
                    query=intent.query,
                    colbert_k=400,
                    bm25_k=400,
                    splade_k=400,
                    rrf_k=200,  # Get top 200 for BGE reranking
                    rerank_k=50   # Final top 50 results
                )
                
                # Add to evidence ledger
                self.evidence_ledger.add_evidence_entry(
                    claim_id=claim.id,
                    paper_id=paper_id,
                    intent=intent,
                    search_results=search_results
                )
                
            except Exception as e:
                logger.error(f"Failed to search for intent '{intent.query}': {e}")
    
    def get_claim_evidence_report(self, claim_id: str) -> Dict[str, Any]:
        """Get detailed evidence report for a claim"""
        
        summary = self.evidence_ledger.get_evidence_summary(claim_id)
        
        if summary['total_entries'] == 0:
            return summary
        
        # Add detailed analysis
        entries = self.evidence_ledger.entries.get(claim_id, [])
        
        # Cross-paper evidence analysis
        evidence_papers = set()
        for entry in entries:
            for result in entry.search_results:
                evidence_papers.add(result.paper_id)
        
        summary['evidence_papers'] = list(evidence_papers)
        summary['cross_paper_evidence_count'] = len(evidence_papers)
        
        # Evidence strength analysis
        supporting_results = []
        contradictory_results = []
        
        for entry in entries:
            if entry.intent_type == "supporting":
                supporting_results.extend(entry.search_results[:5])  # Top 5 per query
            elif entry.intent_type == "contradictory":
                contradictory_results.extend(entry.search_results[:5])
        
        summary['evidence_strength'] = {
            'supporting_count': len(supporting_results),
            'contradictory_count': len(contradictory_results),
            'avg_supporting_score': sum(r.final_score for r in supporting_results) / max(len(supporting_results), 1),
            'avg_contradictory_score': sum(r.final_score for r in contradictory_results) / max(len(contradictory_results), 1)
        }
        
        return summary


async def main():
    """Demo typed object linking"""
    
    # Initialize components
    fusion_pipeline = ProductionFusionPipeline()
    linker = TypedObjectLinker(fusion_pipeline)
    
    print("=== Typed Object Linking Demo ===")
    print("\nC3. Link typed objects to retrieval:")
    print("✓ Generate 4-6 supporting queries per claim")
    print("✓ Generate 4-6 contradictory/edge case queries")  
    print("✓ Generate 2-3 cross-domain queries")
    print("✓ ColBERT + BM25/SPLADE → RRF → BGE rerank (200→50)")
    print("✓ Store in Evidence Ledger keyed by claim ID")
    
    print("\nSearch intent types:")
    print("- Supporting: Find confirming evidence")
    print("- Contradictory: Find challenging evidence")
    print("- Cross-domain: Find related concepts from other AI domains")
    
    print("\nEvidence Ledger structure:")
    example_entry = {
        "claim_id": "C1",
        "paper_id": "paper_123",
        "intent_type": "supporting",
        "query": "neural network accuracy 95% benchmark dataset",
        "search_results": [
            {
                "chunk_id": "chunk_abc",
                "paper_id": "evidence_paper_456",
                "text_preview": "Our model achieves 94.8% accuracy...",
                "final_score": 0.92,
                "final_rank": 1
            }
        ],
        "retrieval_scores": {
            "avg_reranker_score": 0.89,
            "result_count": 15
        }
    }
    print(json.dumps(example_entry, indent=2)[:500] + "...")


if __name__ == "__main__":
    asyncio.run(main())