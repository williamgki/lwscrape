#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from datetime import datetime
import argparse

from docunits_model import (
    DocUnit, DocRefEdge, SourceIds, DocUnitsExtractor, DocUnitsStorage,
    build_citation_edges
)
from structured_pdf_parser import StructuredPDFParser
from reference_normalizer import ReferenceNormalizer
from multi_source_corpus_builder import MultiSourceCorpusBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocUnitsPipeline:
    """Complete DocUnits extraction and storage pipeline"""
    
    def __init__(self, 
                 output_dir: Path = Path("./docunits_output"),
                 db_path: str = "docunits.db"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.extractor = DocUnitsExtractor()
        self.storage = DocUnitsStorage(db_path)
        self.pdf_parser = StructuredPDFParser()
        self.ref_normalizer = ReferenceNormalizer()
        self.corpus_builder = MultiSourceCorpusBuilder()
        
        # Stats tracking
        self.stats = {
            'papers_processed': 0,
            'docunits_created': 0,
            'edges_created': 0,
            'processing_errors': 0,
            'start_time': datetime.now()
        }
    
    async def process_paper(self, paper_data: Dict) -> List[DocUnit]:
        """Process a single paper into DocUnits"""
        try:
            paper_id = paper_data.get('openalex_id') or paper_data.get('arxiv_id') or paper_data.get('doi')
            if not paper_id:
                logger.warning(f"No valid paper ID found: {paper_data}")
                return []
            
            # Create SourceIds
            source_ids = SourceIds(
                openalex_id=paper_data.get('openalex_id'),
                doi=paper_data.get('doi'),
                arxiv_id=paper_data.get('arxiv_id'),
                openreview_id=paper_data.get('openreview_id')
            )
            
            units = []
            
            # Process PDF if available
            pdf_path = paper_data.get('pdf_path')
            if pdf_path and Path(pdf_path).exists():
                pdf_units = await self._process_pdf_to_docunits(
                    Path(pdf_path), paper_id, source_ids
                )
                units.extend(pdf_units)
            
            # Fallback: create basic text units from abstract/content
            if not units:
                fallback_units = self._create_fallback_units(paper_data, paper_id, source_ids)
                units.extend(fallback_units)
            
            self.stats['papers_processed'] += 1
            self.stats['docunits_created'] += len(units)
            
            logger.info(f"Created {len(units)} DocUnits for paper {paper_id}")
            return units
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_data.get('title', 'unknown')}: {e}")
            self.stats['processing_errors'] += 1
            return []
    
    async def _process_pdf_to_docunits(self, 
                                       pdf_path: Path, 
                                       paper_id: str, 
                                       source_ids: SourceIds) -> List[DocUnit]:
        """Extract DocUnits from PDF using GROBID + pdffigures2"""
        units = []
        
        try:
            # Parse PDF with GROBID and pdffigures2
            parse_result = await self.pdf_parser.parse_pdf_complete(pdf_path)
            
            if not parse_result:
                logger.warning(f"Failed to parse PDF: {pdf_path}")
                return []
            
            # Extract DocUnits from GROBID TEI
            if parse_result.get('grobid_tei'):
                tei_units = self.extractor.extract_from_grobid_tei(
                    Path(parse_result['grobid_tei']), paper_id, source_ids
                )
                units.extend(tei_units)
            
            # Extract DocUnits from pdffigures2
            if parse_result.get('pdffigures2_json'):
                fig_units = self.extractor.extract_from_pdffigures2(
                    Path(parse_result['pdffigures2_json']), paper_id, source_ids
                )
                units.extend(fig_units)
            
            # Link references to canonical IDs
            if units:
                units = await self._normalize_references_in_units(units)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return units
    
    def _create_fallback_units(self, 
                              paper_data: Dict, 
                              paper_id: str, 
                              source_ids: SourceIds) -> List[DocUnit]:
        """Create basic DocUnits when PDF parsing fails"""
        units = []
        
        # Abstract unit
        if paper_data.get('abstract'):
            abstract_unit = DocUnit(
                paper_id=paper_id,
                source_ids=source_ids,
                section_id="abstract",
                page=1,
                para_id=f"{paper_id}_abstract",
                type="text",
                text=paper_data['abstract']
            )
            units.append(abstract_unit)
        
        # Title unit
        if paper_data.get('title'):
            title_unit = DocUnit(
                paper_id=paper_id,
                source_ids=source_ids,
                section_id="title",
                page=1,
                para_id=f"{paper_id}_title",
                type="text",
                text=paper_data['title']
            )
            units.append(title_unit)
        
        # Authors unit
        if paper_data.get('authors'):
            authors_text = "; ".join([
                author.get('display_name', '') for author in paper_data['authors']
                if author.get('display_name')
            ])
            if authors_text:
                authors_unit = DocUnit(
                    paper_id=paper_id,
                    source_ids=source_ids,
                    section_id="authors",
                    page=1,
                    para_id=f"{paper_id}_authors",
                    type="text",
                    text=f"Authors: {authors_text}"
                )
                units.append(authors_unit)
        
        return units
    
    async def _normalize_references_in_units(self, units: List[DocUnit]) -> List[DocUnit]:
        """Normalize reference IDs in DocUnits using Crossref/OpenAlex"""
        ref_mapping = {}
        
        # Collect all unique references
        all_refs = set()
        for unit in units:
            all_refs.update(unit.refs)
        
        # Batch normalize references
        for ref_id in all_refs:
            try:
                normalized = await self.ref_normalizer.normalize_reference({'id': ref_id})
                if normalized and normalized.canonical_id:
                    ref_mapping[ref_id] = normalized.canonical_id
                else:
                    ref_mapping[ref_id] = ref_id  # Keep original if normalization fails
            except Exception as e:
                logger.warning(f"Failed to normalize reference {ref_id}: {e}")
                ref_mapping[ref_id] = ref_id
        
        # Update units with normalized references
        for unit in units:
            unit.refs = [ref_mapping.get(ref, ref) for ref in unit.refs]
        
        return units
    
    def build_citation_graph(self, all_units: List[DocUnit]) -> List[DocRefEdge]:
        """Build citation graph from all DocUnits"""
        logger.info("Building citation graph...")
        
        # Create reference mapping (this would be more sophisticated in practice)
        ref_mapping = {}
        paper_ids = {unit.paper_id for unit in all_units}
        
        # Simple mapping: assume normalized refs are already paper IDs
        for unit in all_units:
            for ref in unit.refs:
                if ref in paper_ids:
                    ref_mapping[ref] = ref
        
        edges = build_citation_edges(all_units, ref_mapping)
        self.stats['edges_created'] = len(edges)
        
        logger.info(f"Created {len(edges)} citation edges")
        return edges
    
    async def process_corpus(self, 
                           corpus_data: List[Dict],
                           max_papers: Optional[int] = None) -> None:
        """Process entire corpus into DocUnits"""
        logger.info(f"Processing corpus of {len(corpus_data)} papers")
        
        if max_papers:
            corpus_data = corpus_data[:max_papers]
            logger.info(f"Limited to {max_papers} papers")
        
        all_units = []
        
        # Process papers in batches
        batch_size = 10
        for i in range(0, len(corpus_data), batch_size):
            batch = corpus_data[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} papers)")
            
            # Process batch concurrently
            tasks = [self.process_paper(paper) for paper in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, list):
                    all_units.extend(result)
                else:
                    logger.error(f"Batch processing error: {result}")
            
            # Insert batch into storage
            if all_units:
                batch_units = all_units[-sum(len(r) for r in batch_results if isinstance(r, list)):]
                self.storage.insert_docunits(batch_units)
                logger.info(f"Inserted {len(batch_units)} DocUnits to database")
        
        # Build and store citation graph
        edges = self.build_citation_graph(all_units)
        if edges:
            self.storage.insert_edges(edges)
        
        # Export to Parquet
        self.storage.export_to_parquet(self.output_dir)
        
        # Save processing stats
        await self._save_processing_stats()
        
        logger.info("Corpus processing complete!")
    
    async def _save_processing_stats(self):
        """Save processing statistics"""
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = str(self.stats['end_time'] - self.stats['start_time'])
        self.stats['db_stats'] = self.storage.get_stats()
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        logger.info(f"Processing stats saved to {stats_file}")
    
    def close(self):
        """Clean up resources"""
        self.storage.close()


async def main():
    parser = argparse.ArgumentParser(description="DocUnits Pipeline")
    parser.add_argument("--corpus-file", required=True, help="Path to corpus JSON file")
    parser.add_argument("--output-dir", default="./docunits_output", help="Output directory")
    parser.add_argument("--db-path", default="docunits.db", help="Database path")
    parser.add_argument("--max-papers", type=int, help="Limit number of papers to process")
    
    args = parser.parse_args()
    
    # Load corpus data
    corpus_file = Path(args.corpus_file)
    if not corpus_file.exists():
        logger.error(f"Corpus file not found: {corpus_file}")
        return
    
    with open(corpus_file, 'r') as f:
        corpus_data = json.load(f)
    
    # Initialize and run pipeline
    pipeline = DocUnitsPipeline(
        output_dir=Path(args.output_dir),
        db_path=args.db_path
    )
    
    try:
        await pipeline.process_corpus(corpus_data, args.max_papers)
    finally:
        pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())