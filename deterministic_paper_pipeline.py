#!/usr/bin/env python3

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
from dataclasses import asdict

from typed_paper_extractor import (
    GROBIDTEIParser, TypedObjectExtractor, ExtractionValidator,
    PaperStructure, TypedPaperExtraction
)
from structured_pdf_parser import StructuredPDFParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeterministicPaperPipeline:
    """Complete deterministic paper analysis pipeline"""
    
    def __init__(self,
                 anthropic_client=None,
                 model_name: str = "claude-3-5-sonnet-20241022",
                 output_dir: str = "./typed_extractions"):
        
        # Initialize components
        self.tei_parser = GROBIDTEIParser()
        self.extractor = TypedObjectExtractor(anthropic_client, model_name)
        self.pdf_parser = StructuredPDFParser()
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "structures").mkdir(exist_ok=True)
        (self.output_dir / "extractions").mkdir(exist_ok=True)
        (self.output_dir / "validated").mkdir(exist_ok=True)
        
        # Processing stats
        self.stats = {
            'papers_processed': 0,
            'successful_extractions': 0,
            'validation_failures': 0,
            'processing_errors': 0
        }
    
    async def process_single_paper(self,
                                 pdf_path: Path,
                                 paper_id: str) -> Optional[TypedPaperExtraction]:
        """Complete pipeline for a single paper"""
        
        logger.info(f"Processing paper: {paper_id}")
        start_time = time.time()
        
        try:
            # Step 1: Parse PDF structure with GROBID
            parse_result = await self.pdf_parser.parse_pdf_complete(pdf_path)
            
            if not parse_result or not parse_result.get('grobid_tei'):
                logger.error(f"Failed to parse PDF structure for {paper_id}")
                self.stats['processing_errors'] += 1
                return None
            
            tei_file = Path(parse_result['grobid_tei'])
            
            # Step 2: Extract paper structure from TEI
            paper_structure = self.tei_parser.parse_paper_structure(tei_file, paper_id)
            paper_structure.pdf_path = str(pdf_path)
            
            # Save structure
            structure_file = self.output_dir / "structures" / f"{paper_id}_structure.json"
            self._save_paper_structure(paper_structure, structure_file)
            
            # Step 3: Extract typed objects
            extraction = self.extractor.extract_full_paper(paper_structure)
            
            # Save raw extraction
            extraction_file = self.output_dir / "extractions" / f"{paper_id}_extraction.json"
            self._save_extraction(extraction, extraction_file)
            
            # Step 4: Validate extraction
            validator = ExtractionValidator(paper_structure)
            validated_extraction = validator.validate_extraction(extraction)
            
            # Save validated extraction
            validated_file = self.output_dir / "validated" / f"{paper_id}_validated.json"
            self._save_extraction(validated_extraction, validated_file)
            
            # Update stats
            self.stats['papers_processed'] += 1
            self.stats['successful_extractions'] += 1
            
            processing_time = time.time() - start_time
            logger.info(f"Completed {paper_id} in {processing_time:.2f}s")
            
            return validated_extraction
            
        except Exception as e:
            logger.error(f"Error processing {paper_id}: {e}")
            self.stats['processing_errors'] += 1
            return None
    
    def _save_paper_structure(self, structure: PaperStructure, output_file: Path):
        """Save paper structure to JSON"""
        
        structure_data = {
            'paper_id': structure.paper_id,
            'title': structure.title,
            'abstract': structure.abstract,
            'authors': structure.authors,
            'page_count': structure.page_count,
            'tei_path': structure.tei_path,
            'pdf_path': structure.pdf_path,
            'sections': []
        }
        
        for section in structure.sections:
            section_data = {
                'title': section.title,
                'content': section.content,
                'level': section.level,
                'page_start': section.page_start,
                'page_end': section.page_end,
                'tei_type': section.tei_type
            }
            structure_data['sections'].append(section_data)
        
        with open(output_file, 'w') as f:
            json.dump(structure_data, f, indent=2)
    
    def _save_extraction(self, extraction: TypedPaperExtraction, output_file: Path):
        """Save extraction to JSON"""
        
        # Convert Pydantic models to dict
        extraction_data = extraction.dict()
        
        # Add metadata
        extraction_data['_metadata'] = {
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': self.extractor.model_name,
            'validation_passed': 'validated' in str(output_file)
        }
        
        with open(output_file, 'w') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)
    
    async def process_paper_batch(self,
                                paper_paths: List[Tuple[Path, str]],
                                max_concurrent: int = 3) -> List[TypedPaperExtraction]:
        """Process multiple papers concurrently"""
        
        logger.info(f"Processing batch of {len(paper_paths)} papers")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_path: Path, paper_id: str):
            async with semaphore:
                return await self.process_single_paper(pdf_path, paper_id)
        
        # Process papers concurrently
        tasks = [process_with_semaphore(path, pid) for path, pid in paper_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, TypedPaperExtraction):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
        
        logger.info(f"Batch completed: {len(successful_results)} successful")
        return successful_results
    
    def analyze_extraction_quality(self, paper_id: str) -> Dict[str, Any]:
        """Analyze quality of extraction for a paper"""
        
        extraction_file = self.output_dir / "extractions" / f"{paper_id}_extraction.json"
        validated_file = self.output_dir / "validated" / f"{paper_id}_validated.json"
        
        if not extraction_file.exists() or not validated_file.exists():
            return {"error": "Extraction files not found"}
        
        # Load both versions
        with open(extraction_file, 'r') as f:
            raw_extraction = json.load(f)
        
        with open(validated_file, 'r') as f:
            validated_extraction = json.load(f)
        
        # Analyze differences
        analysis = {
            'paper_id': paper_id,
            'raw_counts': {
                'claims': len(raw_extraction.get('claims', [])),
                'assumptions': len(raw_extraction.get('assumptions', [])),
                'mechanisms': len(raw_extraction.get('mechanisms', [])),
                'metrics': len(raw_extraction.get('metrics', [])),
                'experiments': len(raw_extraction.get('experiments', [])),
                'threat_models': len(raw_extraction.get('threat_models', []))
            },
            'validated_counts': {
                'claims': len(validated_extraction.get('claims', [])),
                'assumptions': len(validated_extraction.get('assumptions', [])),
                'mechanisms': len(validated_extraction.get('mechanisms', [])),
                'metrics': len(validated_extraction.get('metrics', [])),
                'experiments': len(validated_extraction.get('experiments', [])),
                'threat_models': len(validated_extraction.get('threat_models', []))
            }
        }
        
        # Calculate validation success rates
        analysis['validation_rates'] = {}
        for category in analysis['raw_counts']:
            raw_count = analysis['raw_counts'][category]
            validated_count = analysis['validated_counts'][category]
            
            if raw_count > 0:
                rate = validated_count / raw_count
            else:
                rate = 1.0 if validated_count == 0 else 0.0
            
            analysis['validation_rates'][category] = rate
        
        return analysis
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        return {
            'processing_stats': self.stats.copy(),
            'output_files': {
                'structures': len(list((self.output_dir / "structures").glob("*.json"))),
                'extractions': len(list((self.output_dir / "extractions").glob("*.json"))),
                'validated': len(list((self.output_dir / "validated").glob("*.json")))
            },
            'success_rate': (
                self.stats['successful_extractions'] / max(self.stats['papers_processed'], 1)
            ) if self.stats['papers_processed'] > 0 else 0.0
        }
    
    def export_corpus_extractions(self, output_file: Path):
        """Export all validated extractions to single corpus file"""
        
        corpus_data = []
        validated_files = list((self.output_dir / "validated").glob("*.json"))
        
        for validated_file in validated_files:
            try:
                with open(validated_file, 'r') as f:
                    extraction_data = json.load(f)
                
                corpus_data.append(extraction_data)
                
            except Exception as e:
                logger.warning(f"Failed to load {validated_file}: {e}")
        
        # Save corpus
        corpus_export = {
            'extraction_count': len(corpus_data),
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'papers': corpus_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(corpus_export, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(corpus_data)} paper extractions to {output_file}")


class ExtractionAnalyzer:
    """Analyze and summarize extracted typed objects"""
    
    def __init__(self, pipeline: DeterministicPaperPipeline):
        self.pipeline = pipeline
    
    def analyze_corpus_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across the corpus of extractions"""
        
        validated_files = list((self.pipeline.output_dir / "validated").glob("*.json"))
        
        corpus_stats = {
            'total_papers': len(validated_files),
            'category_totals': {
                'claims': 0, 'assumptions': 0, 'mechanisms': 0,
                'metrics': 0, 'experiments': 0, 'threat_models': 0
            },
            'claim_types': {'theoretical': 0, 'empirical': 0, 'engineering': 0},
            'papers_by_category': {
                'claims': 0, 'assumptions': 0, 'mechanisms': 0,
                'metrics': 0, 'experiments': 0, 'threat_models': 0
            }
        }
        
        for validated_file in validated_files:
            try:
                with open(validated_file, 'r') as f:
                    extraction = json.load(f)
                
                # Count totals
                for category in corpus_stats['category_totals']:
                    count = len(extraction.get(category, []))
                    corpus_stats['category_totals'][category] += count
                    
                    if count > 0:
                        corpus_stats['papers_by_category'][category] += 1
                
                # Count claim types
                for claim in extraction.get('claims', []):
                    claim_type = claim.get('type', 'unknown')
                    if claim_type in corpus_stats['claim_types']:
                        corpus_stats['claim_types'][claim_type] += 1
                        
            except Exception as e:
                logger.warning(f"Failed to analyze {validated_file}: {e}")
        
        # Calculate averages
        if corpus_stats['total_papers'] > 0:
            corpus_stats['averages'] = {
                category: total / corpus_stats['total_papers']
                for category, total in corpus_stats['category_totals'].items()
            }
        
        return corpus_stats


async def main():
    """Demo deterministic paper pipeline"""
    
    # Initialize pipeline (would need actual Anthropic client)
    pipeline = DeterministicPaperPipeline()
    
    print("=== Deterministic Paper Analysis Pipeline ===")
    print("\nPipeline stages:")
    print("1. GROBID TEI deterministic segmentation")
    print("2. Structured section extraction")
    print("3. LLM typed object extraction (temperature=0)")
    print("4. Evidence span validation")
    print("5. Page cross-checking")
    
    print(f"\nOutput structure:")
    print("- structures/: Paper segmentation from GROBID TEI")
    print("- extractions/: Raw LLM extractions")
    print("- validated/: Post-processed with evidence validation")
    
    # Example of what would be extracted
    example_extraction = {
        "paper_id": "example_paper",
        "claims": [
            {
                "id": "C1",
                "text": "Our method achieves 95% accuracy on benchmark X",
                "type": "empirical",
                "section": "4. Results",
                "page": 5,
                "evidence_spans": [
                    {
                        "page": 5,
                        "quote": "The proposed approach achieves an accuracy of 95.2% on the standard benchmark dataset X",
                        "span_hint": "Table 2"
                    }
                ]
            }
        ],
        "mechanisms": [
            {
                "id": "M1", 
                "text": "Attention mechanism focuses on relevant input features",
                "page": 3,
                "evidence_spans": [
                    {
                        "page": 3,
                        "quote": "The attention layer learns to weight input features by relevance",
                        "span_hint": "Section 3.1"
                    }
                ]
            }
        ]
    }
    
    print("\nExample extraction format:")
    print(json.dumps(example_extraction, indent=2)[:800] + "...")


if __name__ == "__main__":
    asyncio.run(main())