#!/usr/bin/env python3
"""
Complete Corpus Pipeline
End-to-end processing from paper discovery to columnar storage
"""

import logging
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import asyncio
from typing import List, Optional

# Add local modules to path
sys.path.append('/home/ubuntu/LW_scrape')

from multi_source_corpus_builder import MultiSourceCorpusBuilder
from corpus_deduplicator import CorpusDeduplicator
from multisource_to_wilson_lin import MultiSourceProcessor
from reference_normalizer import ReferenceNormalizer
from columnar_storage_manager import ColumnarStorageManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteCorpusPipeline:
    def __init__(self, 
                 corpus_dir: str = "/home/ubuntu/LW_scrape/multi_source_corpus",
                 final_output_dir: str = "/home/ubuntu/LW_scrape/final_corpus_v2"):
        
        self.corpus_dir = Path(corpus_dir)
        self.final_output_dir = Path(final_output_dir)
        self.final_output_dir.mkdir(exist_ok=True)
        
        # Pipeline statistics
        self.pipeline_stats = {
            'start_time': datetime.now().isoformat(),
            'papers_collected': 0,
            'papers_deduplicated': 0,
            'papers_downloaded': 0,
            'references_extracted': 0,
            'references_normalized': 0,
            'chunks_created': 0,
            'citations_mapped': 0,
            'low_evidence_papers': 0,
            'low_info_survey_nodes': 0
        }
    
    def run_paper_collection(self, openalex_pages: int = 10, arxiv_results: int = 2000, openreview_limit: int = 500) -> str:
        """Step 1: Collect papers from multiple sources"""
        
        logger.info("🚀 STEP 1: Multi-source paper collection")
        
        try:
            # Run collection script
            cmd = [
                'python', 'multi_source_corpus_builder.py',
                '--openalex-pages', str(openalex_pages),
                '--arxiv-results', str(arxiv_results), 
                '--openreview-limit', str(openreview_limit)
            ]
            
            result = subprocess.run(cmd, cwd=self.corpus_dir.parent, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check output file
                papers_file = self.corpus_dir / "raw_papers_collected.json"
                if papers_file.exists():
                    with open(papers_file, 'r') as f:
                        papers = json.load(f)
                    self.pipeline_stats['papers_collected'] = len(papers)
                    logger.info(f"✅ Collected {len(papers)} papers from APIs")
                    return str(papers_file)
                else:
                    logger.error("❌ Papers collection file not found")
                    return ""
            else:
                logger.error(f"❌ Paper collection failed: {result.stderr}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Paper collection error: {e}")
            return ""
    
    def run_deduplication(self, papers_file: str) -> str:
        """Step 2: Deduplicate collected papers"""
        
        logger.info("🚀 STEP 2: Paper deduplication")
        
        try:
            deduplicator = CorpusDeduplicator(str(self.corpus_dir))
            
            # Load papers
            with open(papers_file, 'r') as f:
                papers = json.load(f)
            
            # Deduplicate
            deduplicated_papers = deduplicator.deduplicate_papers(papers)
            deduplicated_papers = deduplicator.assign_paper_ids(deduplicated_papers)
            
            # Save results
            deduplicator.save_deduplicated_papers(deduplicated_papers)
            
            self.pipeline_stats['papers_deduplicated'] = len(deduplicated_papers)
            logger.info(f"✅ Deduplicated to {len(deduplicated_papers)} unique papers")
            
            return str(self.corpus_dir / "deduplicated_papers.json")
            
        except Exception as e:
            logger.error(f"❌ Deduplication error: {e}")
            return ""
    
    def run_pdf_processing(self) -> str:
        """Step 3: Download PDFs and prepare for processing"""
        
        logger.info("🚀 STEP 3: PDF download and processing")
        
        try:
            processor = MultiSourceProcessor(str(self.corpus_dir))
            
            # Load deduplicated papers
            papers = processor.load_deduplicated_papers()
            
            # Download PDFs
            downloaded_papers = processor.download_paper_pdfs(papers)
            
            # Create Wilson Lin input
            input_file = processor.create_wilson_lin_input(downloaded_papers)
            
            self.pipeline_stats['papers_downloaded'] = len(downloaded_papers)
            logger.info(f"✅ Downloaded {len(downloaded_papers)} PDFs")
            
            return input_file
            
        except Exception as e:
            logger.error(f"❌ PDF processing error: {e}")
            return ""
    
    def run_structured_parsing(self) -> bool:
        """Step 4: Parse PDFs with GROBID (if available)"""
        
        logger.info("🚀 STEP 4: Structured PDF parsing")
        
        try:
            # Check if GROBID is available
            cmd = ['python', 'structured_pdf_parser.py', '--corpus-dir', str(self.corpus_dir)]
            
            result = subprocess.run(cmd, cwd=self.corpus_dir.parent, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ Structured parsing completed")
                return True
            else:
                logger.warning(f"⚠️ Structured parsing had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Structured parsing timed out, continuing with basic processing")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Structured parsing error: {e}")
            return False
    
    def run_reference_normalization(self) -> str:
        """Step 5: Normalize references via Crossref/OpenAlex"""
        
        logger.info("🚀 STEP 5: Reference normalization")
        
        try:
            normalizer = ReferenceNormalizer(str(self.corpus_dir))
            
            # Process TEI files if available
            tei_files = list(normalizer.tei_dir.glob("*.tei.xml"))
            
            if tei_files:
                output_file = normalizer.process_all_tei_files()
                
                if output_file:
                    # Count results
                    import pandas as pd
                    df = pd.read_parquet(output_file)
                    self.pipeline_stats['references_extracted'] = len(df)
                    self.pipeline_stats['references_normalized'] = len(df[df['normalization_method'] != 'failed'])
                    
                    logger.info(f"✅ Normalized {len(df)} references")
                    return output_file
                else:
                    logger.warning("⚠️ Reference normalization produced no results")
                    return ""
            else:
                logger.warning("⚠️ No TEI files found for reference extraction")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Reference normalization error: {e}")
            return ""
    
    def run_wilson_lin_chunking(self, papers_input_file: str) -> str:
        """Step 6: Apply Wilson Lin contextual chunking"""
        
        logger.info("🚀 STEP 6: Wilson Lin contextual chunking")
        
        try:
            # Use structure-aware chunker if structured data available
            structured_papers_file = self.corpus_dir / "structured_papers" / "structured_papers.parquet"
            
            if structured_papers_file.exists():
                logger.info("📄 Using structure-aware Wilson Lin chunking")
                
                cmd = [
                    'python', 'structure_aware_wilson_lin.py',
                    '--structured-papers', str(structured_papers_file)
                ]
                
                result = subprocess.run(cmd, cwd=self.corpus_dir.parent, capture_output=True, text=True)
                
                if result.returncode == 0:
                    chunks_file = self.corpus_dir / "structured_papers" / "structure_aware_chunks" / "structure_aware_wilson_lin_chunks.parquet"
                    if chunks_file.exists():
                        import pandas as pd
                        df = pd.read_parquet(chunks_file)
                        self.pipeline_stats['chunks_created'] = len(df)
                        logger.info(f"✅ Created {len(df)} structure-aware chunks")
                        return str(chunks_file)
            
            # Fallback to regular Wilson Lin chunking
            logger.info("📄 Using regular Wilson Lin chunking")
            
            processor = MultiSourceProcessor(str(self.corpus_dir))
            chunks_file = processor.run_wilson_lin_chunking(papers_input_file)
            
            if chunks_file:
                import pandas as pd
                df = pd.read_parquet(chunks_file)
                self.pipeline_stats['chunks_created'] = len(df)
                logger.info(f"✅ Created {len(df)} contextual chunks")
                return chunks_file
            else:
                logger.error("❌ Wilson Lin chunking failed")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Wilson Lin chunking error: {e}")
            return ""
    
    def run_ai_relevance_scoring(self, chunks_file: str) -> str:
        """Step 7: Apply AI relevance scoring"""
        
        logger.info("🚀 STEP 7: AI relevance scoring")
        
        try:
            # Use existing relevance scorer
            cmd = ['python', 'corpus_relevance_scorer.py', '--corpus-file', chunks_file]
            
            result = subprocess.run(cmd, cwd=self.corpus_dir.parent, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                # Look for scored output
                scored_dir = Path(chunks_file).parent / "scored_chunks"
                scored_file = scored_dir / "ai_relevance_scored_chunks.parquet" 
                
                if scored_file.exists():
                    logger.info("✅ AI relevance scoring completed")
                    return str(scored_file)
            
            logger.warning("⚠️ AI relevance scoring not available, using original chunks")
            return chunks_file
            
        except Exception as e:
            logger.warning(f"⚠️ AI relevance scoring error: {e}")
            return chunks_file

    def run_evidence_density_checks(self,
                                    threshold: float = 1.0,
                                    extractions_dir: str = ""):
        """Check evidence density of typed extractions"""

        logger.info("🚀 Running evidence density checks")

        from typed_paper_extractor import TypedPaperExtraction

        if not extractions_dir:
            extractions_dir = self.corpus_dir / 'typed_extractions' / 'validated'

        extraction_path = Path(extractions_dir)
        if not extraction_path.exists():
            logger.info("No typed extractions found; skipping evidence density check")
            return

        low_count = 0
        total = 0
        for json_file in extraction_path.glob('*.json'):
            total += 1
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                extraction = TypedPaperExtraction.parse_obj(data)
            except Exception as e:
                logger.warning(f"Failed to parse extraction {json_file}: {e}")
                continue

            num_claims = len(extraction.claims)
            valid_quotes = sum(len(c.evidence_spans) for c in extraction.claims)
            density = valid_quotes / num_claims if num_claims else 0.0

            if density < threshold:
                logger.warning(
                    f"Evidence density {density:.2f} below threshold for {extraction.paper_id}")
                low_count += 1

        self.pipeline_stats['low_evidence_papers'] = low_count
        logger.info(
            f"Evidence density check complete: {low_count} papers below threshold {threshold}")

    def run_citation_graph_checks(self,
                                  target_paper_ids: Optional[List[str]] = None,
                                  min_refs: int = 5):
        """Build citation graphs and mark low-info survey/tutorial nodes"""

        logger.info("🚀 Running citation graph checks")

        from citation_graph_builder import CitationGraphBuilder

        if target_paper_ids is None:
            target_paper_ids = []
            dedup_file = self.corpus_dir / 'deduplicated_papers.json'
            if dedup_file.exists():
                try:
                    with open(dedup_file, 'r') as f:
                        papers = json.load(f)
                    for paper in papers:
                        openalex_id = paper.get('openalex_id')
                        if openalex_id:
                            target_paper_ids.append(openalex_id)
                        if len(target_paper_ids) >= 1:
                            break
                except Exception as e:
                    logger.warning(f"Failed to load deduplicated papers: {e}")

        if not target_paper_ids:
            logger.info("No target papers available for citation graph check")
            return

        builder = CitationGraphBuilder()
        low_total = 0
        for pid in target_paper_ids:
            try:
                graph = asyncio.run(
                    builder.build_ego_graph(
                        pid,
                        hops=1,
                        include_citations=False,
                        include_references=True,
                        min_survey_refs=min_refs,
                    )
                )
                low_nodes = [n for n, d in graph.nodes(data=True) if d.get('low_info')]
                low_total += len(low_nodes)
            except Exception as e:
                logger.warning(f"Citation graph build failed for {pid}: {e}")

        self.pipeline_stats['low_info_survey_nodes'] = low_total
        logger.info(
            f"Citation graph check complete: {low_total} low-info survey/tutorial nodes found")
    
    def run_columnar_storage(self, chunks_file: str, references_file: str = "") -> bool:
        """Step 8: Store in columnar format with DuckDB"""
        
        logger.info("🚀 STEP 8: Columnar storage with DuckDB")
        
        try:
            storage = ColumnarStorageManager(str(self.final_output_dir / "columnar_storage"))
            
            # Ingest papers
            papers_file = self.corpus_dir / "deduplicated_papers.json"
            if papers_file.exists():
                # Convert JSON to parquet for ingestion
                import pandas as pd
                with open(papers_file, 'r') as f:
                    papers = json.load(f)
                
                papers_df = pd.DataFrame(papers)
                temp_papers_parquet = self.corpus_dir / "temp_papers.parquet"
                papers_df.to_parquet(temp_papers_parquet, index=False)
                
                storage.ingest_papers(str(temp_papers_parquet))
                temp_papers_parquet.unlink()  # Clean up
            
            # Ingest chunks
            if chunks_file and Path(chunks_file).exists():
                storage.ingest_chunks(chunks_file)
            
            # Ingest references
            if references_file and Path(references_file).exists():
                storage.ingest_references(references_file)
            
            # Build citation graph
            citations_count = storage.build_citation_graph()
            self.pipeline_stats['citations_mapped'] = citations_count
            
            # Export analysis datasets
            exports = storage.export_for_analysis(str(self.final_output_dir / "analysis_datasets"))
            
            # Get final statistics
            final_stats = storage.get_corpus_stats()
            
            storage.close()
            
            logger.info("✅ Columnar storage completed")
            logger.info(f"📊 Final corpus stats: {json.dumps(final_stats, indent=2)}")
            
            # Save pipeline completion stats
            self.pipeline_stats['completion_time'] = datetime.now().isoformat()
            self.pipeline_stats['final_corpus_stats'] = final_stats
            
            stats_file = self.final_output_dir / "pipeline_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.pipeline_stats, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Columnar storage error: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete corpus processing pipeline"""
        
        logger.info("="*60)
        logger.info("🚀 STARTING COMPLETE CORPUS PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Paper collection
            papers_file = self.run_paper_collection()
            if not papers_file:
                return False
            
            # Step 2: Deduplication
            deduplicated_file = self.run_deduplication(papers_file)
            if not deduplicated_file:
                return False
            
            # Step 3: PDF processing
            papers_input = self.run_pdf_processing()
            if not papers_input:
                return False
            
            # Step 4: Structured parsing (optional)
            self.run_structured_parsing()
            
            # Step 5: Reference normalization (optional)
            references_file = self.run_reference_normalization()
            
            # Step 6: Wilson Lin chunking
            chunks_file = self.run_wilson_lin_chunking(papers_input)
            if not chunks_file:
                return False
            
            # Step 7: AI relevance scoring (optional)
            scored_chunks = self.run_ai_relevance_scoring(chunks_file)

            # Quality checks
            self.run_evidence_density_checks()
            self.run_citation_graph_checks()

            # Step 8: Columnar storage
            success = self.run_columnar_storage(scored_chunks, references_file)
            
            if success:
                logger.info("="*60)
                logger.info("🎯 COMPLETE CORPUS PIPELINE FINISHED SUCCESSFULLY!")
                logger.info("="*60)
                logger.info(f"📊 Papers collected: {self.pipeline_stats['papers_collected']}")
                logger.info(f"📄 Papers deduplicated: {self.pipeline_stats['papers_deduplicated']}")  
                logger.info(f"📥 Papers downloaded: {self.pipeline_stats['papers_downloaded']}")
                logger.info(f"🔗 References normalized: {self.pipeline_stats['references_normalized']}")
                logger.info(f"📝 Chunks created: {self.pipeline_stats['chunks_created']}")
                logger.info(f"🔗 Citations mapped: {self.pipeline_stats['citations_mapped']}")
                logger.info(f"📉 Low evidence papers: {self.pipeline_stats['low_evidence_papers']}")
                logger.info(f"📚 Low-info survey nodes: {self.pipeline_stats['low_info_survey_nodes']}")
                logger.info(f"💾 Output directory: {self.final_output_dir}")
                logger.info("="*60)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete corpus processing pipeline")
    parser.add_argument("--corpus-dir", default="/home/ubuntu/LW_scrape/multi_source_corpus")
    parser.add_argument("--output-dir", default="/home/ubuntu/LW_scrape/final_corpus_v2")
    parser.add_argument("--openalex-pages", type=int, default=5, help="OpenAlex pages to collect")
    parser.add_argument("--arxiv-results", type=int, default=500, help="ArXiv papers to collect")
    parser.add_argument("--openreview-limit", type=int, default=100, help="OpenReview papers per venue")
    
    args = parser.parse_args()
    
    pipeline = CompleteCorpusPipeline(args.corpus_dir, args.output_dir)
    
    success = pipeline.run_complete_pipeline()
    
    if success:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Final corpus location: {args.output_dir}")
        print(f"🔍 Query with: python columnar_storage_manager.py --stats")
    else:
        print(f"\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()