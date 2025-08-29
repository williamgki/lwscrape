#!/usr/bin/env python3
"""
Structured PDF Parser
Uses GROBID and pdffigures2 to extract structured content from academic PDFs
"""

import json
import os
import subprocess
import logging
import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ParsedPaper:
    """Structured representation of parsed paper"""
    doc_id: str
    title: str
    abstract: str
    authors: List[str]
    sections: List[Dict]
    references: List[Dict]
    figures: List[Dict]
    tables: List[Dict]
    metadata: Dict
    full_text: str
    tei_xml_path: Optional[str] = None
    pdffigures_json_path: Optional[str] = None

class GROBIDClient:
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        self.base_url = grobid_url
        self.session = requests.Session()
        
    def check_service(self) -> bool:
        """Check if GROBID service is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/isalive", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GROBID service not available: {e}")
            return False
    
    def process_fulltext(self, pdf_path: Path, output_dir: Path) -> Optional[Path]:
        """Process PDF with GROBID fulltext extraction"""
        
        if not self.check_service():
            logger.error("GROBID service not available")
            return None
        
        output_file = output_dir / f"{pdf_path.stem}.tei.xml"
        
        # Skip if already processed
        if output_file.exists():
            logger.debug(f"✅ Already processed: {pdf_path.name}")
            return output_file
        
        try:
            with open(pdf_path, 'rb') as pdf_file:
                files = {
                    'input': (pdf_path.name, pdf_file, 'application/pdf')
                }
                
                data = {
                    'consolidateHeader': '1',
                    'consolidateCitations': '1', 
                    'includeRawCitations': '1',
                    'includeRawAffiliations': '1'
                }
                
                logger.info(f"🔄 Processing with GROBID: {pdf_path.name}")
                
                response = self.session.post(
                    f"{self.base_url}/api/processFulltextDocument",
                    files=files,
                    data=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    logger.info(f"✅ GROBID processed: {output_file.name}")
                    return output_file
                else:
                    logger.error(f"❌ GROBID failed for {pdf_path.name}: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ GROBID processing failed for {pdf_path.name}: {e}")
            return None

class PDFFigures2Client:
    def __init__(self, pdffigures2_jar: str):
        self.jar_path = Path(pdffigures2_jar)
        if not self.jar_path.exists():
            logger.error(f"pdffigures2 JAR not found: {pdffigures2_jar}")
    
    def process_batch(self, input_dir: Path, json_output_dir: Path, img_output_dir: Path) -> bool:
        """Process batch of PDFs with pdffigures2"""
        
        json_output_dir.mkdir(exist_ok=True)
        img_output_dir.mkdir(exist_ok=True)
        
        try:
            cmd = [
                'java', '-jar', str(self.jar_path),
                '-g',  # Add section titles and text dump
                '-d', str(json_output_dir),  # JSON output directory
                '-m', str(img_output_dir),   # Image output directory
                str(input_dir)  # Input PDF directory
            ]
            
            logger.info(f"🔄 Running pdffigures2 batch processing...")
            logger.info(f"   Input: {input_dir}")
            logger.info(f"   JSON output: {json_output_dir}")
            logger.info(f"   Images: {img_output_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ pdffigures2 batch processing completed")
                return True
            else:
                logger.error(f"❌ pdffigures2 failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ pdffigures2 timed out")
            return False
        except Exception as e:
            logger.error(f"❌ pdffigures2 processing failed: {e}")
            return False

class TEIParser:
    """Parse GROBID TEI XML output"""
    
    def __init__(self):
        # TEI namespace
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    def parse_tei_file(self, tei_path: Path) -> Dict:
        """Parse TEI XML file into structured data"""
        
        try:
            tree = ET.parse(tei_path)
            root = tree.getroot()
            
            # Extract title
            title = self.extract_title(root)
            
            # Extract abstract
            abstract = self.extract_abstract(root)
            
            # Extract authors
            authors = self.extract_authors(root)
            
            # Extract sections
            sections = self.extract_sections(root)
            
            # Extract references
            references = self.extract_references(root)
            
            # Extract full text
            full_text = self.extract_full_text(root)
            
            return {
                'title': title,
                'abstract': abstract, 
                'authors': authors,
                'sections': sections,
                'references': references,
                'full_text': full_text,
                'tei_xml_path': str(tei_path)
            }
            
        except Exception as e:
            logger.error(f"Error parsing TEI file {tei_path}: {e}")
            return {}
    
    def extract_title(self, root) -> str:
        """Extract paper title"""
        title_elem = root.find('.//tei:titleStmt/tei:title', self.ns)
        return title_elem.text.strip() if title_elem is not None and title_elem.text else ""
    
    def extract_abstract(self, root) -> str:
        """Extract abstract"""
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', self.ns)
        if abstract_elem is not None:
            # Join all text within abstract
            abstract_text = []
            for p in abstract_elem.findall('.//tei:p', self.ns):
                if p.text:
                    abstract_text.append(p.text.strip())
            return ' '.join(abstract_text)
        return ""
    
    def extract_authors(self, root) -> List[str]:
        """Extract author names"""
        authors = []
        
        for author in root.findall('.//tei:sourceDesc//tei:author', self.ns):
            name_parts = []
            
            # First name
            first = author.find('.//tei:forename[@type="first"]', self.ns)
            if first is not None and first.text:
                name_parts.append(first.text.strip())
            
            # Middle name
            middle = author.find('.//tei:forename[@type="middle"]', self.ns)
            if middle is not None and middle.text:
                name_parts.append(middle.text.strip())
            
            # Last name
            last = author.find('.//tei:surname', self.ns)
            if last is not None and last.text:
                name_parts.append(last.text.strip())
            
            if name_parts:
                authors.append(' '.join(name_parts))
        
        return authors
    
    def extract_sections(self, root) -> List[Dict]:
        """Extract document sections with hierarchy"""
        sections = []
        
        body = root.find('.//tei:text/tei:body', self.ns)
        if body is not None:
            for i, div in enumerate(body.findall('.//tei:div', self.ns)):
                section = self.parse_section(div, level=1)
                if section:
                    section['section_id'] = i
                    sections.append(section)
        
        return sections
    
    def parse_section(self, div_elem, level: int = 1) -> Optional[Dict]:
        """Parse individual section"""
        
        # Extract section title
        head = div_elem.find('./tei:head', self.ns)
        title = head.text.strip() if head is not None and head.text else f"Section {level}"
        
        # Extract section text
        paragraphs = []
        for p in div_elem.findall('./tei:p', self.ns):
            if p.text:
                paragraphs.append(p.text.strip())
        
        text_content = '\n'.join(paragraphs)
        
        # Extract subsections
        subsections = []
        for subdiv in div_elem.findall('./tei:div', self.ns):
            subsection = self.parse_section(subdiv, level + 1)
            if subsection:
                subsections.append(subsection)
        
        if text_content or subsections:
            return {
                'title': title,
                'level': level,
                'content': text_content,
                'subsections': subsections,
                'word_count': len(text_content.split()) if text_content else 0
            }
        
        return None
    
    def extract_references(self, root) -> List[Dict]:
        """Extract bibliography references"""
        references = []
        
        back = root.find('.//tei:text/tei:back', self.ns)
        if back is not None:
            for i, bibl in enumerate(back.findall('.//tei:biblStruct', self.ns)):
                ref = self.parse_reference(bibl)
                if ref:
                    ref['ref_id'] = i
                    references.append(ref)
        
        return references
    
    def parse_reference(self, bibl_elem) -> Optional[Dict]:
        """Parse individual reference"""
        
        # Extract title
        title_elem = bibl_elem.find('.//tei:title[@level="a"]', self.ns)
        if title_elem is None:
            title_elem = bibl_elem.find('.//tei:title', self.ns)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        
        # Extract authors
        authors = []
        for author in bibl_elem.findall('.//tei:author', self.ns):
            name_parts = []
            for name_elem in author.findall('.//tei:forename', self.ns):
                if name_elem.text:
                    name_parts.append(name_elem.text.strip())
            surname_elem = author.find('.//tei:surname', self.ns)
            if surname_elem is not None and surname_elem.text:
                name_parts.append(surname_elem.text.strip())
            if name_parts:
                authors.append(' '.join(name_parts))
        
        # Extract journal/venue
        journal_elem = bibl_elem.find('.//tei:title[@level="j"]', self.ns)
        venue = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else ""
        
        # Extract year
        date_elem = bibl_elem.find('.//tei:date[@type="published"]', self.ns)
        year = date_elem.get('when', '') if date_elem is not None else ""
        if year:
            year = year[:4]  # Extract just the year
        
        if title or authors:
            return {
                'title': title,
                'authors': authors,
                'venue': venue,
                'year': year,
                'raw_text': ET.tostring(bibl_elem, encoding='unicode', method='text')
            }
        
        return None
    
    def extract_full_text(self, root) -> str:
        """Extract full text content"""
        text_parts = []
        
        # Extract from body
        body = root.find('.//tei:text/tei:body', self.ns)
        if body is not None:
            for p in body.findall('.//tei:p', self.ns):
                if p.text:
                    text_parts.append(p.text.strip())
        
        return '\n\n'.join(text_parts)

class StructuredPDFParser:
    def __init__(self, corpus_dir: str = "/home/ubuntu/LW_scrape/multi_source_corpus"):
        self.corpus_dir = Path(corpus_dir)
        
        # Setup directories
        self.pdf_dir = self.corpus_dir / "downloaded_papers"
        self.grobid_output = self.corpus_dir / "grobid_tei"
        self.pdffigures_json = self.corpus_dir / "pdffigures2_json"
        self.pdffigures_imgs = self.corpus_dir / "pdffigures2_imgs"
        self.structured_output = self.corpus_dir / "structured_papers"
        
        for dir_path in [self.grobid_output, self.pdffigures_json, self.pdffigures_imgs, self.structured_output]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize clients
        self.grobid = GROBIDClient()
        self.tei_parser = TEIParser()
        
        # Find pdffigures2 jar
        jar_paths = [
            "/home/ubuntu/pdffigures2/target/scala-2.12/pdffigures2-assembly.jar",
            "./pdffigures2/target/scala-2.12/pdffigures2-assembly.jar",
            "/usr/local/bin/pdffigures2-assembly.jar"
        ]
        
        self.pdffigures2_jar = None
        for jar_path in jar_paths:
            if Path(jar_path).exists():
                self.pdffigures2_jar = jar_path
                break
        
        if self.pdffigures2_jar:
            self.pdffigures2 = PDFFigures2Client(self.pdffigures2_jar)
        else:
            logger.warning("⚠️ pdffigures2 JAR not found. Figure extraction will be skipped.")
            self.pdffigures2 = None
    
    def setup_grobid_service(self) -> bool:
        """Setup GROBID Docker service"""
        
        logger.info("🐳 Setting up GROBID service...")
        
        try:
            # Check if GROBID is already running
            if self.grobid.check_service():
                logger.info("✅ GROBID service already running")
                return True
            
            # Start GROBID Docker container
            cmd = [
                'docker', 'run', '--rm', '--init', '-d',
                '-p', '8070:8070',
                '--name', 'grobid-service',
                'grobid/grobid:latest'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("🐳 GROBID Docker container started")
                
                # Wait for service to be ready
                for i in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    if self.grobid.check_service():
                        logger.info("✅ GROBID service ready")
                        return True
                    
                logger.error("❌ GROBID service failed to start")
                return False
            else:
                logger.error(f"❌ Failed to start GROBID Docker: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ GROBID setup failed: {e}")
            return False
    
    def setup_pdffigures2(self) -> bool:
        """Setup pdffigures2 if not available"""
        
        if self.pdffigures2 is not None:
            logger.info("✅ pdffigures2 already available")
            return True
        
        logger.info("🔧 Setting up pdffigures2...")
        
        try:
            # Clone repository
            if not Path("pdffigures2").exists():
                result = subprocess.run([
                    'git', 'clone', 'https://github.com/allenai/pdffigures2'
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"❌ Failed to clone pdffigures2: {result.stderr}")
                    return False
            
            # Build with SBT
            os.chdir('pdffigures2')
            
            result = subprocess.run(['sbt', 'assembly'], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                jar_path = "target/scala-2.12/pdffigures2-assembly.jar"
                if Path(jar_path).exists():
                    self.pdffigures2_jar = str(Path.cwd() / jar_path)
                    self.pdffigures2 = PDFFigures2Client(self.pdffigures2_jar)
                    logger.info("✅ pdffigures2 built successfully")
                    os.chdir('..')
                    return True
                else:
                    logger.error("❌ pdffigures2 JAR not found after build")
                    os.chdir('..')
                    return False
            else:
                logger.error(f"❌ pdffigures2 build failed: {result.stderr}")
                os.chdir('..')
                return False
                
        except Exception as e:
            logger.error(f"❌ pdffigures2 setup failed: {e}")
            if Path.cwd().name == 'pdffigures2':
                os.chdir('..')
            return False
    
    def process_papers_batch(self) -> List[ParsedPaper]:
        """Process all PDFs with both GROBID and pdffigures2"""
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"📚 Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            logger.warning("No PDF files found")
            return []
        
        # Process with GROBID
        logger.info("🔄 Processing with GROBID...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            grobid_futures = {
                executor.submit(self.grobid.process_fulltext, pdf_file, self.grobid_output): pdf_file 
                for pdf_file in pdf_files
            }
            
            grobid_results = {}
            for future in concurrent.futures.as_completed(grobid_futures):
                pdf_file = grobid_futures[future]
                try:
                    tei_file = future.result()
                    if tei_file:
                        grobid_results[pdf_file.stem] = tei_file
                except Exception as e:
                    logger.error(f"GROBID processing failed for {pdf_file.name}: {e}")
        
        # Process with pdffigures2
        pdffigures_results = {}
        if self.pdffigures2:
            logger.info("🔄 Processing with pdffigures2...")
            if self.pdffigures2.process_batch(self.pdf_dir, self.pdffigures_json, self.pdffigures_imgs):
                # Load pdffigures2 results
                for json_file in self.pdffigures_json.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            pdffigures_data = json.load(f)
                        pdffigures_results[json_file.stem] = pdffigures_data
                    except Exception as e:
                        logger.error(f"Error loading pdffigures2 result {json_file}: {e}")
        
        # Combine results
        parsed_papers = []
        for pdf_file in pdf_files:
            logger.info(f"📄 Combining results for: {pdf_file.name}")
            
            parsed_paper = self.combine_parsing_results(
                pdf_file,
                grobid_results.get(pdf_file.stem),
                pdffigures_results.get(pdf_file.stem)
            )
            
            if parsed_paper:
                parsed_papers.append(parsed_paper)
        
        return parsed_papers
    
    def combine_parsing_results(self, pdf_file: Path, tei_file: Optional[Path], pdffigures_data: Optional[Dict]) -> Optional[ParsedPaper]:
        """Combine GROBID and pdffigures2 results"""
        
        doc_id = pdf_file.stem
        
        # Parse GROBID results
        grobid_data = {}
        if tei_file:
            grobid_data = self.tei_parser.parse_tei_file(tei_file)
        
        # Extract figures and tables from pdffigures2
        figures = []
        tables = []
        if pdffigures_data:
            for fig in pdffigures_data.get('figures', []):
                figures.append({
                    'figure_id': fig.get('name', ''),
                    'caption': fig.get('caption', ''),
                    'page': fig.get('page', 0),
                    'bbox': fig.get('regionBoundary', {}),
                    'figure_type': fig.get('figType', 'Figure')
                })
                
                if fig.get('figType') == 'Table':
                    tables.append({
                        'table_id': fig.get('name', ''),
                        'caption': fig.get('caption', ''),
                        'page': fig.get('page', 0),
                        'bbox': fig.get('regionBoundary', {})
                    })
        
        # Create parsed paper
        try:
            parsed_paper = ParsedPaper(
                doc_id=doc_id,
                title=grobid_data.get('title', pdf_file.stem),
                abstract=grobid_data.get('abstract', ''),
                authors=grobid_data.get('authors', []),
                sections=grobid_data.get('sections', []),
                references=grobid_data.get('references', []),
                figures=figures,
                tables=tables,
                metadata={
                    'pdf_file': str(pdf_file),
                    'processing_timestamp': datetime.now().isoformat(),
                    'grobid_processed': tei_file is not None,
                    'pdffigures2_processed': pdffigures_data is not None,
                    'section_count': len(grobid_data.get('sections', [])),
                    'reference_count': len(grobid_data.get('references', [])),
                    'figure_count': len(figures),
                    'table_count': len(tables)
                },
                full_text=grobid_data.get('full_text', ''),
                tei_xml_path=str(tei_file) if tei_file else None,
                pdffigures_json_path=str(self.pdffigures_json / f"{doc_id}.json") if pdffigures_data else None
            )
            
            return parsed_paper
            
        except Exception as e:
            logger.error(f"Error creating parsed paper for {doc_id}: {e}")
            return None
    
    def save_structured_papers(self, parsed_papers: List[ParsedPaper]) -> str:
        """Save structured papers to JSON and parquet"""
        
        # Convert to serializable format
        papers_data = []
        for paper in parsed_papers:
            paper_dict = {
                'doc_id': paper.doc_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'sections': paper.sections,
                'references': paper.references,
                'figures': paper.figures,
                'tables': paper.tables,
                'metadata': paper.metadata,
                'full_text': paper.full_text[:10000],  # Truncate for storage
                'tei_xml_path': paper.tei_xml_path,
                'pdffigures_json_path': paper.pdffigures_json_path
            }
            papers_data.append(paper_dict)
        
        # Save as JSON
        json_file = self.structured_output / "structured_papers.json"
        with open(json_file, 'w') as f:
            json.dump(papers_data, f, indent=2, default=str)
        
        # Save as parquet for faster loading
        parquet_file = self.structured_output / "structured_papers.parquet"
        df = pd.DataFrame(papers_data)
        df.to_parquet(parquet_file, index=False)
        
        # Generate statistics
        stats = {
            'total_papers': len(parsed_papers),
            'papers_with_grobid': len([p for p in parsed_papers if p.tei_xml_path]),
            'papers_with_pdffigures2': len([p for p in parsed_papers if p.pdffigures_json_path]),
            'total_sections': sum(len(p.sections) for p in parsed_papers),
            'total_references': sum(len(p.references) for p in parsed_papers),
            'total_figures': sum(len(p.figures) for p in parsed_papers),
            'total_tables': sum(len(p.tables) for p in parsed_papers),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        stats_file = self.structured_output / "parsing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"💾 Saved {len(parsed_papers)} structured papers:")
        logger.info(f"   JSON: {json_file}")
        logger.info(f"   Parquet: {parquet_file}")
        logger.info(f"   Stats: {stats_file}")
        
        return str(parquet_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Structured PDF parsing with GROBID and pdffigures2")
    parser.add_argument("--corpus-dir", default="/home/ubuntu/LW_scrape/multi_source_corpus")
    parser.add_argument("--setup-services", action="store_true", help="Setup GROBID and pdffigures2")
    parser.add_argument("--skip-grobid", action="store_true", help="Skip GROBID processing")
    parser.add_argument("--skip-pdffigures2", action="store_true", help="Skip pdffigures2 processing")
    
    args = parser.parse_args()
    
    parser = StructuredPDFParser(args.corpus_dir)
    
    # Setup services if requested
    if args.setup_services:
        if not args.skip_grobid:
            parser.setup_grobid_service()
        if not args.skip_pdffigures2:
            parser.setup_pdffigures2()
        return
    
    # Process papers
    logger.info("🚀 Starting structured PDF parsing...")
    
    # Check services
    grobid_ready = parser.grobid.check_service() if not args.skip_grobid else False
    pdffigures2_ready = parser.pdffigures2 is not None if not args.skip_pdffigures2 else False
    
    logger.info(f"📊 Service status:")
    logger.info(f"   GROBID: {'✅ Ready' if grobid_ready else '❌ Not available'}")
    logger.info(f"   pdffigures2: {'✅ Ready' if pdffigures2_ready else '❌ Not available'}")
    
    if not grobid_ready and not pdffigures2_ready:
        logger.error("No parsing services available. Run with --setup-services first.")
        return
    
    # Process papers
    parsed_papers = parser.process_papers_batch()
    
    # Save results
    if parsed_papers:
        output_file = parser.save_structured_papers(parsed_papers)
        
        logger.info("="*60)
        logger.info("🎯 STRUCTURED PARSING COMPLETE")
        logger.info("="*60)
        logger.info(f"📊 Papers processed: {len(parsed_papers)}")
        logger.info(f"💾 Output: {output_file}")
        logger.info("="*60)
    else:
        logger.error("No papers were successfully parsed")

if __name__ == "__main__":
    main()