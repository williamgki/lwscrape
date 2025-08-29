#!/usr/bin/env python3

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import xml.etree.ElementTree as ET
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for strict JSON schema validation
class EvidenceSpan(BaseModel):
    page: int = Field(..., ge=0, description="Page number from PDF viewer")
    quote: str = Field(..., max_length=300, description="Verbatim quote from paper")
    span_hint: Optional[str] = Field(None, description="Character position or bbox hint")

class Claim(BaseModel):
    id: str = Field(..., pattern=r"^C\d+$", description="Claim ID (C1, C2, ...)")
    text: str = Field(..., max_length=500, description="Atomic, falsifiable claim")
    type: str = Field(..., regex=r"^(theoretical|empirical|engineering)$")
    section: str = Field(..., description="Section where claim appears")
    page: int = Field(..., ge=0)
    evidence_spans: List[EvidenceSpan] = Field(..., min_items=1)
    status: str = Field(default="VALID", regex=r"^(VALID|SPECULATIVE)$", description="Validation status")

class Assumption(BaseModel):
    id: str = Field(..., pattern=r"^A\d+$", description="Assumption ID (A1, A2, ...)")
    text: str = Field(..., max_length=400, description="Assumption statement")
    page: int = Field(..., ge=0)
    evidence_spans: List[EvidenceSpan] = Field(..., min_items=1)
    status: str = Field(default="VALID", regex=r"^(VALID|SPECULATIVE)$", description="Validation status")

class Mechanism(BaseModel):
    id: str = Field(..., pattern=r"^M\d+$", description="Mechanism ID (M1, M2, ...)")
    text: str = Field(..., max_length=400, description="How the method works")
    page: int = Field(..., ge=0)
    evidence_spans: List[EvidenceSpan] = Field(..., min_items=1)
    status: str = Field(default="VALID", regex=r"^(VALID|SPECULATIVE)$", description="Validation status")

class Metric(BaseModel):
    id: str = Field(..., pattern=r"^X\d+$", description="Metric ID (X1, X2, ...)")
    name: str = Field(..., max_length=100, description="Metric name")
    definition: str = Field(..., max_length=400, description="Formula or procedure")
    page: int = Field(..., ge=0)
    evidence_spans: List[EvidenceSpan] = Field(..., min_items=1)
    status: str = Field(default="VALID", regex=r"^(VALID|SPECULATIVE)$", description="Validation status")

class Experiment(BaseModel):
    id: str = Field(..., pattern=r"^E\d+$", description="Experiment ID (E1, E2, ...)")
    setup: str = Field(..., max_length=300, description="Experimental setup")
    dataset: str = Field(..., max_length=100, description="Dataset used")
    ablation_gaps: List[str] = Field(default_factory=list, description="Missing ablations")
    page: int = Field(..., ge=0)
    evidence_spans: List[EvidenceSpan] = Field(..., min_items=1)
    status: str = Field(default="VALID", regex=r"^(VALID|SPECULATIVE)$", description="Validation status")

class ThreatModel(BaseModel):
    id: str = Field(..., pattern=r"^T\d+$", description="Threat model ID (T1, T2, ...)")
    text: str = Field(..., max_length=400, description="Attacker capabilities + goals")
    page: int = Field(..., ge=0)
    evidence_spans: List[EvidenceSpan] = Field(..., min_items=1)
    status: str = Field(default="VALID", regex=r"^(VALID|SPECULATIVE)$", description="Validation status")

class TypedPaperExtraction(BaseModel):
    paper_id: str = Field(..., description="Paper identifier")
    claims: List[Claim] = Field(default_factory=list)
    assumptions: List[Assumption] = Field(default_factory=list)
    mechanisms: List[Mechanism] = Field(default_factory=list)
    metrics: List[Metric] = Field(default_factory=list)
    experiments: List[Experiment] = Field(default_factory=list)
    threat_models: List[ThreatModel] = Field(default_factory=list)


@dataclass
class GROBIDSection:
    """Parsed GROBID TEI section"""
    title: str
    content: str
    level: int  # 1 for main sections, 2 for subsections, etc.
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    tei_type: Optional[str] = None  # abstract, introduction, etc.


@dataclass
class PaperStructure:
    """Complete paper structure from GROBID + pdffigures2"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    sections: List[GROBIDSection]
    page_count: int
    tei_path: Optional[str] = None
    pdf_path: Optional[str] = None


class GROBIDTEIParser:
    """Parse GROBID TEI XML for deterministic segmentation"""
    
    def __init__(self):
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    def parse_paper_structure(self, tei_file: Path, paper_id: str) -> PaperStructure:
        """Parse complete paper structure from GROBID TEI"""
        
        tree = ET.parse(tei_file)
        root = tree.getroot()
        
        # Extract title
        title = self._extract_title(root)
        
        # Extract abstract
        abstract = self._extract_abstract(root)
        
        # Extract authors
        authors = self._extract_authors(root)
        
        # Extract section tree
        sections = self._extract_section_tree(root)
        
        # Estimate page count (would be better with actual PDF info)
        page_count = self._estimate_page_count(root)
        
        return PaperStructure(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            sections=sections,
            page_count=page_count,
            tei_path=str(tei_file)
        )
    
    def _extract_title(self, root) -> str:
        """Extract paper title from TEI"""
        title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', self.ns)
        if title_elem is None:
            title_elem = root.find('.//tei:title', self.ns)
        
        return self._clean_text(title_elem.text if title_elem is not None else "")
    
    def _extract_abstract(self, root) -> str:
        """Extract abstract from TEI"""
        abstract_div = root.find('.//tei:div[@type="abstract"]', self.ns)
        if abstract_div is not None:
            abstract_text = ET.tostring(abstract_div, encoding='unicode', method='text')
            return self._clean_text(abstract_text)
        return ""
    
    def _extract_authors(self, root) -> List[str]:
        """Extract author names from TEI"""
        authors = []
        
        author_elems = root.findall('.//tei:author', self.ns)
        for author in author_elems:
            name_elem = author.find('.//tei:persName', self.ns)
            if name_elem is not None:
                # Try to get structured name
                forename = name_elem.find('tei:forename', self.ns)
                surname = name_elem.find('tei:surname', self.ns)
                
                if forename is not None and surname is not None:
                    full_name = f"{forename.text} {surname.text}"
                else:
                    full_name = self._clean_text(name_elem.text or "")
                
                if full_name.strip():
                    authors.append(full_name.strip())
        
        return authors
    
    def _extract_section_tree(self, root) -> List[GROBIDSection]:
        """Extract hierarchical section structure"""
        sections = []
        
        # Find all div elements with type or head elements
        div_elements = root.findall('.//tei:div', self.ns)
        
        for div in div_elements:
            section = self._parse_div_element(div)
            if section:
                sections.append(section)
        
        return sections
    
    def _parse_div_element(self, div) -> Optional[GROBIDSection]:
        """Parse individual div element into section"""
        
        # Get section title
        head_elem = div.find('.//tei:head', self.ns)
        if head_elem is None:
            return None
        
        title = self._clean_text(head_elem.text or "")
        if not title.strip():
            return None
        
        # Get section content (exclude nested divs)
        content_parts = []
        for p in div.findall('./tei:p', self.ns):  # Direct paragraphs only
            p_text = ET.tostring(p, encoding='unicode', method='text')
            content_parts.append(self._clean_text(p_text))
        
        content = "\n\n".join(content_parts)
        
        # Determine section level from nesting or numbering
        level = self._determine_section_level(title, div)
        
        # Get section type if available
        tei_type = div.get('type')
        
        return GROBIDSection(
            title=title,
            content=content,
            level=level,
            tei_type=tei_type
        )
    
    def _determine_section_level(self, title: str, div_elem) -> int:
        """Determine section hierarchy level"""
        
        # Check for numbered sections (1., 1.1, 1.1.1, etc.)
        number_match = re.match(r'^(\d+\.)+', title.strip())
        if number_match:
            dots = number_match.group().count('.')
            return dots
        
        # Fall back to nesting level in XML
        level = 1
        parent = div_elem.getparent()
        while parent is not None and parent.tag.endswith('div'):
            level += 1
            parent = parent.getparent()
        
        return min(level, 4)  # Cap at 4 levels
    
    def _estimate_page_count(self, root) -> int:
        """Estimate page count from content"""
        # This is a rough estimate - would be better with actual PDF info
        all_text = ET.tostring(root, encoding='unicode', method='text')
        # Rough estimate: 3000 chars per page
        estimated_pages = max(1, len(all_text) // 3000)
        return estimated_pages
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


class TypedObjectExtractor:
    """Extract typed objects from paper sections using LLM"""
    
    def __init__(self, api_client=None, model_name: str = "claude-3-5-sonnet-20241022"):
        self.api_client = api_client
        self.model_name = model_name
        
        # JSON schema for validation
        self.json_schema = TypedPaperExtraction.schema()
        
        # Extraction prompt template
        self.extraction_prompt = """You are extracting typed objects from an academic PDF. Only output valid JSON matching the provided schema.

Rules:
• Use only verbatim quotes (≤300 chars) from the paper and include the page number from the PDF viewer.
• If you are unsure, omit the item. Do not summarize.
• Claims must be atomic, falsifiable statements the authors assert.
• Mechanisms describe how the method is supposed to work.
• Metrics must include a definition (formula or procedure).
• Experiments must include setup + dataset + any missing ablations in ablation_gaps.
• Threat models specify attacker/agent capabilities + goals.
• Every item must include at least one evidence_spans with a quote and page.

Paper: {paper_title}
Section: {section_title}
Content: {section_content}

Output valid JSON only."""
    
    def extract_from_section(self, 
                           paper_structure: PaperStructure,
                           section: GROBIDSection) -> Dict[str, Any]:
        """Extract typed objects from a single section"""
        
        if not self.api_client:
            logger.warning("No API client provided. Returning empty extraction.")
            return {"claims": [], "assumptions": [], "mechanisms": [], 
                   "metrics": [], "experiments": [], "threat_models": []}
        
        # Prepare prompt
        prompt = self.extraction_prompt.format(
            paper_title=paper_structure.title,
            section_title=section.title,
            section_content=section.content[:4000]  # Limit content length
        )
        
        try:
            # Make API call with JSON mode
            response = self.api_client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0,  # Deterministic extraction
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse JSON response
            response_text = response.content[0].text
            extracted_data = json.loads(response_text)
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Extraction failed for section '{section.title}': {e}")
            return {"claims": [], "assumptions": [], "mechanisms": [], 
                   "metrics": [], "experiments": [], "threat_models": []}
    
    def extract_full_paper(self, paper_structure: PaperStructure) -> TypedPaperExtraction:
        """Extract typed objects from entire paper"""
        
        logger.info(f"Extracting typed objects from paper: {paper_structure.title}")
        
        # Initialize extraction
        extraction = TypedPaperExtraction(paper_id=paper_structure.paper_id)
        
        # Process each section
        for section in paper_structure.sections:
            if not section.content.strip():
                continue
            
            logger.info(f"Processing section: {section.title}")
            
            section_extraction = self.extract_from_section(paper_structure, section)
            
            # Merge section results into full extraction
            self._merge_section_extraction(extraction, section_extraction, section)
        
        return extraction
    
    def _merge_section_extraction(self, 
                                 full_extraction: TypedPaperExtraction,
                                 section_data: Dict[str, Any],
                                 section: GROBIDSection):
        """Merge section extraction into full paper extraction"""
        
        # Add section context to extracted items
        for claim_data in section_data.get('claims', []):
            try:
                claim = Claim(**claim_data)
                claim.section = section.title
                full_extraction.claims.append(claim)
            except Exception as e:
                logger.warning(f"Invalid claim data: {e}")
        
        for assumption_data in section_data.get('assumptions', []):
            try:
                assumption = Assumption(**assumption_data)
                full_extraction.assumptions.append(assumption)
            except Exception as e:
                logger.warning(f"Invalid assumption data: {e}")
        
        for mechanism_data in section_data.get('mechanisms', []):
            try:
                mechanism = Mechanism(**mechanism_data)
                full_extraction.mechanisms.append(mechanism)
            except Exception as e:
                logger.warning(f"Invalid mechanism data: {e}")
        
        for metric_data in section_data.get('metrics', []):
            try:
                metric = Metric(**metric_data)
                full_extraction.metrics.append(metric)
            except Exception as e:
                logger.warning(f"Invalid metric data: {e}")
        
        for experiment_data in section_data.get('experiments', []):
            try:
                experiment = Experiment(**experiment_data)
                full_extraction.experiments.append(experiment)
            except Exception as e:
                logger.warning(f"Invalid experiment data: {e}")
        
        for threat_data in section_data.get('threat_models', []):
            try:
                threat_model = ThreatModel(**threat_data)
                full_extraction.threat_models.append(threat_model)
            except Exception as e:
                logger.warning(f"Invalid threat model data: {e}")


class ExtractionValidator:
    """Post-processing validator for extracted objects"""
    
    def __init__(self, paper_structure: PaperStructure):
        self.paper_structure = paper_structure
        self.page_contents = self._build_page_index()
    
    def _build_page_index(self) -> Dict[int, str]:
        """Build index of content by page"""
        # This would ideally use actual PDF OCR/GROBID page mapping
        # For now, create simple page distribution
        page_contents = {}
        
        total_content = ""
        for section in self.paper_structure.sections:
            total_content += section.content + "\n\n"
        
        # Rough page distribution
        chars_per_page = len(total_content) // max(self.paper_structure.page_count, 1)
        
        for page_num in range(1, self.paper_structure.page_count + 1):
            start_idx = (page_num - 1) * chars_per_page
            end_idx = page_num * chars_per_page
            page_contents[page_num] = total_content[start_idx:end_idx]
        
        return page_contents
    
    def validate_extraction(self, extraction: TypedPaperExtraction) -> TypedPaperExtraction:
        """Validate and clean extraction"""
        
        validated_extraction = TypedPaperExtraction(paper_id=extraction.paper_id)
        
        # Validate each category
        validated_extraction.claims = self._validate_items(extraction.claims, "claim")
        validated_extraction.assumptions = self._validate_items(extraction.assumptions, "assumption")
        validated_extraction.mechanisms = self._validate_items(extraction.mechanisms, "mechanism")
        validated_extraction.metrics = self._validate_items(extraction.metrics, "metric")
        validated_extraction.experiments = self._validate_items(extraction.experiments, "experiment")
        validated_extraction.threat_models = self._validate_items(extraction.threat_models, "threat_model")
        
        return validated_extraction
    
    def _validate_items(self, items: List, item_type: str) -> List:
        """Validate list of items"""
        validated_items = []
        
        for item in items:
            if self._validate_item(item, item_type):
                validated_items.append(item)
            else:
                logger.warning(f"Rejecting invalid {item_type}: {getattr(item, 'id', 'unknown')}")
        
        return validated_items
    
    def _validate_item(self, item, item_type: str) -> bool:
        """Validate individual item - mark as SPECULATIVE if invalid evidence"""
        
        # Check evidence spans
        if not hasattr(item, 'evidence_spans') or not item.evidence_spans:
            logger.warning(f"{item_type} {getattr(item, 'id', 'unknown')} has no evidence spans - marking SPECULATIVE")
            item.status = "SPECULATIVE"
            return True  # Keep item but mark as speculative
        
        # Validate each evidence span
        valid_spans = 0
        for span in item.evidence_spans:
            if self._validate_evidence_span(span):
                valid_spans += 1
        
        # If no valid evidence spans, mark as speculative
        if valid_spans == 0:
            logger.warning(f"{item_type} {getattr(item, 'id', 'unknown')} has no valid evidence - marking SPECULATIVE")
            item.status = "SPECULATIVE"
            return True  # Keep item but mark as speculative
        
        # Check page validity
        if hasattr(item, 'page'):
            if item.page < 1 or item.page > self.paper_structure.page_count:
                logger.warning(f"{item_type} {getattr(item, 'id', 'unknown')} has invalid page: {item.page} - marking SPECULATIVE")
                item.status = "SPECULATIVE"
                return True  # Keep item but mark as speculative
        
        # Item is valid
        item.status = "VALID"
        return True
    
    def _validate_evidence_span(self, span: EvidenceSpan, log: bool = True) -> bool:
        """Validate evidence span"""

        # Check page exists
        if span.page not in self.page_contents:
            if log:
                logger.warning(f"Evidence span references non-existent page: {span.page}")
            return False

        # Check quote is not empty
        if not span.quote.strip():
            if log:
                logger.warning("Evidence span has empty quote")
            return False

        # Check quote exists in page content (fuzzy match)
        page_content = self.page_contents[span.page].lower()
        quote_lower = span.quote.lower().strip()

        # Try exact match first
        if quote_lower in page_content:
            return True

        # Try partial match (allow for OCR differences)
        words = quote_lower.split()
        if len(words) >= 3:
            # Check if at least 70% of words appear in sequence
            word_matches = sum(1 for word in words if word in page_content)
            if word_matches / len(words) >= 0.7:
                return True

        if log:
            logger.warning(f"Quote not found in page {span.page}: '{span.quote[:50]}...'")
        return False

    def compute_evidence_density(self, extraction: TypedPaperExtraction) -> float:
        """Compute (#valid quotes)/(#claims) for a paper"""

        num_claims = len(extraction.claims)
        if num_claims == 0:
            return 0.0

        valid_quotes = 0
        for claim in extraction.claims:
            for span in claim.evidence_spans:
                if self._validate_evidence_span(span, log=False):
                    valid_quotes += 1

        return valid_quotes / num_claims

    def check_evidence_density(self,
                               extraction: TypedPaperExtraction,
                               threshold: float) -> float:
        """Log evidence density and warn if below threshold"""

        density = self.compute_evidence_density(extraction)
        if density < threshold:
            logger.warning(
                f"Evidence density {density:.2f} below threshold {threshold}")
        else:
            logger.info(
                f"Evidence density {density:.2f} meets threshold {threshold}")
        return density


def main():
    """Demo typed paper extraction"""
    
    # Example usage
    parser = GROBIDTEIParser()
    
    print("=== Schema-First Paper Reading System ===")
    print("\n✓ GROBID TEI deterministic segmentation")
    print("✓ Typed object extraction with strict JSON schema")
    print("✓ Evidence span validation with page cross-checking")
    print("✓ Pydantic models for type safety")
    
    print(f"\nJSON Schema includes:")
    print("- Claims (theoretical|empirical|engineering)")
    print("- Assumptions with evidence")
    print("- Mechanisms (how methods work)")
    print("- Metrics (with definitions)")  
    print("- Experiments (setup + dataset + ablations)")
    print("- Threat models (attacker capabilities)")
    
    print("\nAll items require:")
    print("- Verbatim quotes ≤300 chars")
    print("- Page number validation")
    print("- Evidence span cross-checking")
    print("- No summaries allowed (temperature=0)")


if __name__ == "__main__":
    main()