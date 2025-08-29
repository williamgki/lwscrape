#!/usr/bin/env python3

import uuid
import hashlib
import struct
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import duckdb
import simhash
from datasketch import MinHash
import xml.etree.ElementTree as ET
import json
import re


@dataclass
class SourceIds:
    openalex_id: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openreview_id: Optional[str] = None


@dataclass
class FigureMeta:
    figure_id: str
    name: str
    bbox: List[float]  # [x, y, width, height]
    image_path: str


@dataclass
class CharSpan:
    start: int
    end: int


@dataclass
class Hashes:
    simhash: bytes
    minhash_signature: bytes


@dataclass
class DocUnit:
    paper_id: str
    source_ids: SourceIds
    section_id: str
    page: int
    para_id: str
    type: str  # "text" | "figure" | "table" | "caption"
    text: str
    figure_meta: Optional[FigureMeta] = None
    refs: List[str] = None  # canonical referenced work IDs
    char_span: Optional[CharSpan] = None
    hashes: Optional[Hashes] = None
    ingest_ts: datetime = None
    
    def __post_init__(self):
        if self.refs is None:
            self.refs = []
        if self.ingest_ts is None:
            self.ingest_ts = datetime.now()
        if self.hashes is None:
            self.hashes = self._compute_hashes()
    
    def _compute_hashes(self) -> Hashes:
        """Compute simhash and minhash for deduplication"""
        # SimHash for near-duplicate detection
        sh = simhash.Simhash(self.text)
        simhash_bytes = struct.pack('>Q', sh.value)
        
        # MinHash for similarity estimation
        mh = MinHash()
        words = re.findall(r'\w+', self.text.lower())
        for word in words:
            mh.update(word.encode('utf8'))
        minhash_bytes = bytes(mh.digest())
        
        return Hashes(simhash=simhash_bytes, minhash_signature=minhash_bytes)


@dataclass
class DocRefEdge:
    src_paper_id: str
    dst_paper_id: str
    edge_type: str  # "cites" | "is_cited_by"


class DocUnitsExtractor:
    """Extract DocUnits from GROBID TEI XML and pdffigures2 JSON"""
    
    def __init__(self):
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    def extract_from_grobid_tei(self, tei_file: Path, paper_id: str, source_ids: SourceIds,
                                pdffigures2_textdump: Optional[Path] = None) -> List[DocUnit]:
        """Extract DocUnits from GROBID TEI XML"""
        units: List[DocUnit] = []

        try:
            tree = ET.parse(tei_file)
            root = tree.getroot()

            zone_page_map = self._build_zone_page_map(root)
            pb_page_map = self._build_pb_page_map(root)
            pdffig_pages = self._load_pdffigures2_text(pdffigures2_textdump) if pdffigures2_textdump else []

            # Extract text sections
            sections = root.findall('.//tei:div[@type]', self.ns)

            for section in sections:
                section_title = self._get_section_title(section)
                section_id = section_title or "unknown"

                # Extract paragraphs from this section
                paragraphs = section.findall('.//tei:p', self.ns)

                for i, para in enumerate(paragraphs):
                    text = self._extract_text_content(para)
                    if not text.strip():
                        continue

                    # Extract references from this paragraph
                    refs = self._extract_paragraph_refs(para)

                    # Estimate page number using coordinates/text
                    page = self._estimate_page(para, text, i, zone_page_map, pb_page_map, pdffig_pages)

                    unit = DocUnit(
                        paper_id=paper_id,
                        source_ids=source_ids,
                        section_id=section_id,
                        page=page,
                        para_id=str(uuid.uuid4()),
                        type="text",
                        text=text,
                        refs=refs
                    )
                    units.append(unit)

            # Extract bibliography references
            refs_units = self._extract_bibliography(root, paper_id, source_ids)
            units.extend(refs_units)

        except Exception as e:
            print(f"Error processing TEI file {tei_file}: {e}")

        return units
    
    def extract_from_pdffigures2(self, json_file: Path, paper_id: str, source_ids: SourceIds) -> List[DocUnit]:
        """Extract figure/table DocUnits from pdffigures2 JSON"""
        units = []
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for fig in data:
                if fig.get('figType') in ['Figure', 'Table']:
                    figure_meta = FigureMeta(
                        figure_id=fig.get('name', ''),
                        name=fig.get('name', ''),
                        bbox=fig.get('regionBoundary', {}).get('x1y1x2y2', [0, 0, 0, 0]),
                        image_path=fig.get('renderURL', '')
                    )
                    
                    # Create unit for the figure itself
                    fig_unit = DocUnit(
                        paper_id=paper_id,
                        source_ids=source_ids,
                        section_id=fig.get('page', 1),  # Use page as section for figures
                        page=fig.get('page', 1),
                        para_id=str(uuid.uuid4()),
                        type="figure" if fig.get('figType') == 'Figure' else "table",
                        text=f"{fig.get('name', '')} (visual content)",
                        figure_meta=figure_meta
                    )
                    units.append(fig_unit)
                    
                    # Create unit for the caption
                    if fig.get('caption'):
                        caption_unit = DocUnit(
                            paper_id=paper_id,
                            source_ids=source_ids,
                            section_id=f"caption_{fig.get('name', '')}",
                            page=fig.get('page', 1),
                            para_id=str(uuid.uuid4()),
                            type="caption",
                            text=fig.get('caption', ''),
                            figure_meta=figure_meta
                        )
                        units.append(caption_unit)
        
        except Exception as e:
            print(f"Error processing pdffigures2 file {json_file}: {e}")
        
        return units
    
    def _get_section_title(self, section) -> str:
        """Extract section title from TEI section element"""
        head = section.find('.//tei:head', self.ns)
        if head is not None:
            return head.text or ""
        return ""
    
    def _extract_text_content(self, element) -> str:
        """Extract clean text content from TEI element"""
        text = ET.tostring(element, encoding='unicode', method='text')
        return re.sub(r'\s+', ' ', text).strip()
    
    def _extract_paragraph_refs(self, para) -> List[str]:
        """Extract reference IDs from paragraph citations"""
        refs = []
        citations = para.findall('.//tei:ref[@type="bibr"]', self.ns)
        for cite in citations:
            target = cite.get('target', '').replace('#', '')
            if target:
                refs.append(target)
        return refs

    def _build_zone_page_map(self, root) -> Dict[str, int]:
        """Map facsimile zone ids to page numbers"""
        zone_page: Dict[str, int] = {}
        facsimile = root.find('.//tei:facsimile', self.ns)
        if facsimile is None:
            return zone_page
        for surface in facsimile.findall('.//tei:surface', self.ns):
            n = surface.get('n')
            try:
                page_num = int(n) if n is not None else None
            except ValueError:
                page_num = None
            if page_num is None:
                continue
            for zone in surface.findall('.//tei:zone', self.ns):
                zone_id = zone.get('{http://www.w3.org/XML/1998/namespace}id')
                if zone_id:
                    zone_page[zone_id] = page_num
        return zone_page

    def _build_pb_page_map(self, root) -> Dict[int, int]:
        """Map paragraph elements to page numbers using <pb> markers"""
        pb_map: Dict[int, int] = {}
        page = 1
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            if tag == 'pb':
                n = elem.get('n')
                if n and n.isdigit():
                    page = int(n)
                else:
                    page += 1
            elif tag == 'p':
                pb_map[id(elem)] = page
        return pb_map

    def _load_pdffigures2_text(self, text_file: Path) -> List[str]:
        """Load pdffigures2 text dump split by pages"""
        pages: List[str] = []
        try:
            content = text_file.read_text(encoding='utf-8')
            if '\f' in content:
                pages = content.split('\f')
            else:
                pages = content.split('\n\n')
        except Exception:
            pass
        return pages

    def _strip_ns(self, tag: str) -> str:
        return tag.split('}')[-1] if '}' in tag else tag

    def _estimate_page(self, para, text: str, index: int,
                       zone_page_map: Dict[str, int],
                       pb_page_map: Dict[int, int],
                       pdffig_pages: List[str]) -> int:
        """Determine page number using TEI coordinates or pdffigures2 text"""
        facs = para.get('facs')
        if facs:
            zone_id = facs.lstrip('#')
            if zone_id in zone_page_map:
                return zone_page_map[zone_id]
        if id(para) in pb_page_map:
            return pb_page_map[id(para)]
        if pdffig_pages:
            for i, page_text in enumerate(pdffig_pages, start=1):
                if text and text in page_text:
                    return i
        return max(1, index // 20)
    
    def _extract_bibliography(self, root, paper_id: str, source_ids: SourceIds) -> List[DocUnit]:
        """Extract bibliography as separate DocUnits"""
        units = []
        
        biblstruct_elements = root.findall('.//tei:biblStruct', self.ns)
        for i, biblio in enumerate(biblstruct_elements):
            title_elem = biblio.find('.//tei:title', self.ns)
            title = title_elem.text if title_elem is not None else f"Reference {i+1}"
            
            # Extract full bibliographic text
            bib_text = self._extract_text_content(biblio)
            
            unit = DocUnit(
                paper_id=paper_id,
                source_ids=source_ids,
                section_id="references",
                page=-1,  # References typically at end
                para_id=str(uuid.uuid4()),
                type="reference",
                text=bib_text
            )
            units.append(unit)
        
        return units


class DocUnitsStorage:
    """Manage DocUnits and DocRefEdges in DuckDB/Parquet"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create DocUnits and DocRefEdges tables"""
        
        # DocUnits table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS docunits (
                paper_id VARCHAR,
                source_ids STRUCT(
                    openalex_id VARCHAR,
                    doi VARCHAR,
                    arxiv_id VARCHAR,
                    openreview_id VARCHAR
                ),
                section_id VARCHAR,
                page INTEGER,
                para_id VARCHAR PRIMARY KEY,
                type VARCHAR,
                text TEXT,
                figure_meta STRUCT(
                    figure_id VARCHAR,
                    name VARCHAR,
                    bbox FLOAT[],
                    image_path VARCHAR
                ),
                refs VARCHAR[],
                char_span STRUCT(
                    start INTEGER,
                    end INTEGER
                ),
                hashes STRUCT(
                    simhash BLOB,
                    minhash_signature BLOB
                ),
                ingest_ts TIMESTAMP
            )
        """)
        
        # DocRefEdges table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS docrefedges (
                src_paper_id VARCHAR,
                dst_paper_id VARCHAR,
                edge_type VARCHAR,
                PRIMARY KEY (src_paper_id, dst_paper_id, edge_type)
            )
        """)
        
        # Indexes for fast queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_docunits_paper ON docunits(paper_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_docunits_type ON docunits(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_docunits_section ON docunits(section_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON docrefedges(src_paper_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON docrefedges(dst_paper_id)")
    
    def insert_docunits(self, units: List[DocUnit]):
        """Insert DocUnits into database"""
        data = []
        for unit in units:
            # Convert dataclass to dict for DuckDB insertion
            unit_dict = {
                'paper_id': unit.paper_id,
                'source_ids': asdict(unit.source_ids),
                'section_id': unit.section_id,
                'page': unit.page,
                'para_id': unit.para_id,
                'type': unit.type,
                'text': unit.text,
                'figure_meta': asdict(unit.figure_meta) if unit.figure_meta else None,
                'refs': unit.refs,
                'char_span': asdict(unit.char_span) if unit.char_span else None,
                'hashes': asdict(unit.hashes) if unit.hashes else None,
                'ingest_ts': unit.ingest_ts
            }
            data.append(unit_dict)
        
        df = pd.DataFrame(data)
        self.conn.execute("INSERT INTO docunits SELECT * FROM df")
    
    def insert_edges(self, edges: List[DocRefEdge]):
        """Insert citation edges into database"""
        data = [asdict(edge) for edge in edges]
        df = pd.DataFrame(data)
        self.conn.execute("INSERT OR IGNORE INTO docrefedges SELECT * FROM df")
    
    def export_to_parquet(self, output_dir: Path):
        """Export tables to Parquet files"""
        output_dir.mkdir(exist_ok=True)
        
        # Export DocUnits
        self.conn.execute(f"""
            COPY docunits TO '{output_dir}/DocUnits.parquet' (FORMAT PARQUET)
        """)
        
        # Export DocRefEdges  
        self.conn.execute(f"""
            COPY docrefedges TO '{output_dir}/DocRefEdges.parquet' (FORMAT PARQUET)
        """)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        # DocUnits stats
        result = self.conn.execute("SELECT COUNT(*) FROM docunits").fetchone()
        stats['total_docunits'] = result[0]
        
        result = self.conn.execute("SELECT type, COUNT(*) FROM docunits GROUP BY type").fetchall()
        stats['units_by_type'] = dict(result)
        
        result = self.conn.execute("SELECT COUNT(DISTINCT paper_id) FROM docunits").fetchone()
        stats['unique_papers'] = result[0]
        
        # DocRefEdges stats
        result = self.conn.execute("SELECT COUNT(*) FROM docrefedges").fetchone()
        stats['total_edges'] = result[0]
        
        result = self.conn.execute("SELECT edge_type, COUNT(*) FROM docrefedges GROUP BY edge_type").fetchall()
        stats['edges_by_type'] = dict(result)
        
        return stats
    
    def find_similar_units(self, query_unit: DocUnit, similarity_threshold: float = 0.8) -> List[Dict]:
        """Find similar DocUnits using MinHash similarity"""
        # This would implement MinHash similarity search
        # For now, return placeholder
        return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def build_citation_edges(units: List[DocUnit], reference_mapping: Dict[str, str]) -> List[DocRefEdge]:
    """Build citation graph edges from DocUnits references"""
    edges = []
    
    paper_ids = {unit.paper_id for unit in units}
    
    for unit in units:
        if unit.refs:
            for ref_id in unit.refs:
                # Map reference ID to canonical paper ID
                canonical_id = reference_mapping.get(ref_id)
                
                if canonical_id and canonical_id in paper_ids:
                    # Create citation edge
                    cite_edge = DocRefEdge(
                        src_paper_id=unit.paper_id,
                        dst_paper_id=canonical_id,
                        edge_type="cites"
                    )
                    edges.append(cite_edge)
                    
                    # Create reverse edge
                    cited_edge = DocRefEdge(
                        src_paper_id=canonical_id,
                        dst_paper_id=unit.paper_id,
                        edge_type="is_cited_by"
                    )
                    edges.append(cited_edge)
    
    return edges


if __name__ == "__main__":
    # Test DocUnits extraction
    extractor = DocUnitsExtractor()
    storage = DocUnitsStorage()
    
    # Example usage would be:
    # units = extractor.extract_from_grobid_tei(Path("paper.tei.xml"), "paper_1", SourceIds(doi="10.1234/example"))
    # storage.insert_docunits(units)
    # storage.export_to_parquet(Path("./docunits_output"))
    
    print("DocUnits model implementation complete")