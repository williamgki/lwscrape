from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from docunits_model import DocUnitsExtractor, SourceIds


def test_page_mapping_with_facs_and_pdffigures():
    extractor = DocUnitsExtractor()
    tei = Path('tests/data/sample.tei.xml')
    textdump = Path('tests/data/sample.pdffig.txt')
    units = extractor.extract_from_grobid_tei(tei, 'paper1', SourceIds(), pdffigures2_textdump=textdump)
    text_units = [u for u in units if u.type == 'text']
    pages = [u.page for u in text_units]
    assert pages == [1, 1, 2]
