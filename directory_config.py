from dataclasses import dataclass
from pathlib import Path
import json

CONFIG_PATH = Path(__file__).with_name('directory_config.json')

@dataclass
class DirectoryConfig:
    corpus_dir: Path
    output_dir: Path
    temp_dir: Path

def load_config(config_path: Path = CONFIG_PATH) -> DirectoryConfig:
    """Load directory configuration from JSON file or provide defaults."""
    if config_path.exists():
        data = json.loads(config_path.read_text())
    else:
        data = {}
    return DirectoryConfig(
        corpus_dir=Path(data.get('corpus_dir', '/home/ubuntu/LW_scrape/multi_source_corpus')),
        output_dir=Path(data.get('output_dir', '/home/ubuntu/LW_scrape/scored_corpus')),
        temp_dir=Path(data.get('temp_dir', '/home/ubuntu/LW_scrape/tmp')),
    )
