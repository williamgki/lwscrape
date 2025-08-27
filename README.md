# LessWrong/GreaterWrong Scraper

Scrapes LessWrong posts chronologically with efficient link extraction.

## Features

- **Chronological scraping**: Works backwards from recent posts
- **Comprehensive link extraction**: Captures all internal/external links with context
- **Respectful scraping**: Built-in rate limiting (1.5s delays)
- **Incremental saving**: Saves data every 10 posts to prevent loss
- **Rich metadata**: Extracts titles, authors, dates, points, comments, tags
- **Structured output**: JSON for posts, CSV for links

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from lw_scraper import LWScraper

scraper = LWScraper(delay=1.5)
scraper.scrape_chronologically(max_pages=5, start_offset=0)
```

### Advanced Usage
```python
# Custom configuration
scraper = LWScraper(
    base_url="https://www.greaterwrong.com",
    delay=2.0  # 2 second delay between requests
)

# Start from a specific offset (e.g., skip first 100 posts)
scraper.scrape_chronologically(max_pages=10, start_offset=100)

# Scrape individual post
post = scraper.scrape_post("ct6SMDuexe9uBwDoL")
```

## Output Structure

### Posts Data (JSON)
```json
{
  "post_id": "ct6SMDuexe9uBwDoL",
  "title": "Thoughts on Gradual Disempowerment",
  "author": "Tom Davidson",
  "date": "15 Aug 2025",
  "content": "...",
  "internal_links": ["https://www.greaterwrong.com/posts/..."],
  "external_links": ["https://example.com/..."]
}
```

### Links Data (CSV)
| source_post_id | target_url | link_text | link_type | context |
|----------------|------------|-----------|-----------|---------|
| ct6SMDuexe9uBwDoL | https://... | "AI safety" | external | "...surrounding text..." |

## Output Files

- `data/posts/`: Individual post JSON files + combined files
- `data/links/`: CSV files with all extracted links
- Timestamped filenames prevent overwrites