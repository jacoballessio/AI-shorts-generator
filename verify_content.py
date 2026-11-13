#!/usr/bin/env python3
"""Verify scraped content matches what user described."""

import sys
sys.path.insert(0, 'src/data_collection')
from source_scraper import SourceScraper

scraper = SourceScraper()

# Test the "1 March" and "2 March" sections
test_sources = [
    {
        'title': 'Wikipedia: 2026 Iran conflict#1_March',
        'url': 'https://en.wikipedia.org/wiki/2026_Iran_conflict#1_March',
        'content': '',
        'confidence': 0.88,
        'type': 'wikipedia'
    },
    {
        'title': 'Wikipedia: 2026 Iran conflict#2_March',
        'url': 'https://en.wikipedia.org/wiki/2026_Iran_conflict#2_March',
        'content': '',
        'confidence': 0.90,
        'type': 'wikipedia'
    },
]

print("Testing section-specific content extraction:")
print("="*80)

for source in test_sources:
    result = scraper.scrape_source(source)
    if result:
        print(f"\n{result['title']}")
        print(f"Citation: {result['citation']}")
        print(f"Text length: {len(result['text'])} characters")
        print(f"First 600 chars:")
        print(result['text'][:600])
        print("\n" + "-"*80)
