#!/usr/bin/env python3
"""Test full source detection and scraping for the NBC News video."""

import sys
import json
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/data_collection')

from data_collection.source_detective import SourceDetective
from data_collection.source_scraper import SourceScraper

# NBC News missile video metadata
metadata = {
    'title': 'U.S., Israel launch attacks on Iran',
    'description': 'U.S. and Israeli forces launched coordinated attacks against Iran on Saturday after Iranian missiles struck Doha, Qatar.',
    'uploader': 'NBC News',
    'upload_date': '20260302',
    'duration': 34,
    'id': 'aRmJZ5fPEhI',
    'url': 'https://www.youtube.com/watch?v=aRmJZ5fPEhI'
}

print("="*80)
print("SOURCE DETECTION AND SCRAPING TEST")
print("="*80)
print(f"\nVideo: {metadata['title']}")
print(f"Uploader: {metadata['uploader']}")
print(f"Upload date: {metadata['upload_date']}")
print(f"\nDetecting sources...")

# Step 1: Detect sources
detective = SourceDetective()
title_and_desc = f"{metadata['title']} {metadata['description']}"
sources = detective.find_sources(title_and_desc, metadata)

print(f"\nFound {len(sources)} sources:")
for i, source in enumerate(sources, 1):
    print(f"  {i}. {source['title']}")
    print(f"     URL: {source.get('url', 'N/A')}")
    print(f"     Confidence: {source['confidence']:.2f}")
    print(f"     Type: {source['type']}")
    print()

# Step 2: Scrape Wikipedia sources
print("="*80)
print("SCRAPING WIKIPEDIA SOURCES")
print("="*80)

scraper = SourceScraper()

# Filter to just Wikipedia sources with sections
wiki_sources = [s for s in sources if s['type'] == 'wikipedia' and '#' in s.get('url', '')]

for i, source in enumerate(wiki_sources[:3], 1):  # First 3 Wikipedia sources
    print(f"\n[{i}/{len(wiki_sources)}] {source['title']}")
    print("-"*80)
    
    result = scraper.scrape_source(source)
    
    if result:
        print(f"[OK] Successfully scraped")
        print(f"Length: {len(result['text'])} characters")
        print(f"Citation: {result['citation']}")
        print(f"\nFirst 800 characters:")
        print(result['text'][:800])
        print("...")
    else:
        print("[FAIL] Scraping failed")
    print()

# Step 3: Summary
print("="*80)
print("SUMMARY")
print("="*80)
print(f"\nTotal sources detected: {len(sources)}")
print(f"Wikipedia sources with sections: {len(wiki_sources)}")
print(f"Sources scraped successfully: {sum(1 for s in wiki_sources[:3] if scraper.scrape_source(s))}")

print("\nWikipedia sources:")
for source in wiki_sources:
    url = source.get('url', '')
    section = url.split('#')[1] if '#' in url else 'main article'
    print(f"  - {section}: conf={source['confidence']:.2f}")
