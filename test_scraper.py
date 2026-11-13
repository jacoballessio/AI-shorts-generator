#!/usr/bin/env python3
"""Test the Wikipedia section scraper."""

import sys
sys.path.insert(0, 'src')

try:
    from data_collection.source_scraper import SourceScraper
    print("[OK] Successfully imported SourceScraper")
except Exception as e:
    print(f"[FAIL] Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    scraper = SourceScraper(output_dir='data/sources')
    print("[OK] Successfully created SourceScraper instance")
except Exception as e:
    print(f"[FAIL] Failed to create scraper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# First, let's see what sections actually exist
print("\nFetching actual sections from Wikipedia...")
print("=" * 80)

import requests
from bs4 import BeautifulSoup

api_url = 'https://en.wikipedia.org/w/api.php'
params = {
    'action': 'parse',
    'page': '2026_Iran_conflict',
    'prop': 'text',
    'format': 'json',
}
headers = {
    'User-Agent': 'AI-Shorts-Generator/1.0 (Educational Research Tool)'
}

response = requests.get(api_url, params=params, headers=headers, timeout=10)
data = response.json()
html_content = data['parse']['text']['*']
soup = BeautifulSoup(html_content, 'html.parser')

print("\nAll sections found in article:")
for heading in soup.find_all(['h2', 'h3', 'h4']):
    headline = heading.find('span', class_='mw-headline')
    if headline:
        heading_text = headline.get_text().strip()
        heading_id = headline.get('id', '')
        print(f"  {heading.name}: '{heading_text}' (id='{heading_id}')")

# Test scraping with section
test_sources = [
    {
        'type': 'wikipedia',
        'content': 'Wikipedia: 2026 Iran conflict - 2 March',
        'url': 'https://en.wikipedia.org/wiki/2026_Iran_conflict#2_March',
        'confidence': 0.9,
        'citation': 'Events on 2 March 2026',
    },
{        'type': 'wikipedia',
        'content': 'Wikipedia: 2026 Iran conflict',
        'url': 'https://en.wikipedia.org/wiki/2026_Iran_conflict',
        'confidence': 0.85,
        'citation': 'Background on 2026 Iran conflict',
    },
]

print("Testing Wikipedia scraper...")
print("=" * 80)

for i, source in enumerate(test_sources, 1):
    print(f"\n[{i}/{len(test_sources)}] Testing: {source['content']}")
    result = scraper._handle_wikipedia(source)
    if result and result.get('success'):
        print(f"  [OK] Success!")
        print(f"  Title: {result['title']}")
        print(f"  Citation: {result['citation']}")
        print(f"  Text: {result['text'][:150]}...")
    else:
        print(f"  [FAIL] Failed: {result}")
