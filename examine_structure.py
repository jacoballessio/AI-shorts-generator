#!/usr/bin/env python3
"""Examine HTML structure around a specific section."""

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

print("Fetching Wikipedia page...")
response = requests.get(api_url, params=params, headers=headers, timeout=10)
data = response.json()
html_content = data['parse']['text']['*']
soup = BeautifulSoup(html_content, 'html.parser')

# Find "2 March" section
target = None
for heading in soup.find_all(['h2', 'h3', 'h4']):
    if heading.get_text().strip() == "2 March" or heading.get('id') == "2_March":
        target = heading
        break

if not target:
    print("Could not find '2 March' section!")
else:
    print(f"Found section: {target.get_text().strip()}")
    print(f"Heading tag: {target.name}")
    print(f"Heading ID: {target.get('id')}")
    
    print("\n" + "="*80)
    print("Next 10 siblings:")
    print("="*80)
    
    current = target.find_next_sibling()
    count = 0
    while current and count < 10:
        print(f"\n--- Sibling #{count+1} ---")
        print(f"Tag: {current.name}")
        if current.name == 'p':
            text = current.get_text().strip()[:200]
            print(f"Text: {text}...")
        elif current.name in ['h2', 'h3', 'h4']:
            print(f"Heading: {current.get_text().strip()}")
            print(f"ID: {current.get('id')}")
            break
        else:
            print(f"HTML: {str(current)[:300]}")
        
        current = current.find_next_sibling()
        count += 1
    
    print(f"\nFound {count} siblings before next heading")
