#!/usr/bin/env python3
"""Examine parent and overall structure."""

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
    print(f"Found heading: {target.get_text().strip()}")
    print(f"Parent tag: {target.parent.name}")
    print(f"Parent class: {target.parent.get('class', [])}")
    
    # Check all children of the parent
    print("\n" + "="*80)
    print("ALL children of parent (first 50):")
    print("="*80)
    
    started = False
    count = 0
    for child in target.parent.children:
        if child == target:
            started = True
            print(f"\n>>> [THIS IS THE '2 MARCH' HEADING] <<<")
            continue
        
        if started and count < 30:
            if hasattr(child, 'name') and child.name:
                if child.name == 'p':
                    text = child.get_text().strip()[:150]
                    print(f"\n<p> {text}...")
                elif child.name in ['h2', 'h3', 'h4']:
                    print(f"\n<{child.name}> {child.get_text().strip()} (NEXT SECTION - STOP)")
                    break
                elif child.name == 'ul':
                    items = child.find_all('li', recursive=False)
                    print(f"\n<ul> with {len(items)} items")
                    if items:
                        print(f"  First item: {items[0].get_text().strip()[:100]}")
                else:
                    print(f"\n<{child.name}> {str(child)[:200]}")
                count += 1
            elif isinstance(child, str) and child.strip():
                print(f"\nText node: '{child.strip()[:100]}'")
