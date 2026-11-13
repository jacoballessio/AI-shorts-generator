#!/usr/bin/env python3
"""Examine siblings of the heading's parent div."""

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
target_heading = None
for heading in soup.find_all(['h2', 'h3', 'h4']):
    if heading.get_text().strip() == "2 March" or heading.get('id') == "2_March":
        target_heading = heading
        break

if not target_heading:
    print("Could not find '2 March' section!")
else:
    # Check if heading is wrapped in a div
    parent = target_heading.parent
    print(f"Heading: {target_heading.get_text().strip()}")
    print(f"Heading tag: {target_heading.name}")
    print(f"Parent tag: {parent.name}")
    print(f"Parent class: {parent.get('class', [])}")
    
    # If parent is a mw-heading div, look at its siblings
    if parent.name == 'div' and 'mw-heading' in parent.get('class', []):
        print("\nHeading is wrapped in mw-heading div")
        print("Looking at div's siblings:")
        print("="*80)
        
        current = parent.find_next_sibling()
        count = 0
        section_text = []
        
        while current and count < 20:
            print(f"\n--- Sibling #{count+1} ---")
            print(f"Tag: {current.name}")
            
            if current.name == 'p':
                text = current.get_text().strip()
                print(f"Text ({len(text)} chars): {text[:200]}...")
                section_text.append(text)
            elif current.name == 'div' and 'mw-heading' in current.get('class', []):
                # Next section
                next_heading = current.find(['h2', 'h3', 'h4'])
                if next_heading:
                    print(f"Next section: {next_heading.get_text().strip()}")
                    break
            elif current.name in ['ul', 'ol']:
                items = current.find_all('li', recursive=False)
                print(f"List with {len(items)} items")
                for li in items:
                    text = li.get_text().strip()
                    section_text.append(f"• {text}")
                    print(f"  - {text[:100]}")
            else:
                print(f"Other: {current.name}")
                print(f"  Content: {str(current)[:200]}")
            
            current = current.find_next_sibling()
            count += 1
        
        print("\n" + "="*80)
        print(f"Extracted {len(section_text)} text blocks:")
        print("="*80)
        for i, text in enumerate(section_text[:5], 1):
            print(f"{i}. {text[:150]}...")
