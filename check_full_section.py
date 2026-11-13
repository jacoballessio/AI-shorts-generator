#!/usr/bin/env python3
"""Check full structure from 2 March to next same-level heading."""

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

# Find "2 March" section (H3)
target_heading = None
for heading in soup.find_all(['h2', 'h3', 'h4']):
    if heading.get_text().strip() == "2 March" or heading.get('id') == "2_March":
        target_heading = heading
        break

if not target_heading:
    print("Could not find '2 March' section!")
else:
    # Navigate from parent div
    parent = target_heading.parent
    print(f"Searching from: {target_heading.get_text().strip()} ({target_heading.name})")
    print("="*80)
    
    # Collect everything until next h2/h3
    current = parent.find_next_sibling()
    section_text = []
    subsections = []
    
    while current:
        # Check if it's a heading div
        if current.name == 'div' and 'mw-heading' in current.get('class', []):
            heading = current.find(['h2', 'h3', 'h4'])
            if heading:
                level = heading.name  # h2, h3, h4
                heading_text = heading.get_text().strip()
                
                # Stop if we hit h2 or h3 (same/higher level than our h3)
                if level in ['h2', 'h3']:
                    print(f"\nStopping at {level.upper()}: {heading_text}")
                    break
                
                # It's an h4 subsection - record it
                if level == 'h4':
                    subsections.append(heading_text)
                    print(f"\nSubsection (H4): {heading_text}")
        
        # Extract paragraphs
        elif current.name == 'p':
            text = current.get_text().strip()
            if text:
                section_text.append(text)
                print(f"  Paragraph ({len(text)} chars): {text[:150]}...")
        
        # Extract lists
        elif current.name in ['ul', 'ol']:
            items = current.find_all('li', recursive=False)
            for li in items:
                text = li.get_text().strip()
                if text:
                    section_text.append(f"• {text}")
                    print(f"  List item: {text[:100]}...")
        
        current = current.find_next_sibling()
    
    print("\n" + "="*80)
    print(f"Summary:")
    print(f"  Subsections found: {len(subsections)}")
    if subsections:
        for s in subsections:
            print(f"    - {s}")
    print(f"  Text blocks extracted: {len(section_text)}")
    print(f"  Total characters: {sum(len(t) for t in section_text)}")
    
    if section_text:
        combined = ' '.join(section_text)
        print(f"\nFirst 500 chars:\n{combined[:500]}...")
