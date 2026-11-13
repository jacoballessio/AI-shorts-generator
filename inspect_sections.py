#!/usr/bin/env python3
"""Directly inspect Wikipedia sections and save to file."""

import requests
from bs4 import BeautifulSoup
import json

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

# Save raw HTML first
with open('wiki_raw.html', 'w', encoding='utf-8') as f:
    f.write(html_content[:50000])  # First 50KB
print("Saved first 50KB of HTML to wiki_raw.html")

# Try different ways to find sections
print("\nLooking for h2 tags...")
h2_tags = soup.find_all('h2')
print(f"Found {len(h2_tags)} h2 tags")

print("\nLooking for h3 tags...")
h3_tags = soup.find_all('h3')
print(f"Found {len(h3_tags)} h3 tags")

print("\nLooking for spans with class mw-headline...")
headlines = soup.find_all('span', class_='mw-headline')
print(f"Found {len(headlines)} mw-headline spans")

if headlines:
    print("\nFirst 5 headlines:")
    for h in headlines[:5]:
        print(f"  Text: '{h.get_text()}' ID: '{h.get('id', '')}'")

sections = []
for heading in soup.find_all(['h2', 'h3', 'h4']):
    headline = heading.find('span', class_='mw-headline')
    if headline:
        heading_text = headline.get_text().strip()
        heading_id = headline.get('id', '')
        sections.append({
            'level': heading.name,
            'text': heading_text,
            'id': heading_id,
        })

# Save to file
with open('wiki_sections.json', 'w', encoding='utf-8') as f:
    json.dump(sections, f, indent=2, ensure_ascii=False)

print(f"Found {len(sections)} sections")
print("Sections saved to wiki_sections.json")
print("\nFirst 10 sections:")
for s in sections[:10]:
    print(f"  {s['level']}: text='{s['text']}' id='{s['id']}'")
