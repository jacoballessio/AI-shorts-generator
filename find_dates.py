#!/usr/bin/env python3
"""Find date-related headings."""

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

print("\nAll headings with their IDs:")
print("="*80)
for heading in soup.find_all(['h2', 'h3', 'h4']):
    heading_id = heading.get('id', '')
    heading_text = heading.get_text().strip()
    level = heading.name
    print(f"{level.upper():4} | ID: {heading_id:40} | Text: {heading_text}")

print("\n" + "="*80)
print("Searching for date-related sections (March, 1, 2, 28, February)...")
print("="*80)
for heading in soup.find_all(['h2', 'h3', 'h4']):
    heading_id = heading.get('id', '').lower()
    heading_text = heading.get_text().strip().lower()
    
    # Check if contains date keywords
    date_keywords = ['march', 'february', '28', '1', '2', '3']
    if any(kw in heading_id or kw in heading_text for kw in date_keywords):
        print(f"{heading.name.upper()} | ID: '{heading.get('id')}' | Text: '{heading.get_text().strip()}'")
