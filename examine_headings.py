#!/usr/bin/env python3
"""Examine what's inside the heading tags."""

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

print(f"\nTotal h2 tags: {len(soup.find_all('h2'))}")
print(f"Total h3 tags: {len(soup.find_all('h3'))}")

print("\n" + "="*80)
print("First 3 h2 tags raw HTML:")
print("="*80)
for i, h2 in enumerate(soup.find_all('h2')[:3], 1):
    print(f"\n--- H2 #{i} ---")
    print(str(h2))
    print(f"ID attribute: {h2.get('id', 'NO ID')}")
    print(f"Text content: {h2.get_text()}")
    
    # Check for spans
    spans = h2.find_all('span')
    print(f"Contains {len(spans)} span(s)")
    for j, span in enumerate(spans, 1):
        print(f"  Span #{j}: class={span.get('class', 'NO CLASS')}, id={span.get('id', 'NO ID')}, text='{span.get_text()}'")

print("\n" + "="*80)
print("First 3 h3 tags raw HTML:")
print("="*80)
for i, h3 in enumerate(soup.find_all('h3')[:3], 1):
    print(f"\n--- H3 #{i} ---")
    print(str(h3))
    print(f"ID attribute: {h3.get('id', 'NO ID')}")
    print(f"Text content: {h3.get_text()}")
    
    # Check for spans
    spans = h3.find_all('span')
    print(f"Contains {len(spans)} span(s)")
    for j, span in enumerate(spans, 1):
        print(f"  Span #{j}: class={span.get('class', 'NO CLASS')}, id={span.get('id', 'NO ID')}, text='{span.get_text()}'")
