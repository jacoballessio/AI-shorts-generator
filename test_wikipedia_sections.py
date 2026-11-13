#!/usr/bin/env python3
"""Test script to check what sections exist on Wikipedia 2026 Iran conflict page."""

import requests
from bs4 import BeautifulSoup

try:
    # Fetch the Wikipedia page
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
    response.raise_for_status()
    data = response.json()
    
    if 'parse' in data and 'text' in data['parse']:
        html_content = data['parse']['text']['*']
        soup = BeautifulSoup(html_content, 'html.parser')
        
        print("Available sections on 2026 Iran conflict Wikipedia page:")
        print("=" * 60)
        
        sections_found = False
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            headline = heading.find('span', class_='mw-headline')
            if headline:
                heading_text = headline.get_text().strip()
                heading_id = headline.get('id', '')
                heading_level = heading.name
                print(f"{heading_level}: {heading_text} (id: {heading_id})")
                sections_found = True
        
        if not sections_found:
            print("No sections found (article may only have lead section)")
            
        # Also check the summary API
        print("\n" + "=" * 60)
        print("Summary from REST API:")
        print("=" * 60)
        summary_url = 'https://en.wikipedia.org/api/rest_v1/page/summary/2026_Iran_conflict'
        summary_resp = requests.get(summary_url, headers=headers, timeout=10)
        if summary_resp.status_code == 200:
            summary_data = summary_resp.json()
            print(f"Title: {summary_data.get('title', 'N/A')}")
            print(f"Extract: {summary_data.get('extract', 'N/A')[:200]}...")
        else:
            print(f"Error fetching summary: {summary_resp.status_code}")
    else:
        print("Error: Could not parse Wikipedia page")
        print(f"Response: {data}")
        
except Exception as e:
    print(f"Error: {e}")
