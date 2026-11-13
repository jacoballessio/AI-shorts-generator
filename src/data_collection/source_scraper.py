"""
Source Scraper - fetches full content from detected sources.
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from newspaper import Article
import PyPDF2
from io import BytesIO

try:
    from .web_searcher import search_google
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("⚠️  WebSearcher not available. Install playwright: pip install playwright")


class SourceScraper:
    """Scrape and extract content from research sources."""
    
    def __init__(self, output_dir: str = "data/sources"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def scrape_source(self, source: Dict) -> Optional[Dict]:
        """
        Scrape content from a source.
        
        Args:
            source: Source dict with 'type' and 'content'
            
        Returns:
            Dict with full content, or None if failed
        """
        
        source_type = source['type']
        
        if source_type == 'url':
            return self._scrape_url(source['content'])
        elif source_type == 'twitter':
            return self._scrape_twitter(source)
        elif source_type == 'website':
            return self._scrape_url(source['content'])
        elif source_type == 'wikipedia':
            return self._handle_wikipedia(source)
        elif source_type == 'research_paper':
            return self._handle_research_paper(source)
        elif 'citation' in source or 'search_terms' in source:
            # For non-URL sources, return the detected info
            return {
                'source_type': source_type,
                'content': source.get('content', ''),
                'claim': source.get('claim', ''),
                'citation': source.get('citation', ''),
                'success': True,
            }
        
        return None
    
    def _scrape_twitter(self, source: Dict) -> Optional[Dict]:
        """Handle Twitter sources by going directly to Snaplytics viewer."""
        
        username = source.get('username', source['content'].split('/')[-1])
        
        if not WEB_SEARCH_AVAILABLE:
            return {
                'source_type': 'twitter',
                'username': username,
                'url': source['content'],
                'timestamp': source.get('timestamp'),
                'citation': source.get('citation'),
                'note': 'Web search not available. Install playwright to scrape.',
                'search_url': f'https://twitter.com/{username}',
                'success': False,
            }
        
        # Bypass Google - go directly to Twitter via Snaplytics
        print(f"   Scraping Twitter user via Snaplytics: @{username}")
        twitter_url = f"https://twitter.com/{username}"
        
        try:
            # Use web searcher to extract content from Snaplytics viewer
            from .web_searcher import WebSearcher
            import asyncio
            
            async def fetch_twitter():
                searcher = WebSearcher()
                # Directly extract from Snaplytics-wrapped Twitter URL
                content = await searcher._extract_page_content(twitter_url, "twitter")
                await searcher.close()
                return content
            
            content = asyncio.run(fetch_twitter())
            
            # Check if Snaplytics returned a "page not found" error
            if content and ("Oops! That page can't be found" in content or 
                          "That page can't be found" in content or
                          "nothing was found at this location" in content):
                # This is not a real account/source
                return {
                    'source_type': 'twitter',
                    'username': username,
                    'url': twitter_url,
                    'note': f'Twitter account @{username} does not exist',
                    'success': False,
                }
            
            if content:
                return {
                    'source_type': 'twitter',
                    'username': username,
                    'url': twitter_url,
                    'title': f'@{username} on Twitter',
                    'snippet': f'Twitter profile: @{username}',
                    'content': content,
                    'timestamp': source.get('timestamp'),
                    'citation': source.get('citation'),
                    'via': 'snaplytics',
                    'success': True,
                }
            else:
                return {
                    'source_type': 'twitter',
                    'username': username,
                    'url': source['content'],
                    'note': 'Search did not find results',
                    'success': False,
                }
        except Exception as e:
            print(f"   Error searching for @{username}: {e}")
            return {
                'source_type': 'twitter',
                'username': username,
                'error': str(e),
                'success': False,
            }
    
    def _handle_research_paper(self, source: Dict) -> Optional[Dict]:
        """Handle research paper sources by searching Google Scholar."""
        
        title = source['content']
        
        if not WEB_SEARCH_AVAILABLE:
            return {
                'source_type': 'research_paper',
                'title': title,
                'search_query': source.get('search_query', title),
                'timestamp': source.get('timestamp'),
                'citation': source.get('citation'),
                'note': 'Web search not available. Install playwright to scrape.',
                'scholar_url': f'https://scholar.google.com/scholar?q={title.replace(" ", "+")}',
                'success': False,
            }
        
        # Search Google Scholar
        print(f"   Searching for paper: {title[:60]}...")
        query = f'"{title}" research paper'
        
        try:
            result = search_google(query, source_type="research_paper")
            
            if result and result.get('success'):
                return {
                    'source_type': 'research_paper',
                    'title': title,
                    'url': result['url'],
                    'search_result_title': result['title'],
                    'snippet': result['snippet'],
                    'abstract': result['content'],
                    'timestamp': source.get('timestamp'),
                    'citation': source.get('citation'),
                    'search_query': query,
                    'success': True,
                }
            else:
                return {
                    'source_type': 'research_paper',
                    'title': title,
                    'note': 'Search did not find results',
                    'success': False,
                }
        except Exception as e:
            print(f"   Error searching for paper: {e}")
            return {
                'source_type': 'research_paper',
                'title': title,
                'error': str(e),
                'success': False,
            }
    
    def _fetch_wikipedia_section(self, page_title: str, section_name: str) -> Optional[str]:
        """Fetch content from a specific Wikipedia section."""
        import requests
        from bs4 import BeautifulSoup
        import re
        
        try:
            # Use MediaWiki API to get parsed HTML for the page
            api_url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'parse',
                'page': page_title.replace(' ', '_'),
                'prop': 'text',
                'format': 'json',
            }
            headers = {
                'User-Agent': 'AI-Shorts-Generator/1.0 (Educational Research Tool)'
            }
            
            print(f"   Fetching Wikipedia section: {page_title}#{section_name}")
            response = requests.get(api_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' not in data or 'text' not in data['parse']:
                print(f"   ⚠️  No parsed content in Wikipedia API response")
                return None
            
            html_content = data['parse']['text']['*']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all section headings
            # Wikipedia uses h2, h3, h4 for sections
            # Section anchors are in <span> with class="mw-headline" id="Section_name"
            target_section = None
            section_text = []
            
            # Normalize section name for comparison (spaces vs underscores, case)
            normalized_target = section_name.replace('_', ' ').lower().strip()
            
            # Try multiple variants of the section name
            search_variants = [
                normalized_target,
                section_name.replace('_', ' '),  # Without lowercasing
                section_name.replace(' ', '_'),  # With underscores
                section_name,  # Original
            ]
            
            # For date sections like "2 March", DON'T add just the number
            # (it would match "28 February" etc.)
            
            # Debug: print all available sections
            # Wikipedia parse API returns id directly on heading tags
            all_headings = []
            for heading in soup.find_all(['h2', 'h3', 'h4']):
                heading_text = heading.get_text().strip()
                heading_id = heading.get('id', '')
                # Encode safely for Windows console
                try:
                    all_headings.append(f"{heading_text} (id: {heading_id})")
                except:
                    all_headings.append(f"(id: {heading_id})")
            
            if all_headings:
                try:
                    print(f"   Available sections: {', '.join(all_headings[:10])}")
                except:
                    print(f"   Available sections: [unicode display error - {len(all_headings)} sections found]")
            
            # Find the section heading
            # Wikipedia parse API puts id directly on h2/h3/h4 tags (no mw-headline span)
            for heading in soup.find_all(['h2', 'h3', 'h4']):
                heading_text = heading.get_text().strip()
                heading_id = heading.get('id', '')
                
                # Check both the text and the id attribute
                for variant in search_variants:
                    variant_lower = variant.lower()
                    # Exact match on ID or text
                    if (heading_text.lower() == variant_lower or 
                        heading_id.lower() == variant_lower or
                        heading_id.replace('_', ' ').lower() == variant_lower or
                        heading_text.lower().replace('_', ' ') == variant_lower):
                        target_section = heading
                        print(f"   [OK] Found section: {heading_text}")
                        break
                
                if target_section:
                    break
            
            if not target_section:
                print(f"   [WARN] Section '{section_name}' not found in article")
                return None
            
            # Collect all content until the next heading of same or higher level
            # Wikipedia parse API wraps headings in <div class="mw-heading mw-headingN">
            # So we need to look at siblings of the wrapper div, not the heading itself
            search_element = target_section
            if target_section.parent and target_section.parent.name == 'div' and 'mw-heading' in target_section.parent.get('class', []):
                search_element = target_section.parent
                print(f"   [INFO] Using parent mw-heading div for sibling search")
            
            target_level = target_section.name  # 'h2', 'h3', etc.
            target_level_num = int(target_level[1])
            current = search_element.find_next_sibling()
            
            while current:
                # Stop if we hit another heading of same or higher level
                # Check for mw-heading divs containing headings
                if current.name == 'div' and 'mw-heading' in current.get('class', []):
                    heading = current.find(['h2', 'h3', 'h4'])
                    if heading:
                        current_level_num = int(heading.name[1])
                        if current_level_num <= target_level_num:
                            print(f"   [INFO] Stopped at next {heading.name.upper()}: {heading.get_text().strip()}")
                            break
                        # It's a subsection (higher number = lower level), skip the div but continue
                
                # Also check for bare headings (in case structure changes)
                elif current.name in ['h2', 'h3', 'h4']:
                    current_level = int(current.name[1])
                    if current_level <= target_level_num:
                        break
                
                # Extract text from paragraphs
                if current.name == 'p':
                    text = current.get_text().strip()
                    if text:
                        section_text.append(text)
                
                # Also check for lists within the section
                elif current.name in ['ul', 'ol']:
                    for li in current.find_all('li', recursive=False):
                        text = li.get_text().strip()
                        if text:
                            section_text.append(f"• {text}")
                
                # Also check for div.mw-parser-output children
                elif current.name == 'div':
                    for p in current.find_all('p', recursive=False):
                        text = p.get_text().strip()
                        if text:
                            section_text.append(text)
                
                current = current.find_next_sibling()
            
            # Join and clean up text
            full_text = ' '.join(section_text)
            # Remove citation markers like [1], [2], etc.
            full_text = re.sub(r'\[\d+\]', '', full_text)
            # Remove extra whitespace
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            if full_text:
                print(f"   [OK] Extracted {len(full_text)} characters from section")
            else:
                print(f"   [WARN] Section found but no content extracted")
            
            return full_text if full_text else None
            
        except Exception as e:
            print(f"   Warning: Could not fetch section '{section_name}' from {page_title}: {e}")
            return None
    
    def _handle_wikipedia(self, source: Dict) -> Optional[Dict]:
        """Handle Wikipedia sources - fetch actual article content via API."""
        
        url = source.get('url', '')
        content = source.get('content', '')
        
        # Extract topic name from URL or content
        topic = ''
        topic_for_display = ''
        section_anchor = None
        
        if url:
            # Extract from URL: https://en.wikipedia.org/wiki/2026_Iran_conflict#2_March
            full_url_part = url.split('/wiki/')[-1]
            
            # Check if there's a section anchor
            if '#' in full_url_part:
                topic_part, section_anchor = full_url_part.split('#', 1)
                topic = topic_part.replace('_', ' ')
                section_anchor = section_anchor.replace('_', ' ')  # "2 March"
                topic_for_display = f"{topic}#{section_anchor.replace(' ', '_')}"
            else:
                topic = full_url_part.replace('_', ' ')
                topic_for_display = topic
        elif 'Wikipedia:' in content:
            # Extract from content: "Wikipedia: Doha" -> Doha
            topic = content.split('Wikipedia:')[-1].strip()
            topic_for_display = topic
        
        if not topic:
            return {
                'source_type': 'wikipedia',
                'title': 'Wikipedia',
                'text': 'Could not extract topic',
                'url': url,
                'success': False,
            }
        
        try:
            # If there's a section anchor, try to fetch section-specific content
            if section_anchor:
                section_content = self._fetch_wikipedia_section(topic, section_anchor)
                if section_content:
                    return {
                        'source_type': 'wikipedia',
                        'title': f"Wikipedia: {topic_for_display}",
                        'text': section_content[:2000],
                        'url': url,
                        'topic': topic_for_display,
                        'extract': section_content[:500],
                        'citation': source.get('citation', f'Wikipedia section on {section_anchor}'),
                        'success': True,
                    }
                else:
                    # Section doesn't exist yet - note this in citation
                    print(f"   [INFO] Section '{section_anchor}' not found, using main article")
            
            # Fall back to page summary if no section or section fetch failed
            # Use Wikipedia REST API to get article summary
            # Wikipedia requires a User-Agent header
            api_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(" ", "_")}'
            headers = {
                'User-Agent': 'AI-Shorts-Generator/1.0 (Educational Research Tool)'
            }
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract summary and full description
                extract = data.get('extract', '')
                description = data.get('description', '')
                
                # Combine description and extract for context
                full_text = f"{description}. {extract}" if description else extract
                
                # Adjust citation based on whether we wanted a section but couldn't find it
                if section_anchor:
                    citation = f"Wikipedia article on {topic} (section '{section_anchor}' not yet available)"
                else:
                    citation = source.get('citation', f'Wikipedia article on {topic}')
                
                return {
                    'source_type': 'wikipedia',
                    'title': f"Wikipedia: {topic_for_display}",
                    'text': full_text[:2000],  # Limit to 2000 chars like other sources
                    'url': url,  # Preserve original URL with anchor
                    'topic': topic_for_display,
                    'extract': extract[:500],  # Short summary
                    'citation': citation,
                    'success': True,
                }
            else:
                # Fallback if API fails
                return {
                    'source_type': 'wikipedia',
                    'title': f'Wikipedia: {topic}',
                    'text': f'Background information about {topic} (unable to fetch full content)',
                    'url': url,
                    'topic': topic,
                    'citation': source.get('citation', f'Wikipedia article on {topic}'),
                    'success': True,
                }
        except Exception as e:
            print(f"   Error fetching Wikipedia article for {topic}: {e}")
            return {
                'source_type': 'wikipedia',
                'title': f'Wikipedia: {topic}',
                'text': f'Background information about {topic}',
                'url': url,
                'topic': topic,
                'error': str(e),
                'citation': source.get('citation', f'Wikipedia article on {topic}'),
                'success': True,  # Still mark as success since we have the URL
            }
    
    def _scrape_url(self, url: str) -> Optional[Dict]:
        """Scrape article or PDF from URL."""
        
        try:
            # Check if it's a PDF
            if url.lower().endswith('.pdf'):
                return self._scrape_pdf(url)
            else:
                return self._scrape_article(url)
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def _scrape_article(self, url: str) -> Optional[Dict]:
        """Scrape web article using newspaper3k."""
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            return {
                'source_type': 'article',
                'url': url,
                'title': article.title,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'text': article.text,
                'summary': article.meta_description,
                'keywords': article.meta_keywords,
                'success': True,
            }
            
        except Exception as e:
            print(f"Error scraping article {url}: {e}")
            
            # Fallback: try basic requests
            try:
                response = requests.get(url, timeout=10)
                return {
                    'source_type': 'webpage',
                    'url': url,
                    'text': response.text[:5000],  # Limit text
                    'success': True,
                }
            except:
                return None
    
    def _scrape_pdf(self, url: str) -> Optional[Dict]:
        """Scrape PDF content."""
        
        try:
            response = requests.get(url, timeout=30)
            pdf_file = BytesIO(response.content)
            
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return {
                'source_type': 'pdf',
                'url': url,
                'text': text,
                'num_pages': len(pdf_reader.pages),
                'success': True,
            }
            
        except Exception as e:
            print(f"Error scraping PDF {url}: {e}")
            return None
    
    def scrape_all_sources(self, sources: List[Dict]) -> List[Dict]:
        """Scrape all detected sources and filter out unsuccessful ones."""
        
        scraped = []
        failed_count = 0
        
        for i, source in enumerate(sources):
            print(f"[{i+1}/{len(sources)}] Scraping: {source.get('content', 'N/A')[:50]}...")
            
            result = self.scrape_source(source)
            
            if result:
                # Only include successful scrapes
                if result.get('success', False):
                    # Merge with original source info
                    result['original_source'] = source
                    scraped.append(result)
                else:
                    failed_count += 1
                    print(f"  ⏭️  Skipped: {result.get('note', 'Failed to scrape')}")
            else:
                failed_count += 1
                print(f"  ❌ Failed")
        
        if failed_count > 0:
            print(f"\n⚠️  Filtered out {failed_count} unsuccessful sources")
        
        return scraped
    
    def save_sources(self, sources: List[Dict], video_id: str):
        """Save scraped sources to JSON."""
        
        output_path = self.output_dir / f"{video_id}_sources.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(sources)} sources to {output_path}")


if __name__ == "__main__":
    # Example usage
    scraper = SourceScraper(output_dir="data/sources")
    
    # Example sources from detective
    sources = [
        {
            'type': 'url',
            'content': 'https://www.nature.com/articles/example',
            'confidence': 0.95
        }
    ]
    
    # scraped = scraper.scrape_all_sources(sources)
    # scraper.save_sources(scraped, "video_id_123")
    
    print("Source Scraper ready. Uncomment examples to test.")
