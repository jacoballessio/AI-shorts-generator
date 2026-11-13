"""
Web Searcher - Uses Playwright to search Google and extract content from sources.

Note: Twitter/X.com links are routed through https://snaplytics.io/twitter-viewer/ 
to bypass anti-bot mechanisms.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
import re


class WebSearcher:
    """Search Google and extract content using Playwright."""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self._initialized = False
    
    async def _ensure_browser(self):
        """Lazy initialize Playwright browser."""
        if not self._initialized:
            try:
                from playwright.async_api import async_playwright
                self.playwright = await async_playwright().start()
                
                # Launch browser with options to avoid detection
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox'
                    ]
                )
                
                # Create context with realistic user-agent
                self.context = await self.browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US'
                )
                
                self._initialized = True
            except ImportError:
                print("⚠️  Playwright not installed. Install with: pip install playwright")
                print("   Then run: playwright install chromium")
                raise
    
    async def search_and_extract(self, query: str, source_type: str = "general") -> Optional[Dict]:
        """
        Search Google for a query and extract content from top result.
        
        Args:
            query: Search query
            source_type: Type of source (twitter, website, research_paper, etc.)
            
        Returns:
            Dict with extracted content
        """
        await self._ensure_browser()
        
        page = await self.context.new_page()
        
        try:
            # Perform Google search
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for results with longer timeout and multiple selectors
            try:
                await page.wait_for_selector('div#search, div#rso, div#main', timeout=15000)
            except:
                # If we can't find search results, try to screenshot for debugging
                print(f"      Warning: Could not find search results container")
                return None
            
            # Small delay for dynamic content
            await asyncio.sleep(2)
            
            # Extract first organic result
            results = []
            
            # Try to find regular search results
            result_elements = await page.query_selector_all('div.g, div.tF2Cxc, div[data-sokoban-container]')
            
            for element in result_elements[:3]:  # Get top 3 results
                try:
                    # Extract title
                    title_el = await element.query_selector('h3')
                    title = await title_el.inner_text() if title_el else ""
                    
                    # Extract link
                    link_el = await element.query_selector('a')
                    href = await link_el.get_attribute('href') if link_el else ""
                    
                    # Extract snippet
                    snippet_el = await element.query_selector('div[data-sncf], div.VwiC3b, span.aCOpRe')
                    snippet = await snippet_el.inner_text() if snippet_el else ""
                    
                    if href and title:
                        results.append({
                            'title': title,
                            'url': href,
                            'snippet': snippet
                        })
                except Exception as e:
                    continue
            
            if not results:
                return None
            
            # Get the best result
            best_result = results[0]
            
            # Try to extract more content from the page
            content = await self._extract_page_content(best_result['url'], source_type)
            
            return {
                'query': query,
                'search_url': search_url,
                'title': best_result['title'],
                'url': best_result['url'],
                'snippet': best_result['snippet'],
                'content': content,
                'additional_results': results[1:],
                'success': True
            }
            
        except Exception as e:
            print(f"   Error searching for '{query}': {str(e)[:100]}")
            return None
        finally:
            await page.close()
    
    async def _extract_page_content(self, url: str, source_type: str) -> str:
        """Extract content from a webpage."""
        await self._ensure_browser()
        
        page = await self.context.new_page()
        
        try:
            # Route Twitter/X.com links through Snaplytics to bypass anti-bot
            if source_type == "twitter" or 'twitter.com' in url or 'x.com' in url:
                # Use Snaplytics Twitter viewer
                snaplytics_url = f"https://snaplytics.io/twitter-viewer/{url}"
                await page.goto(snaplytics_url, wait_until="domcontentloaded", timeout=15000)
            else:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            # Give page time to load
            await asyncio.sleep(2)
            
            if source_type == "twitter":
                # Extract tweet content
                content = await self._extract_twitter_content(page)
            elif source_type == "research_paper":
                # Extract paper abstract
                content = await self._extract_paper_content(page)
            else:
                # Extract general page content
                content = await self._extract_general_content(page)
            
            return content
            
        except Exception as e:
            print(f"   Error extracting content from {url}: {str(e)[:50]}")
            return ""
        finally:
            await page.close()
    
    async def _extract_twitter_content(self, page) -> str:
        """Extract content from a Twitter profile or tweet (via Snaplytics viewer)."""
        try:
            # Wait for content to load
            await asyncio.sleep(3)
            
            # First try Snaplytics-specific selectors
            tweet_containers = await page.query_selector_all('.tweet-container, .twitter-tweet, .tweet-content')
            
            tweets = []
            for container in tweet_containers[:5]:  # Get first 5 tweets
                try:
                    text = await container.inner_text()
                    if text and len(text.strip()) > 10:
                        tweets.append(text.strip())
                except:
                    continue
            
            if tweets:
                return "\n\n---\n\n".join(tweets)
            
            # Fallback: Try native Twitter selectors (if Snaplytics shows original)
            tweet_elements = await page.query_selector_all('article[data-testid="tweet"]')
            
            for tweet_el in tweet_elements[:5]:
                try:
                    text_el = await tweet_el.query_selector('[data-testid="tweetText"]')
                    if text_el:
                        text = await text_el.inner_text()
                        tweets.append(text)
                except:
                    continue
            
            if tweets:
                return "\n\n---\n\n".join(tweets)
            
            # Final fallback: get visible text content
            body = await page.query_selector('body')
            if body:
                text = await body.inner_text()
                # Filter out navigation/UI text
                lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20]
                return '\n\n'.join(lines[:10])[:2000]
                
        except Exception as e:
            print(f"      Twitter extraction error: {str(e)[:50]}")
        
        return ""
    
    async def _extract_paper_content(self, page) -> str:
        """Extract abstract from a research paper page."""
        try:
            # Common selectors for paper abstracts
            selectors = [
                'section[id*="abstract"]',
                'div[class*="abstract"]',
                'p[class*="abstract"]',
                '#abstract',
                '.abstract'
            ]
            
            for selector in selectors:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if len(text) > 50:  # Valid abstract
                        return text
            
            # Fallback: get first few paragraphs
            paragraphs = await page.query_selector_all('p')
            content = []
            for p in paragraphs[:5]:
                text = await p.inner_text()
                if len(text) > 50:
                    content.append(text)
            
            return "\n\n".join(content)
            
        except:
            pass
        
        return ""
    
    async def _extract_general_content(self, page) -> str:
        """Extract general text content from a webpage."""
        try:
            # Try to find main content area
            main_selectors = ['article', 'main', '#content', '.content', '[role="main"]']
            
            for selector in main_selectors:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if len(text) > 100:
                        # Clean and truncate
                        return text[:3000]
            
            # Fallback: get all paragraphs
            paragraphs = await page.query_selector_all('p')
            content = []
            for p in paragraphs[:10]:
                text = await p.inner_text()
                if len(text) > 30:
                    content.append(text)
            
            result = "\n\n".join(content)
            return result[:3000]
            
        except:
            pass
        
        return ""
    
    async def close(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._initialized = False
    
    def search_and_extract_sync(self, query: str, source_type: str = "general") -> Optional[Dict]:
        """Synchronous wrapper for search_and_extract."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is running, create a new one
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.search_and_extract(query, source_type))
        else:
            return asyncio.run(self.search_and_extract(query, source_type))


# Convenience function for pipeline use
async def search_google_async(query: str, source_type: str = "general") -> Optional[Dict]:
    """Search Google and extract content (async)."""
    searcher = WebSearcher()
    try:
        result = await searcher.search_and_extract(query, source_type)
        return result
    finally:
        await searcher.close()


def search_google(query: str, source_type: str = "general") -> Optional[Dict]:
    """Search Google and extract content (sync)."""
    return asyncio.run(search_google_async(query, source_type))


if __name__ == "__main__":
    # Test
    result = search_google("mistallbowl1 twitter", "twitter")
    if result:
        print(f"Found: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet']}")
        print(f"Content preview: {result['content'][:200]}...")
