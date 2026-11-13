"""
Research Agent Module
Autonomously discovers trending topics, news, and viral content across multiple platforms.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
try:
    import asyncpraw  # Async Reddit API
    ASYNCPRAW_AVAILABLE = True
except ImportError:
    ASYNCPRAW_AVAILABLE = False
    print("Note: asyncpraw not installed. Install with: pip install asyncpraw")

from dotenv import load_dotenv

load_dotenv()


class TopicCandidate:
    """Represents a potential video topic with metadata and scoring."""
    
    def __init__(self, title: str, description: str, source: str, 
                 engagement_score: float, trending_score: float, 
                 url: str = "", metadata: Dict = None):
        self.title = title
        self.description = description
        self.source = source
        self.engagement_score = engagement_score
        self.trending_score = trending_score
        self.url = url
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    @property
    def viral_potential(self) -> float:
        """
        Calculate overall viral potential (0-1 scale).
        Combines engagement and trending metrics.
        """
        return (self.engagement_score * 0.6 + self.trending_score * 0.4)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "engagement_score": self.engagement_score,
            "trending_score": self.trending_score,
            "viral_potential": self.viral_potential,
            "url": self.url,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ResearchAgent:
    """
    Autonomous agent that researches trending topics across multiple platforms.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sources_enabled = self.config.get('sources', ['reddit', 'google_trends', 'news'])
        self.viral_threshold = self.config.get('viral_threshold', 0.4)  # Lowered from 0.6 to get more results
        self.style = self.config.get('style', 'engaging')  # Video style preference
        
        # Initialize API clients
        self._init_reddit()
        self._init_news_api()
        
    def _init_reddit(self):
        """Initialize Reddit API client."""
        if not ASYNCPRAW_AVAILABLE:
            self.reddit = None
            return
        
        try:
            # Note: asyncpraw client will be created in async context
            self.reddit_config = {
                'client_id': os.getenv('REDDIT_CLIENT_ID'),
                'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
                'user_agent': 'AI-Shorts-Generator/1.0'
            }
            self.reddit = True  # Flag that config exists
        except Exception as e:
            print(f"Warning: Reddit API not configured: {e}")
            self.reddit = None
    
    def _init_news_api(self):
        """Initialize News API client."""
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            print("Warning: News API key not configured")
    
    async def research_topics(self, num_topics: int = 10) -> List[TopicCandidate]:
        """
        Main research method. Gathers trending topics from all enabled sources.
        
        Args:
            num_topics: Number of top topics to return
            
        Returns:
            List of TopicCandidate objects sorted by viral potential
        """
        print(f"🔍 Starting research across {len(self.sources_enabled)} sources...")
        
        all_candidates = []
        
        # Run research from all sources concurrently
        tasks = []
        if 'reddit' in self.sources_enabled and self.reddit:
            tasks.append(self._research_reddit())
        if 'google_trends' in self.sources_enabled:
            tasks.append(self._research_google_trends())
        if 'news' in self.sources_enabled and self.news_api_key:
            tasks.append(self._research_news())
        if 'twitter' in self.sources_enabled:
            tasks.append(self._research_twitter())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, list):
                all_candidates.extend(result)
            elif isinstance(result, Exception):
                print(f"Warning: Research source failed: {result}")
        
        # Apply style-based scoring adjustment
        for candidate in all_candidates:
            candidate.style_score = self._calculate_style_match(candidate)
            # Adjust viral potential based on style match
            candidate.adjusted_viral_potential = (candidate.viral_potential * 0.7) + (candidate.style_score * 0.3)
        
        # Sort by adjusted viral potential and filter
        candidates = sorted(all_candidates, key=lambda x: x.adjusted_viral_potential, reverse=True)
        candidates = [c for c in candidates if c.viral_potential >= self.viral_threshold]
        
        # Add diversity through weighted random sampling
        # Take more candidates than needed, then randomly sample with bias toward higher scores
        if len(candidates) > num_topics:
            # Take top candidates (3x the requested amount for good pool)
            pool_size = min(len(candidates), num_topics * 3)
            candidate_pool = candidates[:pool_size]
            
            # Weighted random selection - higher viral potential = higher probability
            # Use random scoring method since random.sample() doesn't support weights
            import random
            
            # Assign each candidate a random score weighted by adjusted viral potential
            scored_candidates = []
            for c in candidate_pool:
                # Use adjusted viral potential that includes style matching
                score_base = getattr(c, 'adjusted_viral_potential', c.viral_potential)
                # Random value weighted by score squared (favors higher scores)
                random_score = random.random() ** (1.0 / (score_base ** 2))
                scored_candidates.append((random_score, c))
            
            # Sort by random score and take top N
            scored_candidates.sort(key=lambda x: x[0])
            selected = [c for _, c in scored_candidates[:num_topics]]
            candidates = selected
        else:
            candidates = candidates[:num_topics]
        
        print(f"✅ Found {len(candidates)} high-potential topics (filtered for '{self.style}' style)")
        return candidates
    
    def _calculate_style_match(self, candidate: TopicCandidate) -> float:
        """
        Calculate how well a topic matches the desired video style.
        Returns score from 0.0 to 1.0.
        """
        # Style keyword matching
        style_keywords = {
            'engaging': ['shocking', 'surprising', 'unexpected', 'revealed', 'discover', 'secret', 'hidden', 'truth'],
            'dramatic': ['crisis', 'disaster', 'breaking', 'urgent', 'critical', 'warning', 'danger', 'threat'],
            'educational': ['how', 'why', 'explain', 'learn', 'understand', 'science', 'study', 'research', 'guide'],
            'humorous': ['funny', 'hilarious', 'comedy', 'joke', 'laugh', 'meme', 'wtf', 'absurd', 'ridiculous'],
            'inspirational': ['success', 'achievement', 'overcome', 'inspiring', 'motivate', 'triumph', 'hero', 'dream'],
            'controversial': ['debate', 'controversy', 'scandal', 'outrage', 'banned', 'censored', 'fight', 'vs'],
            'storytelling': ['story', 'tale', 'journey', 'experience', 'happened', 'lived', 'survived', 'witness']
        }
        
        keywords = style_keywords.get(self.style.lower(), [])
        if not keywords:
            return 0.5  # Neutral score for unknown styles
        
        # Check title and description for keyword matches
        text = (candidate.title + ' ' + candidate.description).lower()
        matches = sum(1 for keyword in keywords if keyword in text)
        
        # Calculate score (0.3 base + up to 0.7 from matches)
        match_score = min(matches / 5.0, 1.0)  # Normalize to max 1.0
        return 0.3 + (match_score * 0.7)
    
    async def _research_reddit(self) -> List[TopicCandidate]:
        """Research trending topics from Reddit."""
        print("📱 Researching Reddit...")
        candidates = []
        
        if not self.reddit or not ASYNCPRAW_AVAILABLE:
            print("  Reddit research skipped (not configured or asyncpraw not available)")
            return candidates
        
        # Target subreddits for viral content
        subreddits = [
            'all', 'technology', 'news', 'worldnews', 'entertainment',
            'movies', 'gaming', 'science', 'todayilearned', 'Futurology',
            'nextfuckinglevel', 'interestingasfuck', 'Damnthatsinteresting',
            'mildlyinteresting', 'explainlikeimfive'
        ]
        
        # Different sort methods for diversity
        sort_methods = ['hot', 'new', 'rising', 'top']
        
        try:
            # Create async Reddit client
            async with asyncpraw.Reddit(**self.reddit_config) as reddit:
                import random
                
                # Randomly select subreddits for diversity
                selected_subreddits = random.sample(subreddits, min(6, len(subreddits)))
                
                for subreddit_name in selected_subreddits:
                    subreddit = await reddit.subreddit(subreddit_name)
                    
                    # Randomly pick a sort method for variety
                    sort_method = random.choice(sort_methods)
                    
                    # Get posts based on sort method
                    if sort_method == 'hot':
                        posts = subreddit.hot(limit=15)
                    elif sort_method == 'new':
                        posts = subreddit.new(limit=15)
                    elif sort_method == 'rising':
                        posts = subreddit.rising(limit=15)
                    else:  # top
                        posts = subreddit.top('day', limit=15)
                    
                    async for post in posts:
                        if post.stickied:  # Skip pinned posts
                            continue
                        
                        # Calculate engagement score
                        engagement = (post.score + post.num_comments * 2) / 1000
                        engagement_score = min(engagement / 10, 1.0)  # More lenient normalization
                        
                        # Calculate trending score based on recency and velocity
                        post_age_hours = (datetime.now().timestamp() - post.created_utc) / 3600
                        trending_score = max(0, 1.0 - (post_age_hours / 24))  # Decay over 24h
                        
                        # Extract top comments for context
                        top_comments = await self._extract_top_comments(post)
                        
                        candidate = TopicCandidate(
                            title=post.title,
                            description=post.selftext[:500] if post.selftext else "",
                            source=f"reddit/r/{subreddit_name}",
                            engagement_score=engagement_score,
                            trending_score=trending_score,
                            url=f"https://reddit.com{post.permalink}",
                            metadata={
                                "upvotes": post.score,
                                "comments": post.num_comments,
                                "subreddit": subreddit_name,
                                "author": str(post.author),
                                "post_url": post.url,  # Original post URL (could be image/video)
                                "thumbnail": post.thumbnail if hasattr(post, 'thumbnail') and post.thumbnail not in ['self', 'default', 'nsfw'] else None,
                                "is_video": post.is_video if hasattr(post, 'is_video') else False,
                                "media_urls": self._extract_media_urls(post),
                                "top_comments": top_comments
                            }
                        )
                        candidates.append(candidate)
            
            print(f"  Found {len(candidates)} Reddit topics")
        except Exception as e:
            print(f"  Reddit research error: {e}")
        
        return candidates
    
    def _extract_media_urls(self, post) -> Dict[str, List[str]]:
        """Extract image and video URLs from a Reddit post."""
        media = {"images": [], "videos": [], "youtube": []}
        
        try:
            # Check post URL
            url = post.url if hasattr(post, 'url') else ""
            
            # YouTube links
            if any(domain in url for domain in ['youtube.com', 'youtu.be']):
                media["youtube"].append(url)
            
            # Direct image links
            elif any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                media["images"].append(url)
            
            # Reddit image hosting
            elif 'i.redd.it' in url or 'i.imgur.com' in url:
                media["images"].append(url)
            
            # Reddit video
            elif hasattr(post, 'is_video') and post.is_video:
                if hasattr(post, 'media') and post.media and 'reddit_video' in post.media:
                    media["videos"].append(post.media['reddit_video'].get('fallback_url', ''))
            
            # Gallery posts
            if hasattr(post, 'gallery_data') and post.gallery_data:
                # Multiple images in gallery
                if hasattr(post, 'media_metadata'):
                    for item_id in post.gallery_data['items']:
                        item = post.media_metadata.get(item_id['media_id'], {})
                        if 's' in item:
                            media["images"].append(item['s'].get('u', '').replace('&amp;', '&'))
        
        except Exception as e:
            pass  # Silently fail on media extraction
        
        return media
    
    async def _extract_top_comments(self, post) -> List[Dict[str, str]]:
        """Extract top comments from a Reddit post for additional context."""
        comments = []
        try:
            await post.comments.replace_more(limit=0)  # Don't expand "more comments"
            
            # Get top 5 comments
            for comment in post.comments[:5]:
                if hasattr(comment, 'body') and len(comment.body) > 20:
                    comments.append({
                        "text": comment.body[:300],  # Limit comment length
                        "score": comment.score if hasattr(comment, 'score') else 0
                    })
        except Exception as e:
            pass  # Silently fail on comment extraction
        
        return comments
    
    async def _scrape_article_content(self, url: str) -> Dict[str, any]:
        """Scrape full article content and media from a URL."""
        article_data = {
            "full_text": "",
            "key_points": [],
            "media_urls": {"images": [], "videos": [], "youtube": []},
            "scraped": False
        }
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return article_data
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract main article content
            article_text = ""
            
            # Try common article containers
            article_containers = [
                soup.find('article'),
                soup.find('div', class_=lambda x: x and ('article' in x.lower() or 'content' in x.lower() or 'story' in x.lower())),
                soup.find('div', id=lambda x: x and ('article' in x.lower() or 'content' in x.lower())),
            ]
            
            for container in article_containers:
                if container:
                    # Get all paragraphs
                    paragraphs = container.find_all('p')
                    if len(paragraphs) > 3:  # Likely the main content
                        article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
                        break
            
            # Fallback: get all paragraphs
            if not article_text:
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
            
            # Clean up text
            article_text = ' '.join(article_text.split()[:800])  # Limit to ~800 words
            
            # Extract key points (first few sentences from different paragraphs)
            paragraphs = soup.find_all('p')
            key_points = []
            for p in paragraphs[:10]:  # First 10 paragraphs
                text = p.get_text().strip()
                if 30 < len(text) < 200 and text not in key_points:
                    key_points.append(text)
                if len(key_points) >= 5:
                    break
            
            # Extract media
            # Images
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src and not any(x in src for x in ['logo', 'icon', 'avatar', 'pixel', '1x1']):
                    # Convert relative URLs to absolute
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        from urllib.parse import urljoin
                        src = urljoin(url, src)
                    
                    if src.startswith('http'):
                        article_data["media_urls"]["images"].append(src)
            
            # Videos and embeds
            for iframe in soup.find_all('iframe'):
                src = iframe.get('src', '')
                if 'youtube.com' in src or 'youtu.be' in src:
                    article_data["media_urls"]["youtube"].append(src)
            
            for video in soup.find_all('video'):
                src = video.get('src') or (video.find('source') and video.find('source').get('src'))
                if src:
                    article_data["media_urls"]["videos"].append(src)
            
            # Limit media
            article_data["media_urls"]["images"] = article_data["media_urls"]["images"][:10]
            article_data["media_urls"]["videos"] = article_data["media_urls"]["videos"][:5]
            article_data["media_urls"]["youtube"] = article_data["media_urls"]["youtube"][:3]
            
            article_data["full_text"] = article_text
            article_data["key_points"] = key_points
            article_data["scraped"] = len(article_text) > 200
            
        except Exception as e:
            print(f"  ⚠️ Article scraping failed for {url}: {e}")
        
        return article_data
    
    async def _research_google_trends(self) -> List[TopicCandidate]:
        """Research trending topics from Google Trends."""
        print("📈 Researching Google Trends...")
        candidates = []
        
        try:
            # Using unofficial Google Trends scraping
            url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=US"
            response = requests.get(url, timeout=50)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:10]
                
                for item in items:
                    title = item.find('title').text if item.find('title') else ""
                    description = item.find('description').text if item.find('description') else ""
                    link = item.find('link').text if item.find('link') else ""
                    
                    # Google Trends items are by definition trending
                    candidate = TopicCandidate(
                        title=title,
                        description=description,
                        source="google_trends",
                        engagement_score=0.8,  # High engagement assumed
                        trending_score=0.95,  # Very trending
                        url=link
                    )
                    candidates.append(candidate)
                
                print(f"  Found {len(candidates)} Google Trends topics")
        except Exception as e:
            print(f"  Google Trends research error: {e}")
        
        return candidates
    
    async def _research_news(self) -> List[TopicCandidate]:
        """Research trending news from News API."""
        print("📰 Researching news sources...")
        candidates = []
        
        if not self.news_api_key:
            return candidates
        
        try:
            # Get top headlines
            url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=50)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])[:15]
                
                for article in articles:
                    # Calculate engagement based on source popularity
                    source_tier = 0.7  # Could be enhanced with source ranking
                    
                    # Calculate trending based on publish time
                    pub_date = article.get('publishedAt', '')
                    try:
                        pub_datetime = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        hours_old = (datetime.now(pub_datetime.tzinfo) - pub_datetime).total_seconds() / 3600
                        trending_score = max(0, 1.0 - (hours_old / 12))
                    except:
                        trending_score = 0.5
                    
                    # Scrape full article content
                    article_url = article.get('url', '')
                    article_content = await self._scrape_article_content(article_url) if article_url else {}
                    
                    # Merge media from article scraping with existing
                    media_urls = article_content.get('media_urls', {'images': [], 'videos': [], 'youtube': []})
                    
                    candidate = TopicCandidate(
                        title=article.get('title', ''),
                        description=article.get('description', ''),
                        source=f"news/{article.get('source', {}).get('name', 'unknown')}",
                        engagement_score=source_tier,
                        trending_score=trending_score,
                        url=article_url,
                        metadata={
                            "author": article.get('author', ''),
                            "published": pub_date,
                            "source": article.get('source', {}).get('name', ''),
                            "article_content": article_content.get('full_text', ''),
                            "key_points": article_content.get('key_points', []),
                            "media_urls": media_urls,
                            "image_url": article.get('urlToImage', '')
                        }
                    )
                    candidates.append(candidate)
                
                print(f"  Found {len(candidates)} news topics")
        except Exception as e:
            print(f"  News research error: {e}")
        
        return candidates
    
    async def _research_twitter(self) -> List[TopicCandidate]:
        """Research trending topics from Twitter/X."""
        print("🐦 Researching Twitter trends...")
        candidates = []
        
        # Note: Twitter API v2 requires authentication
        # This is a placeholder for future implementation
        # You would need to set up Twitter API credentials
        
        print("  Twitter integration pending - requires API setup")
        return candidates
    
    def get_best_topic(self, candidates: List[TopicCandidate] = None) -> Optional[TopicCandidate]:
        """
        Get the single best topic for video creation.
        
        Args:
            candidates: List of candidates (if None, will research)
            
        Returns:
            The highest-scoring TopicCandidate
        """
        if not candidates:
            # Run research synchronously
            candidates = asyncio.run(self.research_topics())
        
        return candidates[0] if candidates else None
    
    def save_research(self, candidates: List[TopicCandidate], filepath: str):
        """Save research results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_topics": len(candidates),
            "topics": [c.to_dict() for c in candidates]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Research saved to {filepath}")


# Example usage
if __name__ == "__main__":
    agent = ResearchAgent()
    
    # Run research
    topics = asyncio.run(agent.research_topics(num_topics=10))
    
    # Display results
    print("\n🎯 Top Trending Topics:\n")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. [{topic.viral_potential:.2f}] {topic.title}")
        print(f"   Source: {topic.source}")
        print(f"   {topic.description[:100]}...")
        print()
    
    # Save to file
    agent.save_research(topics, "research_results.json")
