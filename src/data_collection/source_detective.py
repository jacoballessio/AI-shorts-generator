"""
Source Detective - reverse-engineers research sources from videos.
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime
import anthropic
from openai import OpenAI

try:
    import torch
except ImportError:
    torch = None


_NEWS_STOPWORDS = frozenset({
    'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'to', 'of', 'and', 'or',
    'but', 'so', 'if', 'is', 'are', 'was', 'were', 'be', 'do', 'did', 'has', 'have',
    'here', 'there', 'this', 'that', 'these', 'those', 'now', 'then', 'just', 'also',
    'one', 'two', 'three', 'first', 'second', 'last', 'new', 'old', 'big', 'small',
    'we', 'you', 'he', 'she', 'they', 'it', 'my', 'your', 'his', 'her', 'our',
    'very', 'really', 'actually', 'basically', 'literally', 'obviously', 'exactly',
    'right', 'left', 'up', 'down', 'well', 'good', 'bad', 'great', 'amazing',
    'today', 'yesterday', 'tomorrow', 'maybe', 'like', 'love', 'people', 'things',
    'way', 'time', 'year', 'day', 'week', 'month', 'thing', 'every', 'some', 'what',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'american', 'national', 'federal', 'united',
})


class NewsAPISearcher:
    """Search for news articles via newsapi.org/v2/everything."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict]:
        """Return a list of source dicts for articles matching *query*.

        Args:
            query: Free-text search string.
            from_date / to_date: ISO-8601 date strings (YYYY-MM-DD).
            max_results: Maximum number of articles to return.
        """
        import urllib.request
        import urllib.parse

        params: Dict = {
            "q": query,
            "apiKey": self.api_key,
            "sortBy": "relevance",
            "language": "en",
            "pageSize": max_results,
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        url = self.BASE_URL + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "AI-shorts-generator/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[NewsAPI] Request failed: {exc}")
            return []

        sources = []
        for article in data.get("articles", []):
            article_url = article.get("url", "")
            if not article_url or article_url == "https://removed.com":
                continue
            sources.append({
                "type": "news",
                "content": article_url,
                "url": article_url,
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source_name": article.get("source", {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "confidence": 0.85,
                "timestamp": None,
                "citation": "{} — {}".format(
                    article.get("source", {}).get("name", "News"),
                    (article.get("title") or "")[:80],
                ),
            })
        return sources

    @staticmethod
    def build_query(title: str, transcript: str) -> str:
        """Build a concise, clean search query from a video title/transcript.

        Strips commentary/reaction framing ("SLAMS", "reminisces about",
        "WOW:", etc.) that is specific to the content creator's take and
        would prevent matching the underlying news story in NewsAPI.
        """
        clean = re.sub(r"#\w+", "", title).strip()
        # Strip surrounding quote characters that would force exact-phrase search
        clean = clean.strip('"\'')
        # Strip leading all-caps exclamations and platform prefixes
        clean = re.sub(
            r"^(?:BREAKING|WATCH|NEW|EXCLUSIVE|WOW|OMG|JUST IN|MUST WATCH)[:\s!]+",
            "", clean, flags=re.IGNORECASE,
        ).strip()
        # Strip creator-name watermarks at start/end (e.g. "Brian Tyler Cohen:", "|BTC")
        clean = re.sub(r"^[A-Za-z ]{3,30}:\s+", "", clean).strip()
        clean = re.sub(r"\s*\|[^|]+$", "", clean).strip()
        # Strip reaction/commentary framing verbs that obscure the real topic
        # e.g. "GOP Senator reminisces about when meth was homemade"
        #   -> "meth was homemade" (drop the subject+verb)
        clean = re.sub(
            r"^.{0,50}?\b(?:reminisces?|reminds?|explains?|admits?|reveals?|slams?|blasts?|"
            r"rips?|destroys?|responds?|reacts?|speaks? out|calls? out|claps? back|"
            r"goes? off|lays? out|breaks? down|sounds? off|fires? back|doubles? down|"
            r"drops?|says?|claims?|insists?|argues?|confirms?|denies?|defends?|attacks?)\b"
            r"[^:]{0,40}?(?:about|on|over|at|to|that|when|how|why|for)\s+",
            "", clean, flags=re.IGNORECASE,
        ).strip()
        if len(clean) >= 15:
            return clean[:150]
        # Fallback: take the first sentence of the transcript
        first_sent = re.split(r"[.!?]", transcript.strip())[0].strip()
        return first_sent[:150] if first_sent else transcript[:150].strip()

    @staticmethod
    def keyword_query(title: str, transcript: str) -> str:
        """Extract the most informative keywords from title + transcript for a
        broad fallback search when the cleaned title returns 0 results."""
        _SKIP = _NEWS_STOPWORDS | {
            "meth", "drug", "drugs", "says", "said", "like", "just", "didn",
            "doesn", "wasn", "isn", "about", "going", "getting", "thing",
            "homemade", "homegrown",
        }
        combined = title + " " + transcript
        # Prefer proper nouns first, then fall back to any long word
        proper = re.findall(r"\b([A-Z][a-z]{3,})\b", combined)
        words = re.findall(r"\b([a-zA-Z]{5,})\b", combined.lower())
        seen: set = set()
        chosen: List[str] = []
        for w in proper + words:
            wl = w.lower()
            if wl not in _SKIP and wl not in seen:
                seen.add(wl)
                chosen.append(w)
            if len(chosen) >= 5:
                break
        return " ".join(chosen)


_DDG_STATE: Dict = {"last_call": 0.0}  # module-level rate-limit state for DDG
_DDG_MIN_INTERVAL = 0.45  # seconds between DDG requests

# Module-level Playwright browser shared across all GoogleNewsRSSSearcher instances
_PW_STATE: Dict = {"playwright": None, "browser": None}


def _get_pw_browser():
    """Return a long-lived Playwright Chromium browser, launching it on first call.

    Returns None if Playwright is not installed.
    """
    if _PW_STATE["browser"] is not None:
        return _PW_STATE["browser"]
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        _PW_STATE["playwright"] = pw
        _PW_STATE["browser"] = browser
        return browser
    except Exception:
        return None


def _close_pw_browser() -> None:
    """Shut down the shared Playwright browser (call at program exit if desired)."""
    if _PW_STATE["browser"]:
        try:
            _PW_STATE["browser"].close()
        except Exception:
            pass
        _PW_STATE["browser"] = None
    if _PW_STATE["playwright"]:
        try:
            _PW_STATE["playwright"].stop()
        except Exception:
            pass
        _PW_STATE["playwright"] = None


class GoogleNewsRSSSearcher:
    """Search Google News via the free, unlimited RSS feed — no API key required.

    Uses feedparser (already in requirements) to decode the RSS; falls back to
    xml.etree.ElementTree if feedparser is somehow unavailable.

    Date filtering is done via Google search operators (``after:`` / ``before:``)
    appended to the query string, which Google News RSS honours.
    """

    _SEARCH_URL = (
        "https://news.google.com/rss/search"
        "?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    def search(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        max_results: int = 8,
    ) -> List[Dict]:
        """Return article dicts for *query*, optionally filtered by *from_date* / *to_date*.

        Dates are ISO-8601 strings (YYYY-MM-DD).  Filtering is done post-fetch
        on the RSS ``pubDate`` field because Google News RSS does not reliably
        honour ``after:`` / ``before:`` search operators.  If date filtering
        would eliminate all results (e.g. for older videos beyond the RSS
        horizon), the filter is relaxed and all results are returned so the
        caller still gets some signal.
        """
        import urllib.request
        import urllib.parse

        url = self._SEARCH_URL.format(query=urllib.parse.quote_plus(query))
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AI-shorts/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = resp.read()
        except Exception as exc:
            print(f"[GoogleNews] Fetch failed: {exc}")
            return []

        results = self._parse_rss(raw, max_results=max_results * 3)  # over-fetch for filtering

        # Post-fetch date filtering on pubDate
        if from_date or to_date:
            filtered = self._filter_by_date(results, from_date, to_date)
            # Only apply filter if it doesn't eliminate all results
            if filtered:
                results = filtered

        return results[:max_results]

    @staticmethod
    def _filter_by_date(
        articles: List[Dict],
        from_date: Optional[str],
        to_date: Optional[str],
    ) -> List[Dict]:
        """Keep articles whose pubDate falls within [from_date, to_date]."""
        from email.utils import parsedate_to_datetime
        from datetime import timezone

        def _parse(dt_str: str) -> Optional[datetime]:
            try:
                dt = parsedate_to_datetime(dt_str)
                # Normalise to UTC-aware for comparison
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                try:
                    return datetime.strptime(dt_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    return None

        lo = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if from_date else None
        hi = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if to_date else None
        # Add one day to hi so that "to_date" is inclusive
        if hi:
            from datetime import timedelta
            hi = hi + timedelta(days=1)

        out: List[Dict] = []
        for a in articles:
            pub = _parse(a.get("published_at", ""))
            if pub is None:
                out.append(a)  # can't determine date — include it
                continue
            if lo and pub < lo:
                continue
            if hi and pub > hi:
                continue
            out.append(a)
        return out

    @staticmethod
    def _resolve_google_url(gurl: str, title: str, source_name: str = "") -> str:
        """Resolve a Google News redirect URL to the real publisher article URL.

        Primary strategy: Playwright headless Chromium — navigate to the URL and
        wait for the JS redirect to complete, then read the final ``page.url``.
        A single shared browser instance is reused for the whole run so startup
        cost is paid only once.

        Fallback: DuckDuckGo HTML search when Playwright is unavailable or fails,
        rate-limited to avoid 403s.
        """
        if "news.google.com" not in gurl:
            return gurl

        # --- Primary: Playwright ---
        browser = _get_pw_browser()
        if browser is not None:
            try:
                page = browser.new_page()
                page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
                page.goto(gurl, wait_until="commit", timeout=15000)
                # Wait until the URL is no longer on a Google domain (JS redirect)
                try:
                    page.wait_for_url(
                        lambda url: url is not None and "google.com" not in url,
                        timeout=10000,
                    )
                except Exception:
                    pass  # Timeout is fine — read whatever URL we landed on
                final = page.url
                page.close()
                if final and "google.com" not in final and final.startswith("http"):
                    return final
            except Exception as exc:
                print(f"[GoogleNews] Playwright resolve failed: {exc}")

        # --- Fallback: DuckDuckGo HTML search ---
        import time
        import urllib.request
        import urllib.parse
        from urllib.error import HTTPError

        query = re.sub(r"\s*-\s*\w[\w ]{1,30}$", "", title).strip()[:100]
        ddg_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        def _do_fetch() -> Optional[str]:
            elapsed = time.monotonic() - _DDG_STATE["last_call"]
            if elapsed < _DDG_MIN_INTERVAL:
                time.sleep(_DDG_MIN_INTERVAL - elapsed)
            _DDG_STATE["last_call"] = time.monotonic()
            req = urllib.request.Request(ddg_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", "ignore")

        html: Optional[str] = None
        try:
            html = _do_fetch()
        except HTTPError as exc:
            if exc.code == 403:
                print(f"[GoogleNews] DDG 403 – backing off 3s and retrying…")
                time.sleep(3.0)
                try:
                    html = _do_fetch()
                except Exception as exc2:
                    print(f"[GoogleNews] DDG retry failed: {exc2}")
            else:
                print(f"[GoogleNews] DDG lookup failed: {exc}")
        except Exception as exc:
            print(f"[GoogleNews] DDG lookup failed: {exc}")

        if not html:
            return gurl

        matches = re.findall(r"/l/\?uddg=([^&\"]+)", html)
        real_urls = [
            urllib.parse.unquote(m)
            for m in matches
            if "google.com" not in urllib.parse.unquote(m)
            and urllib.parse.unquote(m).startswith("http")
        ]
        if not real_urls:
            return gurl

        source_kw = source_name.lower().replace(" ", "")[:20] if source_name else ""
        if source_kw:
            preferred = [u for u in real_urls if source_kw in u.lower()]
            if preferred:
                return preferred[0]

        return real_urls[0]

    @staticmethod
    def _parse_rss(raw: bytes, max_results: int) -> List[Dict]:
        """Parse RSS bytes into article dicts using feedparser (preferred) or
        xml.etree.ElementTree as a fallback."""
        try:
            import feedparser  # already in requirements

            feed = feedparser.parse(raw)
            results: List[Dict] = []
            for entry in feed.entries[:max_results]:
                link = entry.get("link", "") or ""
                title = entry.get("title", "") or ""
                source_name = ""
                src = entry.get("source")
                if isinstance(src, dict):
                    source_name = src.get("title", "")
                pub = entry.get("published", "") or ""
                if not link or not link.startswith("http"):
                    continue
                results.append({
                    "type": "news",
                    "content": link,
                    "url": link,
                    "title": title,
                    "description": "",
                    "source_name": source_name,
                    "published_at": pub,
                    "confidence": 0.80,
                    "timestamp": None,
                    "citation": f"{source_name} — {title[:80]}",
                })
            return results
        except ImportError:
            pass  # fall through to ElementTree

        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            print(f"[GoogleNews] XML parse error: {exc}")
            return []

        results = []
        for item in root.findall(".//item")[:max_results]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            # ElementTree sometimes misses bare <link> text; scan children
            if not link:
                for child in item:
                    if child.tag == "link":
                        link = (child.text or "").strip()
                        break
            if not link or not link.startswith("http"):
                continue
            source_el = item.find("source")
            source_name = ""
            if source_el is not None and source_el.text:
                source_name = source_el.text.strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            results.append({
                "type": "news",
                "content": link,
                "url": link,
                "title": title,
                "description": "",
                "source_name": source_name,
                "published_at": pub_date,
                "confidence": 0.80,
                "timestamp": None,
                "citation": f"{source_name} — {title[:80]}",
            })
        return results

    @staticmethod
    def resolve_urls(articles: List[Dict]) -> None:
        """Resolve Google News redirect URLs in-place for the given articles.

        Call this AFTER ranking/deduplication so we only pay the Playwright
        cost for articles we actually keep.
        """
        for a in articles:
            gurl = a.get("url", "")
            if "news.google.com" in gurl:
                resolved = GoogleNewsRSSSearcher._resolve_google_url(
                    gurl, a.get("title", ""), a.get("source_name", "")
                )
                a["url"] = resolved
                a["content"] = resolved


class SourceDetective:
    """Detect and extract research sources from video content."""
    
    def __init__(self, provider: str = "local", newsapi_key: Optional[str] = None):
        """Initialize with LLM provider.

        Args:
            provider: 'local' (default, uses transformers), 'anthropic', 'openai', or 'none'
            newsapi_key: Optional newsapi.org API key for news-article search.
        """
        
        self.provider = provider
        self.client = None
        self.model = None
        self.tokenizer = None
        self.llm_model = None
        
        if provider == "local":
            # Use local LLM via transformers
            try:
                if torch is None:
                    raise ImportError("PyTorch not available")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print("   Loading local LLM for source detection...")
                model_name = "microsoft/Phi-3.5-mini-instruct"  # Small, fast model
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                print("   ✅ Local LLM loaded")
            except Exception as e:
                print(f"   ⚠️  Failed to load local LLM: {e}")
                print("   Implicit source detection will be skipped")
                self.provider = "none"
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = "claude-3-5-sonnet-20241022"
            else:
                print("   ⚠️  No ANTHROPIC_API_KEY found, implicit source detection disabled")
                self.provider = "none"
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model = "gpt-4o"
            else:
                print("   ⚠️  No OPENAI_API_KEY found, implicit source detection disabled")
                self.provider = "none"

        # Google News RSS — free, unlimited, no key needed
        self.gnews = GoogleNewsRSSSearcher()

        # NewsAPI — optional supplement (100 req/day on free tier)
        self.newsapi: Optional[NewsAPISearcher] = (
            NewsAPISearcher(newsapi_key) if newsapi_key else None
        )

    def detect_sources(
        self,
        transcript: str,
        description: str,
        labels: List[Dict],
        upload_date: Optional[str] = None,
        topic: str = "",
    ) -> List[Dict]:
        """
        Detect all sources referenced in a video.
        
        Args:
            transcript: Video transcript text
            description: Video description
            labels: Visual frame labels with descriptions
            upload_date: Video upload date (YYYYMMDD format) for temporal context
        
        Returns:
            List of {type, content, confidence, timestamp, citation}
        """
        
        sources = []
        
        # 1. Explicit citations in transcript
        explicit = self._find_explicit_citations(transcript)
        sources.extend(explicit)
        
        # 2. URLs in description
        urls = self._extract_urls(description)
        sources.extend(urls)
        
        # 3. On-screen text from visual labels
        onscreen = self._find_onscreen_sources(labels)
        sources.extend(onscreen)
        
        # 3.5. Wikipedia sources for named entities
        wikipedia = self._find_wikipedia_sources(transcript, labels, upload_date, description)
        sources.extend(wikipedia)

        # 3.6. Google News RSS + NewsAPI articles for news/politics content
        news_articles = self._find_news_articles(transcript, description, upload_date, topic)
        sources.extend(news_articles)

        # 4. Implicit references using LLM (with upload date context)
        if self.provider != "none":
            implicit = self._find_implicit_sources(transcript, labels, upload_date)
            sources.extend(implicit)
        
        # Deduplicate and sort by confidence
        sources = self._deduplicate_sources(sources)
        sources.sort(key=lambda x: x['confidence'], reverse=True)
        
        return sources
    
    def _find_explicit_citations(self, transcript: str) -> List[Dict]:
        """Find explicit citations like 'According to X' or 'Study shows'."""
        
        patterns = [
            r'according to ([^,.]+)',
            r'a study (?:published|from|in) ([^,.]+)',
            r'research(?:ers)? (?:at|from) ([^,.]+)',
            r'(?:published|reported) in ([^,.]+)',
            r'scientists at ([^,.]+)',
            r'paper (?:in|from) ([^,.]+)',
        ]
        
        sources = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                citation = match.group(1).strip()
                
                sources.append({
                    'type': 'explicit_citation',
                    'content': citation,
                    'confidence': 0.9,
                    'timestamp': None,
                    'citation': match.group(0),
                })
        
        return sources
    
    def _extract_urls(self, description: str) -> List[Dict]:
        """Extract URLs from video description, filtering out promotional links."""
        
        # Find all URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, description)
        
        # Promotional URL patterns to filter out
        promo_patterns = [
            r'/(subscribe|like|follow|join)',  # Social media CTAs
            r'nbcnews\.to/',  # URL shorteners for CTAs
            r'/mobile$',  # App download links
            r'podcast',  # Podcast (anywhere in URL)
            r'link\.',  # Newsletter signups
            r'apps\.',  # App stores
            r'heres-the-scoop',  # Specific podcast names
        ]
        
        sources = []
        
        for url in urls:
            # Skip promotional URLs
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in promo_patterns):
                continue
            
            # Parse domain and path
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            
            # Skip homepage URLs (just domain with / or empty path)
            if path in ['', '/']:
                continue
            
            # Detect Wikipedia
            if 'wikipedia.org' in domain:
                sources.append({
                    'type': 'wikipedia',
                    'content': url,
                    'domain': domain,
                    'confidence': 0.95,
                    'timestamp': None,
                    'citation': f"Wikipedia: {domain}",
                })
            else:
                sources.append({
                    'type': 'url',
                    'content': url,
                    'domain': domain,
                    'confidence': 0.95,
                    'timestamp': None,
                    'citation': f"Link in description: {domain}",
                })
        
        return sources
    
    def _find_onscreen_sources(self, labels: List[Dict]) -> List[Dict]:
        """Find sources from on-screen text in visual labels by extracting entities and searching."""
        
        sources = []
        seen_usernames = set()  # Avoid duplicates
        
        # Extract potential sources from visual descriptions
        for label in labels:
            description = label['description']
            text_visible = label.get('text_visible', '') or ''  # Ensure it's a string, not None
            
            # Combine all text for analysis
            all_text = f"{description} {text_visible}"
            
            # Try to parse JSON descriptions from VLM
            parsed_usernames = []
            try:
                if description.startswith('{'):
                    import json
                    desc_data = json.loads(description)
                    # Look for usernames in object alt attributes or other fields
                    if isinstance(desc_data.get('objects'), list):
                        for obj in desc_data['objects']:
                            if isinstance(obj, dict) and 'alt' in obj:
                                # Extract username-like values
                                alt_text = obj['alt']
                                if re.match(r'^[a-zA-Z0-9_]{3,20}$', alt_text):
                                    parsed_usernames.append(alt_text)
                    # Also check description text within JSON
                    if 'description' in desc_data:
                        description = desc_data['description']
            except (json.JSONDecodeError, AttributeError):
                pass  # Continue with original description
            
            # Look for Twitter/social media usernames with enhanced patterns
            twitter_users = re.findall(r'@([a-zA-Z0-9_]+)', all_text)
            
            # Match various username mention patterns
            twitter_users += re.findall(r'user named ["\']?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            twitter_users += re.findall(r'written by (?:a )?user named ["\']?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            twitter_users += re.findall(r'(?:posted|tweeted|created) by ["\']?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            twitter_users += re.findall(r'(?:username|handle|account)[:,]? ["\']?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            twitter_users += re.findall(r'(?:twitter|tweet|post).*?(?:by|from) ["\']?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            
            # Extract from "visible on the Twitter page of" patterns
            twitter_users += re.findall(r'Twitter page of (?:(?:the user|another user) )?["\']?@?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            twitter_users += re.findall(r'visible on the [^"]+ of ["\']?@?([a-zA-Z0-9_]+)["\']?', all_text, re.IGNORECASE)
            
            twitter_users += parsed_usernames  # Add usernames from JSON parsing
            
            # Extract potential watermarks from OCR text (short alphanumeric strings)
            if text_visible and 'OCR:' in text_visible:
                ocr_part = text_visible.split('OCR:')[1]
                # Look for short words that might be usernames/watermarks (4-15 chars)
                potential_usernames = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]{3,14})\b', ocr_part)
                
                # Expanded list of common words to filter out
                common_words = {
                    'the', 'and', 'for', 'symbol', 'female', 'male', 'venus', 'mars', 
                    'with', 'that', 'this', 'have', 'from', 'more', 'would', 'their', 
                    'there', 'which', 'about', 'could', 'should', 'gender', 'genders',
                    'plav', 'play', 'choose', 'anything', 'neptune', 'jupiter', 'saturn',
                    'uranus', 'mercury', 'earth', 'planet', 'planets', 'symbo', 'symbcl', 
                    'svmbol', 'swmpol', 'symbclior', 'macrd', 'mans', 'Vann'
                }
                
                for username in potential_usernames:
                    # Filter out common words and very short garbled text
                    if (username.lower() not in common_words and 
                        len(username) >= 5 and  # Require at least 5 chars for watermarks
                        not re.match(r'^[a-z]{4}$', username.lower())):  # Filter simple 4-letter words
                        twitter_users.append(username)
            
            # Deduplicate and add sources
            for username in twitter_users:
                if username and username.lower() not in seen_usernames:
                    seen_usernames.add(username.lower())
                    sources.append({
                        'type': 'twitter',
                        'content': f'https://twitter.com/{username}',
                        'username': username,
                        'confidence': 0.85 if '@' in all_text else 0.65,  # Higher confidence if @ symbol present
                        'timestamp': label['timestamp'],
                        'citation': f'Twitter user @{username} visible at {label["timestamp"]:.1f}s',
                    })
            
            # Look for websites/domains mentioned
            domains = re.findall(r'([a-zA-Z0-9-]+\\.(?:com|org|edu|net|gov))', all_text, re.IGNORECASE)
            for domain in domains:
                sources.append({
                    'type': 'website',
                    'content': f'https://{domain}',
                    'domain': domain,
                    'confidence': 0.80,
                    'timestamp': label['timestamp'],
                    'citation': f'Website {domain} visible at {label["timestamp"]:.1f}s',
                })
            
            # Look for research papers/publications (only when explicitly cited, not casual descriptions)
            # Match patterns like 'according to a study titled "X"' or 'paper called "X"'
            paper_patterns = [
                r'(?:according to|cited|titled|named|called)\s+(?:a|the)?\s*(?:study|paper|research)[^"\')]*["\']([^"\']{})["\']',
                r'(?:study|paper|research)\s+(?:titled|named|called)[^"\')]*["\']([^"\']{})["\']',
            ]
            for pattern in paper_patterns:
                paper_indicators = re.findall(pattern, description, re.IGNORECASE)
                for paper_title in paper_indicators:
                    if len(paper_title) > 15:  # Stricter length check
                        sources.append({
                            'type': 'research_paper',
                            'content': paper_title,
                            'search_query': f'"{paper_title}" research paper',
                            'confidence': 0.75,
                            'timestamp': label['timestamp'],
                            'citation': f'Research paper "{paper_title[:50]}..." visible at {label["timestamp"]:.1f}s',
                        })
        
        return sources
    
    def _find_wikipedia_sources(
        self,
        transcript: str,
        labels: List[Dict],
        upload_date: Optional[str] = None,
        description: str = "",
    ) -> List[Dict]:
        """Find Wikipedia background links for multi-word proper-noun phrases.

        Requires at least two capitalised words to avoid matching sentence-start
        common words like 'Looking', 'Bill', 'Here', etc.
        """
        sources: List[Dict] = []

        # Skip Wikipedia — Google News RSS (always available) is higher quality
        if self.gnews or self.newsapi:
            return sources

        all_content = transcript + " " + description
        for label in labels:
            all_content += " " + (label.get("text_visible") or "")

        # Require at least TWO consecutive capitalized words (real proper nouns)
        candidates = re.findall(
            r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b", all_content
        )

        seen: set = set()
        for phrase in candidates:
            low = phrase.lower()
            if low in _NEWS_STOPWORDS or low in seen:
                continue
            seen.add(low)
            sources.append({
                "type": "wikipedia",
                "content": f"Wikipedia: {phrase}",
                "url": f"https://en.wikipedia.org/wiki/{phrase.replace(' ', '_')}",
                "confidence": 0.50,
                "timestamp": None,
                "citation": f"Background: {phrase}",
            })
            if len(sources) >= 3:  # small cap — news articles are preferred
                break

        return sources
    
    def _find_news_articles(
        self,
        transcript: str,
        description: str,
        upload_date: Optional[str],
        topic: str,
    ) -> List[Dict]:
        """Search NewsAPI for articles relevant to this video's content.

        Fires when the topic is news/politics OR the transcript contains
        political/current-events keywords, as long as `self.newsapi` is set.
        Also runs a second query on key named entities for broader coverage.
        """
        _NEWS_KEYWORDS = {
            "trump", "biden", "harris", "congress", "senate", "president",
            "election", "legislation", "bill", "vote", "court", "supreme",
            "newsom", "administration", "democrat", "republican", "policy",
            "government", "minister", "parliament", "white house", "nato",
            "iran", "ukraine", "russia", "china", "israel", "gaza", "war",
        }
        topic_lower = topic.lower()
        is_news = "news" in topic_lower or "politic" in topic_lower
        content_lower = (transcript + " " + description).lower()
        has_news_content = any(kw in content_lower for kw in _NEWS_KEYWORDS)

        if not (is_news or has_news_content):
            return []

        # Date window: 7 days before upload → upload date
        # This ensures retrieved articles were published near the video's creation
        from_date: Optional[str] = None
        to_date: Optional[str] = None
        if upload_date:
            try:
                from datetime import timedelta
                date_obj = datetime.strptime(upload_date, "%Y%m%d")
                to_date = date_obj.strftime("%Y-%m-%d")
                from_date = (date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
            except Exception:
                pass

        results: List[Dict] = []
        seen_urls: set = set()

        def _run_query(q: str, max_r: int = 8) -> None:
            if not q.strip():
                return
            all_articles: List[Dict] = []
            # Google News RSS — always available, free, unlimited
            gnews_results = self.gnews.search(q, from_date=from_date, to_date=to_date, max_results=max_r)
            all_articles.extend(gnews_results)
            # NewsAPI — optional supplement
            na_results: List[Dict] = []
            if self.newsapi:
                na_results = self.newsapi.search(q, from_date=from_date, to_date=to_date, max_results=max_r)
                all_articles.extend(na_results)
            print(f"[News] '{q[:70]}' → {len(all_articles)} article(s) (gnews={len(gnews_results)}" +
                  (f", newsapi={len(na_results)}" if self.newsapi else "") + ")")
            for a in all_articles:
                url = a.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(a)

        # Primary query: cleaned video title (commentary framing stripped)
        title_line = re.split(r"\s+#", description.split("\n")[0])[0].strip()[:200]
        primary_query = NewsAPISearcher.build_query(title_line, transcript)
        _run_query(primary_query, max_r=8)

        # If the title query found nothing, try a keyword-based fallback
        if not results:
            kw_query = NewsAPISearcher.keyword_query(title_line, transcript)
            if kw_query and kw_query.lower() != primary_query.lower():
                _run_query(kw_query, max_r=8)

        # Secondary query: key named entities from transcript (people, places, orgs)
        # Extract 2–4 word proper-noun phrases not already in the primary query
        entities = re.findall(
            r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b",
            transcript + " " + title_line,
        )
        primary_lower = primary_query.lower()
        unique_entities = []
        seen_ent: set = set()
        for e in entities:
            el = e.lower()
            if el not in seen_ent and el not in _NEWS_STOPWORDS and e.lower() not in primary_lower:
                seen_ent.add(el)
                unique_entities.append(e)
            if len(unique_entities) >= 3:
                break

        if unique_entities:
            entity_query = " ".join(unique_entities[:2])
            _run_query(entity_query, max_r=5)

        return results

    def _find_implicit_sources(self, transcript: str, labels: List[Dict], upload_date: Optional[str] = None) -> List[Dict]:
        """Use LLM to detect implicit references and suggest likely sources."""
        
        # Extract on-screen text from labels
        onscreen_text = ""
        for label in labels:
            text_visible = label.get('text_visible', '')
            if text_visible:
                onscreen_text += f"\n[{label.get('timestamp', 0)}s] {text_visible}"
        
        # Format upload date for context
        date_context = ""
        if upload_date:
            try:
                from datetime import datetime
                date_obj = datetime.strptime(upload_date, "%Y%m%d")
                date_context = f"\n\nVideo upload date: {date_obj.strftime('%B %d, %Y')} (search for sources published around this time)"
            except:
                pass
        
        prompt = f"""Analyze this video content and identify implicit sources that could be found through search.

Look for:
1. FACTUAL CLAIMS that could be verified:
   - News events (e.g., "missiles intercepted in Doha" → search "missiles intercepted Doha March 2026")
   - Statistics or data points (e.g., "unemployment is at 5%" → search for economic reports)
   - Scientific facts or discoveries (e.g., "new study shows..." → search for research papers)
   - Historical events or dates mentioned
   - Geographic locations and proper nouns that might have Wikipedia articles

2. SOURCES to search for:
   - News articles or reports (type: "news")
   - Wikipedia articles for topics/locations/events (type: "wikipedia")
   - Research papers or studies (type: "study")
   - Official statements or documents (type: "report")
   - Expert interviews or quotes (type: "interview"){date_context}

Transcript:
{transcript[:2000]}

On-screen text:{onscreen_text[:1000] if onscreen_text else " (none)"}

For each claim that could have a verifiable source, provide:
1. The specific claim/fact
2. Search terms to find the source (include date/timeframe if relevant)
3. Source type (news/wikipedia/study/report/interview)
4. Confidence level (high/medium/low)

Output as JSON list:
[{{"claim": "...", "search_terms": "...", "source_type": "...", "confidence": "high"}}]
"""
        
        try:
            if self.provider == "local":
                # Use local LLM
                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4000)
                
                if self.llm_model.device.type == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                content = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
            
            # Parse JSON response
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            results = json.loads(content.strip())
            
            sources = []
            confidence_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            
            for result in results:
                sources.append({
                    'type': result.get('source_type', 'implicit_reference'),
                    'content': result['search_terms'],
                    'claim': result['claim'],
                    'confidence': confidence_map.get(result['confidence'], 0.5),
                    'timestamp': None,
                    'citation': f"Implied source: {result['claim'][:60]}...",
                })
            
            return sources
            
        except Exception as e:
            print(f"Error finding implicit sources: {e}")
            return []
    
    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """Remove duplicate sources based on content similarity."""
        
        unique = []
        seen = set()
        
        for source in sources:
            # Create a simple hash of the content
            content_hash = source['content'].lower()[:100]
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(source)
        
        return unique


if __name__ == "__main__":
    # Example usage
    detective = SourceDetective(provider="anthropic")
    
    # Example data
    transcript = """
    According to a study published in Nature, researchers at MIT discovered...
    Recent research shows that this phenomenon occurs at specific wavelengths...
    """
    
    description = """
    Links:
    - Paper: https://nature.com/articles/example
    - More info: https://mit.edu/research
    """
    
    labels = [
        {'timestamp': 45.2, 'description': 'Screenshot of research paper from Nature journal'},
        {'timestamp': 120.5, 'description': 'Graph showing data from the study'},
    ]
    
    # sources = detective.detect_sources(transcript, description, labels)
    # print(json.dumps(sources, indent=2))
    
    print("Source Detective ready. Uncomment examples to test.")
