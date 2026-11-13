"""
YouTube video scraper - downloads videos, transcripts, and metadata.
"""
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import yt_dlp
from datetime import datetime


class YouTubeScraper:
    """Scrape YouTube videos and extract metadata."""
    
    def __init__(self, output_dir: str = "data/raw_videos", cookies_file: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file = cookies_file
        # Request pacing helps avoid temporary YouTube session rate limits.
        self._request_sleep_seconds = float(os.getenv("YT_DLP_SLEEP_REQUESTS", "1.5"))
        # Newer yt-dlp expects js_runtimes config values to be dict-like objects.
        runtime_names = [
            r.strip() for r in os.getenv("YT_DLP_JS_RUNTIMES", "node,deno").split(",") if r.strip()
        ]
        self._js_runtimes = {name: {} for name in runtime_names}

    def _with_common_ydl_opts(self, ydl_opts: Dict) -> Dict:
        """Apply common yt-dlp options for JS runtime and throttling."""
        merged = dict(ydl_opts)
        merged.setdefault('sleep_interval_requests', self._request_sleep_seconds)
        merged.setdefault('js_runtimes', self._js_runtimes)
        return merged

    def _build_collection_url_candidates(self, source_url: str) -> List[str]:
        """Build URL candidates for channel/hashtag collections.

        yt-dlp can be sensitive to collection URL variants; try common alternates
        while preserving caller intent.
        """
        candidates: List[str] = []

        def _add(url: str):
            if url and url not in candidates:
                candidates.append(url)

        normalized = source_url.strip()
        _add(normalized)

        # Support hashtag shorts URL variants, e.g. /hashtag/funny/shorts
        hashtag_match = re.search(r"(https?://(?:www\.)?youtube\.com/hashtag/[^/?#]+)", normalized)
        if hashtag_match:
            hashtag_base = hashtag_match.group(1)
            _add(hashtag_base)
            _add(f"{hashtag_base}/shorts")

        # Channel handle variants commonly used in this repo.
        if "/@" in normalized and normalized.endswith("/shorts"):
            _add(normalized.rsplit("/shorts", 1)[0])

        return candidates

    def _extract_collection_entries(self, collection_url: str, max_videos: int) -> List[Dict]:
        """Extract collection entries from channel/hashtag style URLs."""
        ydl_opts_meta = self._with_common_ydl_opts({
            'skip_download': True,
            'quiet': True,
            'ignoreerrors': True,
            'playlistend': max_videos,
            'extract_flat': 'in_playlist',  # Light-weight: just get metadata, don't fully extract each entry
            'no_warnings': True,
        })

        for candidate in self._build_collection_url_candidates(collection_url):
            with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
                try:
                    info = ydl.extract_info(candidate, download=False)
                except Exception as e:
                    print(f"[WARN] Failed to fetch collection URL {candidate}: {e}")
                    continue

            entries = (info.get('entries') or [])[:max_videos] if info else []
            # Filter out None entries and entries without valid IDs
            entries = [e for e in entries if e and isinstance(e, dict) and e.get('id')]
            if entries:
                if candidate != collection_url:
                    print(f"[INFO] Using fallback collection URL: {candidate}")
                return entries

        return []
        
    def scrape_channel(self, channel_url: str, max_videos: int = 100) -> List[Dict]:
        """Download videos and metadata from a channel."""

        videos = []

        # Step 1: fetch collection metadata only (channel, handle, or hashtag feeds)
        entries = self._extract_collection_entries(channel_url, max_videos)
        if not entries:
            print("Error fetching channel/hashtag collection entries")
            return videos

        # Step 2: download each video using the robust multi-strategy scrape_video
        for entry in entries:
            if not entry or not isinstance(entry, dict):
                continue
            video_id = entry.get('id')
            if not video_id:
                continue
            existing = list(self.output_dir.glob(f"{video_id}.*"))
            already_downloaded = any(p.suffix in ('.mp4', '.webm', '.mkv') for p in existing)
            metadata_path = self.output_dir / f"{video_id}.json"
            # Skip only when the actual media file already exists.
            if already_downloaded:
                print(f"\nSkipping (already downloaded): {entry.get('title')} ({video_id})")
                if metadata_path.exists():
                    import json as _json
                    with open(metadata_path, encoding="utf-8") as f:
                        videos.append(_json.load(f))
                else:
                    videos.append(self._extract_metadata(entry))
                continue
            if metadata_path.exists():
                print(f"\nRetrying download for cached metadata-only entry: {entry.get('title')} ({video_id})")
            print(f"\nProcessing: {entry.get('title')} ({video_id})")
            result = self.scrape_video(
                f"https://www.youtube.com/watch?v={video_id}",
                download_video=True
            )
            if result:
                videos.append(result)
            else:
                # Still record metadata-only entry so the pipeline can process it
                metadata = self._extract_metadata(entry)
                if not metadata_path.exists():
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                videos.append(metadata)

        return videos
    
    def scrape_video(self, video_url: str, download_video: bool = True) -> Optional[Dict]:
        """
        Download video and metadata with multiple fallback strategies.
        
        Args:
            video_url: YouTube video URL (supports regular videos and Shorts)
            download_video: Whether to download the actual video file
        """
        
        # Detect if it's a Shorts URL and normalize it
        is_shorts = '/shorts/' in video_url
        if is_shorts:
            print(f"[SHORTS] Detected YouTube Shorts URL")
            video_id = video_url.split('/shorts/')[-1].split('?')[0]
            print(f"   Video ID: {video_id}")
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            video_id = None
            for part in video_url.split('?')[0].split('/'):
                if part and part not in ('www.youtube.com', 'youtube.com', 'watch', 'youtu.be'):
                    video_id = part
            if 'v=' in video_url:
                video_id = video_url.split('v=')[-1].split('&')[0]

        # Skip download if file already exists
        if video_id:
            metadata_path = self.output_dir / f"{video_id}.json"
            existing = list(self.output_dir.glob(f"{video_id}.*"))
            has_video = any(p.suffix in ('.mp4', '.webm', '.mkv') for p in existing)
            if has_video:
                print(f"[SKIP] Already downloaded: {video_id}")
                if metadata_path.exists():
                    import json as _json
                    with open(metadata_path, encoding="utf-8") as f:
                        return _json.load(f)
                return None
            elif metadata_path.exists() and not download_video:
                print(f"[SKIP] Using cached metadata (no video file): {video_id}")
                import json as _json
                with open(metadata_path, encoding="utf-8") as f:
                    return _json.load(f)
            elif metadata_path.exists():
                print(f"[RETRY] Cached metadata exists but no video file: {video_id}")
        
        # Try multiple strategies in order of reliability
        strategies = [
            ('cookies', self._try_with_cookies),
            ('android_vr', self._try_android_vr_client),
            ('ios', self._try_ios_client),
            ('android_music', self._try_android_music_client),
            ('basic', self._try_basic_download),
        ]
        
        for strategy_name, strategy_func in strategies:
            print(f"\n[STRATEGY] Trying strategy: {strategy_name}")
            try:
                result = strategy_func(video_url, download_video)
                if result:
                    print(f"[OK] Success with {strategy_name} strategy!")
                    return result
            except Exception as e:
                print(f"   [FAILED] {strategy_name} failed: {str(e)[:100]}")
                continue
        
        print(f"\n[FAILED] All download strategies failed")
        print(f"\n[TIP] Export cookies from your browser to cookies.txt")
        print(f"   Chrome: https://chrome.google.com/webstore (search 'get cookies.txt')")
        print(f"   Firefox: https://addons.mozilla.org (search 'cookies.txt')")
        if video_id:
            metadata_path = self.output_dir / f"{video_id}.json"
            if metadata_path.exists():
                print(f"[FALLBACK] Returning cached metadata for {video_id}")
                import json as _json
                with open(metadata_path, encoding="utf-8") as f:
                    return _json.load(f)
        return None
    
    def _try_with_cookies(self, video_url: str, download_video: bool) -> Optional[Dict]:
        """Try download using browser cookies (most reliable for 403 errors)."""
        
        # Check for cookies file
        cookies_path = self.cookies_file
        if not cookies_path:
            # Try common locations
            possible_cookies = [
                'cookies.txt',
                'youtube_cookies.txt',
                Path.home() / 'cookies.txt',
                self.output_dir.parent / 'cookies.txt',
            ]
            for path in possible_cookies:
                if Path(path).exists():
                    cookies_path = str(path)
                    print(f"   Found cookies at: {path}")
                    break
        
        if not cookies_path or not Path(cookies_path).exists():
            raise Exception("No cookies file found")
        
        ydl_opts = self._with_common_ydl_opts({
            'format': 'bv*[height<=1080]+ba/b[height<=1080]/best',
            'cookiefile': cookies_path,
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        })
        
        return self._download_with_opts(video_url, ydl_opts, download_video)
    
    def _try_android_vr_client(self, video_url: str, download_video: bool) -> Optional[Dict]:
        """Try using Android VR client (often bypasses 403 errors)."""
        
        ydl_opts = self._with_common_ydl_opts({
            'format': 'bv*[height<=720]+ba/b[height<=720]/best',
            'extractor_args': {'youtube': {'player_client': ['android_vr']}},
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        })
        
        return self._download_with_opts(video_url, ydl_opts, download_video)
    
    def _try_ios_client(self, video_url: str, download_video: bool) -> Optional[Dict]:
        """Try using iOS client."""
        
        ydl_opts = self._with_common_ydl_opts({
            'format': 'bv*[height<=1080]+ba/b[height<=1080]/best',
            'extractor_args': {'youtube': {'player_client': ['ios']}},
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        })
        
        return self._download_with_opts(video_url, ydl_opts, download_video)
    
    def _try_android_music_client(self, video_url: str, download_video: bool) -> Optional[Dict]:
        """Try using Android Music client."""
        
        ydl_opts = self._with_common_ydl_opts({
            'format': 'bv*[height<=720]+ba/b[height<=720]/best',
            'extractor_args': {'youtube': {'player_client': ['android_music']}},
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        })
        
        return self._download_with_opts(video_url, ydl_opts, download_video)
    
    def _try_basic_download(self, video_url: str, download_video: bool) -> Optional[Dict]:
        """Try basic download with simple format selection."""
        
        ydl_opts = self._with_common_ydl_opts({
            'format': 'best[height<=720]/best',
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        })
        
        return self._download_with_opts(video_url, ydl_opts, download_video)
    
    def _download_with_opts(self, video_url: str, ydl_opts: Dict, download_video: bool) -> Optional[Dict]:
        """Common download logic with given options."""
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, extract info
            info = ydl.extract_info(video_url, download=False)
            
            if not info:
                raise Exception("Failed to extract video info")
            
            # Download if requested
            if download_video:
                print(f"   [DOWNLOAD] Downloading video...")
                ydl.download([video_url])
                print(f"   [OK] Video downloaded")
            
            # Extract and save metadata
            metadata = self._extract_metadata(info)
            video_id = info['id']
            metadata_path = self.output_dir / f"{video_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
    
    def _extract_metadata(self, info: Dict) -> Dict:
        """Extract relevant metadata from video info."""
        
        if not info or not isinstance(info, dict):
            return {}
        
        # Get transcript from subtitles
        transcript = self._get_transcript(info)
        
        return {
            'video_id': info.get('id'),
            'title': info.get('title'),
            'description': info.get('description'),
            'duration': info.get('duration'),
            'upload_date': info.get('upload_date'),
            'uploader': info.get('uploader'),
            'channel': info.get('channel'),
            'view_count': info.get('view_count'),
            'like_count': info.get('like_count'),
            'tags': info.get('tags', []),
            'categories': info.get('categories', []),
            'transcript': transcript,
            'thumbnail': info.get('thumbnail'),
            'webpage_url': info.get('webpage_url') or (
                f"https://www.youtube.com/watch?v={info.get('id')}" if info.get('id') else None
            ),
        }
    
    def _get_transcript(self, info: Dict) -> str:
        """Extract transcript from subtitles if available."""
        
        subtitles = info.get('subtitles', {})
        auto_captions = info.get('automatic_captions', {})
        
        # Try manual subtitles first, then auto-generated
        for subs in [subtitles, auto_captions]:
            if 'en' in subs:
                # Get the first subtitle format
                sub_data = subs['en'][0] if isinstance(subs['en'], list) else subs['en']
                
                # If data is already downloaded
                if 'data' in sub_data:
                    return self._parse_subtitle_data(sub_data['data'])
                    
        return ""
    
    def _parse_subtitle_data(self, data: str) -> str:
        """Parse subtitle data to plain text."""
        import re
        
        # Remove VTT/SRT formatting
        text = re.sub(r'\d+:\d+:\d+[.,]\d+ --> \d+:\d+:\d+[.,]\d+', '', data)
        text = re.sub(r'<[^>]+>', '', text)  # Remove tags
        text = re.sub(r'\d+\n', '', text)  # Remove line numbers
        text = ' '.join(text.split())  # Clean whitespace
        
        return text


if __name__ == "__main__":
    # Example usage
    scraper = YouTubeScraper(output_dir="data/raw_videos")
    
    # Scrape a single video
    video_url = "https://www.youtube.com/watch?v=EXAMPLE"
    # metadata = scraper.scrape_video(video_url)
    
    # Scrape channel
    channel_url = "https://www.youtube.com/@veritasium"
    # videos = scraper.scrape_channel(channel_url, max_videos=10)
    
    print("Scraper ready. Uncomment the examples above to test.")
