"""
Media Labeler Module
Labels images and videos from articles using VLM and video_labeler.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import aiohttp
import hashlib

# Lazy imports for video_labeler (loaded when needed)
_VIDEO_LABELER_LOADED = False
VideoLabeler = None
LabelerConfig = None
VLMAnalyzer = None


def _ensure_video_labeler_loaded():
    """Lazy load video_labeler module."""
    global _VIDEO_LABELER_LOADED, VideoLabeler, LabelerConfig, VLMAnalyzer
    
    if _VIDEO_LABELER_LOADED:
        return
    
    # Add video_labeler to path
    video_labeler_path = Path(__file__).parent.parent / "video_labeler"
    if video_labeler_path.exists() and str(video_labeler_path) not in sys.path:
        sys.path.insert(0, str(video_labeler_path))
    
    try:
        from video_labeler import VideoLabeler as VL, LabelerConfig as LC
        from video_labeler.vlm_analyzer import VLMAnalyzer as VLM
        
        VideoLabeler = VL
        LabelerConfig = LC
        VLMAnalyzer = VLM
        _VIDEO_LABELER_LOADED = True
    except ImportError as e:
        raise ImportError(
            f"Failed to import video_labeler. Make sure it's installed: {e}\n"
            f"Expected path: {video_labeler_path}"
        )


class LabeledMedia:
    """Represents a labeled media asset (image or video)."""
    
    def __init__(self, 
                 media_type: str,  # "image" or "video"
                 url: str,
                 local_path: Optional[str] = None,
                 description: str = "",
                 analysis: Optional[Dict] = None):
        self.media_type = media_type
        self.url = url
        self.local_path = local_path
        self.description = description
        self.analysis = analysis  # Full analysis for videos, simple description for images
    
    @property
    def keywords(self) -> List[str]:
        """Extract keywords from analysis."""
        if self.media_type == "video" and self.analysis:
            return self.analysis.get('global_keywords', [])[:10]
        elif self.media_type == "image":
            # Extract key words from description
            words = self.description.lower().split()
            return [w for w in words if len(w) > 3][:10]
        return []
    
    @property
    def clip_worthy_segments(self) -> List[Dict]:
        """Get clip-worthy segments for videos."""
        if self.media_type == "video" and self.analysis:
            segments = self.analysis.get('segments', [])
            return [s for s in segments if s.get('clip_worthy', False)]
        return []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "media_type": self.media_type,
            "url": self.url,
            "local_path": self.local_path,
            "description": self.description,
            "keywords": self.keywords
        }
        
        if self.media_type == "video" and self.analysis:
            result["clip_worthy_segments"] = self.clip_worthy_segments
            result["duration"] = self.analysis.get('duration', 0)
            result["summary"] = self.analysis.get('summary', '')
        
        return result


class MediaLabeler:
    """
    Labels images and videos using VLM analysis.
    """
    
    def __init__(self, cache_dir: str = "./cache/media"):
        # Ensure video_labeler is loaded
        _ensure_video_labeler_loaded()
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize VLM for image labeling
        self.vlm_analyzer = VLMAnalyzer(
            provider="local",
            model="HuggingFaceTB/SmolVLM-Instruct"
        )
        
        # Initialize video labeler
        self.video_labeler_config = LabelerConfig(
            vlm_provider="local",
            vlm_model="HuggingFaceTB/SmolVLM-Instruct",
            frame_interval=120,  # Every 4 seconds
            max_frames=None,  # Analyze entire video
            use_whisper=True,
            whisper_model="base",
            analyze_actions=True,
            analyze_objects=True,
            analyze_scenes=True,
            generate_summary=True,
            generate_keywords=True,
            auto_segment=True,
            batch_size=5
        )
        self.video_labeler = None  # Lazy init
    
    async def label_media_from_article(self, 
                                      media_urls: Dict[str, List[str]],
                                      article_id: str) -> List[LabeledMedia]:
        """
        Download and label all media from an article.
        
        Args:
            media_urls: Dict with 'images', 'videos', 'youtube' keys
            article_id: Unique identifier for caching
            
        Returns:
            List of LabeledMedia objects
        """
        print(f"\n🏷️  Labeling media from article: {article_id}")
        
        labeled_media = []
        
        # Label images
        for img_url in media_urls.get('images', [])[:5]:  # Limit to 5 images
            try:
                labeled_img = await self._label_image(img_url, article_id)
                if labeled_img:
                    labeled_media.append(labeled_img)
            except Exception as e:
                print(f"  ⚠️ Failed to label image {img_url}: {e}")
        
        # Label videos
        for vid_url in media_urls.get('videos', [])[:3]:  # Limit to 3 videos
            try:
                labeled_vid = await self._label_video(vid_url, article_id)
                if labeled_vid:
                    labeled_media.append(labeled_vid)
            except Exception as e:
                print(f"  ⚠️ Failed to label video {vid_url}: {e}")
        
        print(f"  ✅ Labeled {len(labeled_media)} media assets")
        return labeled_media
    
    async def _label_image(self, url: str, article_id: str) -> Optional[LabeledMedia]:
        """Download and label an image using VLM."""
        # Generate cache filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        local_path = self.cache_dir / article_id / f"image_{url_hash}.jpg"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download image
        if not local_path.exists():
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            content = await response.read()
                            with open(local_path, 'wb') as f:
                                f.write(content)
                        else:
                            print(f"  ⚠️ Failed to download image: HTTP {response.status}")
                            return None
                except Exception as e:
                    print(f"  ⚠️ Download error: {e}")
                    return None
        
        # Label image with VLM
        try:
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Load and convert image to base64
            img = Image.open(local_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Generate description with VLM
            prompt = """Analyze this image and provide:
1. A detailed description of what's shown
2. Main objects/people/elements visible
3. The context or purpose of the image
4. Any text or signs visible

Provide a concise 1-2 sentence description."""
            
            description = self.vlm_analyzer._call_vlm(img_b64, prompt)
            
            return LabeledMedia(
                media_type="image",
                url=url,
                local_path=str(local_path),
                description=description
            )
            
        except Exception as e:
            print(f"  ⚠️ VLM labeling error: {e}")
            return None
    
    async def _label_video(self, url: str, article_id: str) -> Optional[LabeledMedia]:
        """Download and label a video using video_labeler."""
        # Generate cache filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        local_path = self.cache_dir / article_id / f"video_{url_hash}.mp4"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download video
        if not local_path.exists():
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                        if response.status == 200:
                            content = await response.read()
                            with open(local_path, 'wb') as f:
                                f.write(content)
                        else:
                            print(f"  ⚠️ Failed to download video: HTTP {response.status}")
                            return None
                except Exception as e:
                    print(f"  ⚠️ Download error: {e}")
                    return None
        
        # Label video with video_labeler
        try:
            # Initialize labeler if needed
            if self.video_labeler is None:
                self.video_labeler = VideoLabeler(self.video_labeler_config)
            
            # Analyze video
            print(f"  🎬 Analyzing video: {url}")
            analysis = self.video_labeler.analyze_video(
                str(local_path),
                output_dir=str(self.cache_dir / article_id / "analysis")
            )
            
            # Get summary description
            summary = analysis.get('summary', '')
            if not summary and analysis.get('segments'):
                # Use first segment description as fallback
                summary = analysis['segments'][0].get('description', 'Video content')
            
            return LabeledMedia(
                media_type="video",
                url=url,
                local_path=str(local_path),
                description=summary,
                analysis=analysis
            )
            
        except Exception as e:
            print(f"  ⚠️ Video labeling error: {e}")
            return None
    
    def find_best_media_for_segment(self, 
                                   segment_text: str,
                                   labeled_media: List[LabeledMedia],
                                   media_type: Optional[str] = None) -> Optional[LabeledMedia]:
        """
        Find the best matching media for a script segment.
        
        Args:
            segment_text: The script segment text
            labeled_media: List of labeled media to search
            media_type: Preferred type ("image" or "video"), or None for any
            
        Returns:
            Best matching LabeledMedia or None
        """
        if not labeled_media:
            return None
        
        # Simple keyword matching
        segment_keywords = set(segment_text.lower().split())
        
        best_match = None
        best_score = 0
        
        for media in labeled_media:
            if media_type and media.media_type != media_type:
                continue
            
            # Calculate overlap
            media_keywords = set(' '.join(media.keywords).lower().split())
            overlap = len(segment_keywords & media_keywords)
            
            # Bonus for description match
            if any(word in media.description.lower() for word in segment_keywords):
                overlap += 2
            
            if overlap > best_score:
                best_score = overlap
                best_match = media
        
        return best_match if best_score > 0 else None
