"""
Media Generator Module
Handles TTS voice generation, image fetching from web, and Stable Diffusion image generation.
"""

import os
import asyncio
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import subprocess

# TTS imports
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    # Check for new ElevenLabs client API
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    # Fallback to legacy API
    try:
        from elevenlabs import generate, save
        ELEVENLABS_AVAILABLE = True
    except ImportError:
        ELEVENLABS_AVAILABLE = False

# Whisper for accurate word timings
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Image processing
from PIL import Image
import io

# Stable Diffusion - lazy import (only when needed)
SD_AVAILABLE = False
try:
    # Test if imports work without actually importing
    import importlib.util
    spec = importlib.util.find_spec("diffusers")
    if spec is not None:
        spec_torch = importlib.util.find_spec("torch")
        if spec_torch is not None:
            SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False

load_dotenv()


@dataclass
class MediaAssets:
    """Container for all generated media assets."""
    audio_path: str
    audio_duration: float
    images: List[str]  # Paths to image files
    video_clips: List[str]  # Paths to video clip files
    captions: List[Dict]  # Word-level timing for captions
    metadata: Dict


class MediaGenerator:
    """
    Generates all media assets needed for video creation:
    - TTS voiceover
    - Images from web and Stable Diffusion
    - Caption timing data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', './cache/media'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TTS settings
        self.tts_provider = self.config.get('tts_provider', 'edge')  # edge, elevenlabs
        self.voice_name = self.config.get('voice_name', 'en-US-AriaNeural')  # Edge TTS voice
        self.elevenlabs_voice_id = self.config.get('elevenlabs_voice_id', 'nPczCjzI2devNBz1zQrb')  # George - free tier
        self.elevenlabs_model = self.config.get('elevenlabs_model', 'eleven_turbo_v2_5')  # ElevenLabs model
        
        # Image settings
        self.use_stable_diffusion = self.config.get('use_stable_diffusion', True)
        self.sd_model_path = self.config.get('sd_model_path', 'runwayml/stable-diffusion-v1-5')
        
        # Initialize Stable Diffusion if available and enabled
        self.sd_pipe = None
        if self.use_stable_diffusion and SD_AVAILABLE:
            self._init_stable_diffusion()
    
    def _init_stable_diffusion(self):
        """Initialize Stable Diffusion pipeline."""
        try:
            print("🎨 Initializing Stable Diffusion (this may take a moment)...")
            
            # Import here to avoid module-level import issues
            from diffusers import StableDiffusionPipeline
            import torch
            
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                self.sd_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None  # Disable for speed
            )
            
            if torch.cuda.is_available():
                self.sd_pipe = self.sd_pipe.to("cuda")
                print("✅ Stable Diffusion loaded on GPU")
            else:
                print("✅ Stable Diffusion loaded on CPU (slower)")
        except Exception as e:
            print(f"⚠️ Failed to load Stable Diffusion: {e}")
            print("   Continuing without Stable Diffusion image generation...")
            self.sd_pipe = None
    
    async def generate_all_assets(self, script: Dict, video_id: str) -> MediaAssets:
        """
        Generate all media assets for a video.
        
        Args:
            script: VideoScript dictionary
            video_id: Unique identifier for this video
            
        Returns:
            MediaAssets object with paths to all generated files
        """
        print(f"\n🎬 Generating media assets for video: {video_id}")
        
        # Create output directory for this video
        video_dir = self.output_dir / video_id
        video_dir.mkdir(exist_ok=True)
        
        # Extract media URLs from topic if available
        topic = script.get('metadata', {}).get('topic_full', {})
        topic_metadata = topic.get('metadata', {})
        source_media = topic_metadata.get('media_urls', {'images': [], 'videos': [], 'youtube': []})
        
        # Generate TTS audio
        audio_path, audio_duration, captions = await self.generate_tts(
            script['full_script'],
            output_path=str(video_dir / "voiceover.mp3")
        )
        
        # Generate images and video clips in parallel
        num_media_needed = max(5, int(audio_duration / 3))  # Media every 3 seconds
        
        # Download media from research sources first
        source_images = await self._download_source_images(source_media.get('images', []), video_dir)
        source_videos = await self._download_source_videos(source_media.get('videos', []), video_dir)
        youtube_clips = await self._download_youtube_clips(source_media.get('youtube', []), video_dir, max_clips=2)
        
        # Generate additional images if needed
        images_needed = max(0, num_media_needed - len(source_images) - len(source_videos) - len(youtube_clips))
        generated_images = await self.generate_images(
            sd_prompts=script.get('image_prompts', []),
            search_queries=script.get('search_queries', []),
            num_images=images_needed,
            output_dir=video_dir
        ) if images_needed > 0 else []
        
        # Download Pexels video clips
        pexels_videos = await self._download_pexels_videos(
            search_queries=script.get('search_queries', []),
            num_clips=min(2, max(0, num_media_needed // 3)),  # ~1/3 can be video clips
            output_dir=video_dir
        )
        
        all_images = source_images + generated_images
        all_videos = source_videos + youtube_clips + pexels_videos
        
        print(f"  ✅ Total media: {len(all_images)} images, {len(all_videos)} video clips")
        
        return MediaAssets(
            audio_path=audio_path,
            audio_duration=audio_duration,
            images=all_images,
            video_clips=all_videos,
            captions=captions,
            metadata={
                "video_id": video_id,
                "script_word_count": script.get('metadata', {}).get('word_count', 0),
                "num_images": len(all_images),
                "num_video_clips": len(all_videos),
                "tts_provider": self.tts_provider,
                "source_images": len(source_images),
                "source_videos": len(source_videos),
                "youtube_clips": len(youtube_clips)
            }
        )
    
    async def generate_tts(self, text: str, output_path: str) -> Tuple[str, float, List[Dict]]:
        """
        Generate TTS audio from text.
        
        Args:
            text: Script text
            output_path: Where to save the audio file
            
        Returns:
            Tuple of (audio_path, duration_seconds, word_timings)
        """
        print(f"🎙️ Generating TTS using {self.tts_provider}...")
        
        if self.tts_provider == "edge" and EDGE_TTS_AVAILABLE:
            return await self._generate_edge_tts(text, output_path)
        elif self.tts_provider == "elevenlabs":
            if not ELEVENLABS_AVAILABLE:
                print(f"  ⚠️ ElevenLabs not available, falling back to Edge TTS")
                print(f"  Install with: pip install elevenlabs")
                if EDGE_TTS_AVAILABLE:
                    return await self._generate_edge_tts(text, output_path)
            else:
                return await self._generate_elevenlabs_tts(text, output_path)
        
        # Fallback to edge-tts
        if EDGE_TTS_AVAILABLE:
            print(f"  ⚠️ TTS provider '{self.tts_provider}' not available, using Edge TTS")
            return await self._generate_edge_tts(text, output_path)
        else:
            raise RuntimeError("No TTS provider available. Install edge-tts or elevenlabs.")
    
    async def _generate_edge_tts(self, text: str, output_path: str) -> Tuple[str, float, List[Dict]]:
        """Generate TTS using Edge TTS (free)."""
        
        communicate = edge_tts.Communicate(text, self.voice_name)
        
        # Generate with word timings
        word_timings = []
        audio_written = False
        
        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                    audio_written = True
                elif chunk["type"] == "WordBoundary":
                    word_timings.append({
                        "word": chunk.get("text", ""),
                        "start": chunk.get("offset", 0) / 10000000,  # Convert to seconds
                        "duration": chunk.get("duration", 0) / 10000000
                    })
        
        if not audio_written:
            raise RuntimeError("No audio data generated from Edge TTS")
        
        # Get audio duration
        duration = self._get_audio_duration(output_path)
        
        # If we didn't get word timings, create basic captions from text
        if not word_timings:
            print(f"  ⚠️ No word boundaries received, trying Whisper for accurate timing...")
            word_timings = await self._extract_whisper_timings(output_path, text)
            
            # Fall back to estimation if Whisper fails
            if not word_timings:
                print(f"  ⚠️ Whisper unavailable, creating estimated captions...")
                word_timings = self._estimate_word_timings(text, duration)
        
        print(f"  ✅ TTS generated: {duration:.2f}s, {len(word_timings)} words")
        return output_path, duration, word_timings
    
    async def _generate_elevenlabs_tts(self, text: str, output_path: str) -> Tuple[str, float, List[Dict]]:
        """Generate TTS using ElevenLabs (paid, higher quality) with Whisper for word timings."""
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment")
        
        try:
            # Try modern ElevenLabs client API first
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings
            from elevenlabs.core.api_error import ApiError
            
            print(f"  🎤 Generating ElevenLabs TTS with voice {self.elevenlabs_voice_id}...")
            
            client = ElevenLabs(api_key=api_key)
            
            # Generate audio with voice settings
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id=self.elevenlabs_voice_id,
                model_id=self.elevenlabs_model,
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True
                ),
                output_format="mp3_44100_128"
            )
            
            # Save audio stream
            with open(output_path, 'wb') as f:
                for chunk in audio_generator:
                    f.write(chunk)
            
            print(f"    ✅ Audio saved to {output_path}")
            
        except ImportError as e:
            # Missing ElevenLabs package or old version
            print(f"  ⚠️  ElevenLabs import error: {e}")
            print(f"  Falling back to Edge TTS...")
            return await self._generate_edge_tts(text, output_path)
            
        except Exception as e:
            # API errors (subscription issues, invalid voice, etc.)
            error_msg = str(e)
            if 'subscription' in error_msg.lower():
                print(f"  ⚠️  ElevenLabs subscription error: Voice not available on your plan")
                print(f"  💡 Try a free-tier voice like Rachel (21m00Tcm4TlvDq8ikWAM)")
            elif 'authorization' in error_msg.lower() or '401' in error_msg:
                print(f"  ⚠️  ElevenLabs API key issue: {error_msg[:100]}...")
            else:
                print(f"  ⚠️  ElevenLabs error: {error_msg[:100]}...")
            
            print(f"  Falling back to Edge TTS...")
            return await self._generate_edge_tts(text, output_path)
        
        # Get duration
        duration = self._get_audio_duration(output_path)
        
        # Use Whisper for accurate word timings
        word_timings = await self._extract_whisper_timings(output_path, text)
        
        if not word_timings:
            # Fallback: estimate word timings
            print("    ⚠️  Whisper unavailable, estimating word timings...")
            words = text.split()
            time_per_word = duration / len(words) if words else 0
            
            for i, word in enumerate(words):
                word_timings.append({
                    "word": word,
                    "start": i * time_per_word,
                    "duration": time_per_word
                })
        
        print(f"  ✅ ElevenLabs TTS generated: {duration:.2f}s with {len(word_timings)} word timings")
        return output_path, duration, word_timings
    
    async def _extract_whisper_timings(self, audio_path: str, original_text: str) -> List[Dict]:
        """Extract word-level timings from audio using Whisper."""
        if not WHISPER_AVAILABLE:
            return []
        
        try:
            print(f"    🎤 Running Whisper to extract word timings...")
            
            # Load Whisper model (use small model for speed, or base for better accuracy)
            # Run in thread pool since Whisper is CPU-intensive
            import torch
            
            def run_whisper():
                # Use 'tiny' for speed, 'base' for better accuracy, 'small' for balance
                model = whisper.load_model("base")
                result = model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    language="en"
                )
                return result
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_whisper)
            
            # Extract word timings
            word_timings = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word_timings.append({
                                "word": word_info['word'].strip(),
                                "start": word_info['start'],
                                "duration": word_info['end'] - word_info['start']
                            })
            
            if word_timings:
                print(f"    ✅ Whisper extracted {len(word_timings)} word timings")
                return word_timings
            else:
                print(f"    ⚠️ Whisper didn't return word timings")
                return []
                
        except Exception as e:
            print(f"    ⚠️ Whisper extraction failed: {e}")
            return []
    
    def _estimate_word_timings(self, text: str, duration: float) -> List[Dict]:
        """Create estimated word timings when TTS doesn't provide them."""
        words = text.split()
        if not words:
            return []
        
        # Estimate timing with some variation based on word length
        # Longer words take slightly more time to say
        import re
        
        # Calculate relative durations based on word length and punctuation
        word_weights = []
        for word in words:
            # Base weight on character count
            weight = len(word)
            
            # Add pause for punctuation
            if any(p in word for p in ['.', '!', '?', ',', ';', ':']):
                weight += 3  # Extra pause for punctuation
            
            word_weights.append(max(weight, 1))
        
        total_weight = sum(word_weights)
        time_per_weight = duration / total_weight
        
        word_timings = []
        current_time = 0
        
        for word, weight in zip(words, word_weights):
            word_duration = time_per_weight * weight
            word_timings.append({
                "word": word,
                "start": current_time,
                "duration": word_duration
            })
            current_time += word_duration
        
        return word_timings
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return float(result.stdout.strip())
        except:
            # Fallback: estimate from word count (150 words/min)
            return 10.0  # Default estimate
    
    async def generate_images(self, sd_prompts: List[str], search_queries: List[str],
                             num_images: int, output_dir: Path) -> List[str]:
        """
        Generate images from both Stable Diffusion and web searches.
        
        Args:
            sd_prompts: Prompts for Stable Diffusion
            search_queries: Queries for web image search
            num_images: Total number of images needed
            output_dir: Where to save images
            
        Returns:
            List of image file paths
        """
        print(f"🖼️ Generating {num_images} images...")
        
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        image_paths = []
        
        # Generate with Stable Diffusion first (if available and prompts provided)
        sd_count = min(len(sd_prompts), num_images // 2) if self.sd_pipe and sd_prompts else 0
        if sd_count > 0:
            sd_images = await self._generate_sd_images(sd_prompts[:sd_count], images_dir)
            image_paths.extend(sd_images)
        
        # Fetch remaining from web, passing sd_prompts for fallback
        web_count = num_images - len(image_paths)
        if web_count > 0:
            web_images = await self._fetch_web_images(search_queries, web_count, images_dir, sd_prompts)
            image_paths.extend(web_images)
        
        print(f"  ✅ Generated {len(image_paths)} images")
        return image_paths
    
    async def _generate_sd_images(self, prompts: List[str], output_dir: Path) -> List[str]:
        """Generate images using Stable Diffusion."""
        print(f"  🎨 Generating {len(prompts)} Stable Diffusion images...")
        
        if not self.sd_pipe:
            return []
        
        image_paths = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Generate image
                image = self.sd_pipe(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=768,
                    width=512  # 9:16 aspect ratio for shorts
                ).images[0]
                
                # Save
                output_path = output_dir / f"sd_{i:03d}.png"
                image.save(output_path)
                image_paths.append(str(output_path))
                
                print(f"    Generated: {prompt[:50]}...")
            except Exception as e:
                print(f"    ⚠️ SD generation failed for prompt {i}: {e}")
        
        return image_paths
    
    async def _fetch_web_images(self, queries: List[str], num_images: int,
                                output_dir: Path, sd_prompts: List[str] = None) -> List[str]:
        """Fetch images from web sources (Pexels, Unsplash)."""
        print(f"  🌐 Fetching {num_images} images from web...")
        
        if not queries:
            print(f"    ⚠️ No search queries provided for web images")
            queries = ["abstract", "nature", "technology"]  # Fallback queries
        
        print(f"    🔍 Using queries: {queries}")
        image_paths = []
        images_per_query = max(1, num_images // len(queries)) if queries else num_images
        
        for i, query in enumerate(queries):
            if len(image_paths) >= num_images:
                break
            
            # Try Pexels first
            pexels_images = await self._fetch_pexels_images(query, images_per_query)
            
            for j, image_data in enumerate(pexels_images):
                if len(image_paths) >= num_images:
                    break
                
                try:
                    # Download image
                    response = requests.get(image_data['url'], timeout=50)
                    if response.status_code == 200:
                        output_path = output_dir / f"web_{len(image_paths):03d}.jpg"
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        image_paths.append(str(output_path))
                except Exception as e:
                    print(f"    ⚠️ Failed to download image: {e}")
        
        # If we still need more images, use SD instead of placeholders
        if len(image_paths) < num_images:
            shortage = num_images - len(image_paths)
            print(f"    ⚠️ Only fetched {len(image_paths)}/{num_images} images from web")
            
            # Try to generate more with Stable Diffusion
            if self.sd_pipe and self.use_stable_diffusion:
                # Use sd_prompts if available, otherwise convert search queries to SD prompts
                prompts_for_fallback = []
                if sd_prompts:
                    # Cycle through sd_prompts if we need more than available
                    for i in range(shortage):
                        prompts_for_fallback.append(sd_prompts[i % len(sd_prompts)])
                elif queries:
                    # Convert search queries to SD prompts
                    for i in range(shortage):
                        query = queries[i % len(queries)]
                        sd_prompt = f"{query}, photorealistic, cinematic lighting, dramatic composition, high detail, 4k"
                        prompts_for_fallback.append(sd_prompt)
                
                if prompts_for_fallback:
                    print(f"    🎨 Generating {shortage} additional images with Stable Diffusion...")
                    additional_sd = await self._generate_sd_images(
                        prompts=prompts_for_fallback,
                        output_dir=output_dir
                    )
                    image_paths.extend(additional_sd)
                    print(f"    ✅ SD generated {len(additional_sd)} fallback images")
        
        # Only use placeholders as absolute last resort (if SD also failed)
        if len(image_paths) < num_images:
            shortage = num_images - len(image_paths)
            print(f"    ⚠️ Creating {shortage} placeholder images (both web and SD failed)")
            while len(image_paths) < num_images:
                placeholder_path = self._create_placeholder_image(
                    output_dir / f"placeholder_{len(image_paths):03d}.png"
                )
                image_paths.append(placeholder_path)
        
        return image_paths
    
    async def _fetch_pexels_images(self, query: str, num_images: int) -> List[Dict]:
        """Fetch images from Pexels API."""
        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            print(f"    ⚠️ PEXELS_API_KEY not found in environment")
            return []
        
        print(f"    🔍 Searching Pexels for '{query}'...")
        
        try:
            headers = {"Authorization": api_key}
            url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}&orientation=portrait"
            
            response = requests.get(url, headers=headers, timeout=50)
            print(f"    📡 Pexels API response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                photos = data.get("photos", [])
                print(f"    ✅ Found {len(photos)} images from Pexels")
                return [
                    {"url": photo["src"]["large2x"], "photographer": photo["photographer"]}
                    for photo in photos
                ]
            else:
                print(f"    ❌ Pexels API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"    ⚠️ Pexels API error: {e}")
        
        return []
    
    def _create_placeholder_image(self, output_path: Path) -> str:
        """Create a placeholder image."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create vertical image (9:16)
        img = Image.new('RGB', (512, 768), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        
        # Add text
        text = "AI Generated\nContent"
        bbox = draw.textbbox((0, 0), text, font=None)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((512 - text_width) // 2, (768 - text_height) // 2)
        draw.text(position, text, fill=(200, 200, 200))
        
        img.save(output_path)
        return str(output_path)
    
    async def _download_source_images(self, image_urls: List[str], output_dir: Path) -> List[str]:
        """Download images from research sources (Reddit, news, etc.)."""
        if not image_urls:
            return []
        
        print(f"  📥 Downloading {len(image_urls)} images from research sources...")
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        downloaded = []
        for i, url in enumerate(image_urls[:10]):  # Limit to 10 source images
            try:
                response = requests.get(url, timeout=50, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    # Detect file extension
                    ext = url.split('?')[0].split('.')[-1].lower()
                    if ext not in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                        ext = 'jpg'
                    
                    output_path = images_dir / f"source_{i:03d}.{ext}"
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    downloaded.append(str(output_path))
                    print(f"    ✅ Downloaded source image {i+1}/{len(image_urls)}")
            except Exception as e:
                print(f"    ⚠️ Failed to download image {url}: {e}")
        
        return downloaded
    
    async def _download_source_videos(self, video_urls: List[str], output_dir: Path) -> List[str]:
        """Download video clips from research sources (Reddit videos)."""
        if not video_urls:
            return []
        
        print(f"  📥 Downloading {len(video_urls)} videos from research sources...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        downloaded = []
        for i, url in enumerate(video_urls[:5]):  # Limit to 5 source videos
            try:
                response = requests.get(url, timeout=50, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    output_path = videos_dir / f"source_{i:03d}.mp4"
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    downloaded.append(str(output_path))
                    print(f"    ✅ Downloaded source video {i+1}/{len(video_urls)}")
            except Exception as e:
                print(f"    ⚠️ Failed to download video {url}: {e}")
        
        return downloaded
    
    async def _download_youtube_clips(self, youtube_urls: List[str], output_dir: Path, max_clips: int = 2) -> List[str]:
        """Download and clip YouTube videos using yt-dlp."""
        if not youtube_urls:
            return []
        
        print(f"  🎥 Downloading {min(len(youtube_urls), max_clips)} YouTube clips...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        downloaded = []
        for i, url in enumerate(youtube_urls[:max_clips]):
            try:
                import yt_dlp
                
                output_template = str(videos_dir / f"youtube_{i:03d}.%(ext)s")
                
                ydl_opts = {
                    'format': 'best[height<=720]',  # 720p or lower for faster download
                    'outtmpl': output_template,
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'postprocessors': [{
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4',
                    }],
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info).replace('.webm', '.mp4').replace('.mkv', '.mp4')
                    
                    if Path(filename).exists():
                        downloaded.append(filename)
                        print(f"    ✅ Downloaded YouTube clip {i+1}/{len(youtube_urls)}")
            except ImportError:
                print(f"    ⚠️ yt-dlp not installed. Install with: pip install yt-dlp")
                break
            except Exception as e:
                print(f"    ⚠️ Failed to download YouTube video {url}: {e}")
        
        return downloaded
    
    async def _download_pexels_videos(self, search_queries: List[str], num_clips: int, output_dir: Path) -> List[str]:
        """Download video clips from Pexels Video API."""
        if num_clips == 0 or not search_queries:
            return []
        
        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            return []
        
        print(f"  🎬 Downloading {num_clips} video clips from Pexels...")
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        downloaded = []
        clips_per_query = max(1, num_clips // len(search_queries))
        
        for query in search_queries:
            if len(downloaded) >= num_clips:
                break
            
            try:
                headers = {"Authorization": api_key}
                url = f"https://api.pexels.com/videos/search?query={query}&per_page={clips_per_query}&orientation=portrait"
                
                response = requests.get(url, headers=headers, timeout=50)
                if response.status_code == 200:
                    data = response.json()
                    videos = data.get("videos", [])
                    
                    for video in videos[:clips_per_query]:
                        if len(downloaded) >= num_clips:
                            break
                        
                        # Get the HD video file
                        video_files = video.get("video_files", [])
                        hd_file = next((f for f in video_files if f.get("quality") == "hd"), None)
                        if not hd_file:
                            hd_file = video_files[0] if video_files else None
                        
                        if hd_file:
                            video_url = hd_file.get("link")
                            try:
                                vid_response = requests.get(video_url, timeout=50)
                                if vid_response.status_code == 200:
                                    output_path = videos_dir / f"pexels_{len(downloaded):03d}.mp4"
                                    with open(output_path, 'wb') as f:
                                        f.write(vid_response.content)
                                    downloaded.append(str(output_path))
                                    print(f"    ✅ Downloaded Pexels video {len(downloaded)}/{num_clips}")
                            except Exception as e:
                                print(f"    ⚠️ Failed to download Pexels video: {e}")
            except Exception as e:
                print(f"    ⚠️ Pexels video API error for query '{query}': {e}")
        
        return downloaded
    
    def save_assets_manifest(self, assets: MediaAssets, output_path: str):
        """Save manifest of all generated assets."""
        manifest = {
            "audio_path": assets.audio_path,
            "audio_duration": assets.audio_duration,
            "images": assets.images,
            "num_captions": len(assets.captions),
            "metadata": assets.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"💾 Assets manifest saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example script
    script = {
        "full_script": "This is incredible news! Scientists have just made a breakthrough discovery. The implications are massive. You won't believe what this means for the future.",
        "image_prompts": [
            "scientists celebrating in laboratory, photorealistic, cinematic lighting",
            "futuristic technology, glowing particles, sci-fi",
            "excited crowd celebrating, dramatic, high energy"
        ],
        "search_queries": [
            "science breakthrough laboratory",
            "futuristic technology",
            "celebration excitement"
        ],
        "metadata": {"word_count": 25}
    }
    
    # Generate media
    generator = MediaGenerator()
    assets = asyncio.run(generator.generate_all_assets(script, "test_video_001"))
    
    print(f"\n✅ All assets generated:")
    print(f"  Audio: {assets.audio_path} ({assets.audio_duration:.2f}s)")
    print(f"  Images: {len(assets.images)} files")
    print(f"  Captions: {len(assets.captions)} words")
