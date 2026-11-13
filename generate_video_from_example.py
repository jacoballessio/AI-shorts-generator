#!/usr/bin/env python3
"""
Generate Video from Training Example JSON
Creates a video from the training dataset format with scene descriptions, UI elements, and transcripts.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil

# MoviePy for video composition
try:
    from moviepy import (
        VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
        TextClip, concatenate_videoclips, ColorClip
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        from moviepy.editor import (
            VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
            TextClip, concatenate_videoclips, ColorClip
        )
        MOVIEPY_AVAILABLE = True
    except ImportError:
        print("Error: MoviePy not installed. Install with: pip install moviepy")
        sys.exit(1)

# PIL for rendering
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# UI Template renderer for realistic phone mockups
from ui_template_renderer import render_ui_template

# Optional: TTS for transcript
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

# Stable Audio for local SFX generation (lazy-loaded to avoid heavy imports)
STABLE_AUDIO_AVAILABLE = True  # We'll check actual availability when loading

# Pexels API
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def _configure_stdio() -> None:
    """Prefer UTF-8 console output so status logs do not fail on Windows code pages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def _safe_exception_text(exc: Exception, limit: int | None = None) -> str:
    text = str(exc).encode("ascii", "backslashreplace").decode("ascii")
    if limit is not None:
        return text[:limit]
    return text

# Optional: Stable Diffusion for image generation (imported on-demand)
SD_AVAILABLE = False

# Optional: ASID audiovisual captioning (lazy-loaded, skip for now due to import issues)
try:
    from asid_captioner import ASIDCaptioner
    ASID_AVAILABLE = True
    asid_model = None  # Don't instantiate at module load time
except ImportError:
    ASID_AVAILABLE = False
    asid_model = None



class VideoGenerator:
    """Generates video from training example JSON."""
    
    def __init__(self, width: int = 1080, height: int = 1920, fps: int = 30, use_pexels: bool = True, use_elevenlabs: bool = True, trim_silence: bool = False):
        self.width = width
        self.height = height
        self.fps = fps
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_gen_"))
        self.sd_pipe = None
        self.use_pexels = use_pexels
        self.use_elevenlabs = use_elevenlabs
        self.trim_silence = trim_silence
        self.pexels_api_key = os.getenv("PEXELS_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.whisper_model = None
        self.whisper_disabled = False
        # Per-segment speaker time spans in generated narration audio:
        # (start_sec, end_sec, speaker_id)
        self.segment_speaker_spans: Dict[int, List[Tuple[float, float, int]]] = {}
        
        # Lazy-load Stable Audio model
        self.stable_audio_pipe = None
        self.cog_pipe = None
        self.cog_model_id = "zai-org/CogVideoX-2b"
        self.cog_scene_char_threshold = 888
        self.cog_num_inference_steps = 50
        self.cog_num_frames = 49
        self.cog_guidance_scale = 6.0
        self.cog_output_fps = 8
        
    def __del__(self):
        """Cleanup temp directory."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _split_transcript_by_speaker(self, transcript: str) -> List[Tuple[int, str]]:
        """Split transcript into (speaker_id, text) chunks using [SPEAKER_X] tags."""
        if not transcript:
            return []

        tag_pattern = re.compile(r"\[\s*SPEAKER[_\s]*([0-9]+)\s*\]", flags=re.IGNORECASE)
        matches = list(tag_pattern.finditer(transcript))

        if not matches:
            cleaned = transcript.strip()
            return [(1, cleaned)] if cleaned else []

        chunks: List[Tuple[int, str]] = []
        for i, match in enumerate(matches):
            try:
                speaker_id = int(match.group(1))
            except Exception:
                speaker_id = 1
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
            text = transcript[start:end].strip()
            if text:
                chunks.append((speaker_id, text))

        return chunks

    def _get_elevenlabs_voice_for_speaker(self, speaker_id: int) -> str:
        """Map speaker IDs to ElevenLabs voices."""
        voice_map = {
            1: "TX3LPaxmHKxFdv7VOQHJ",        # Liam
            2: "FGY2WhTYpPnrIDTdsKH5",        # Laura
            3: "JBFqnCBsd6RMkjVDRZzb",        # George
        }
        return voice_map.get(speaker_id, voice_map[1])

    def _get_edge_voice_for_speaker(self, speaker_id: int) -> str:
        """Map speaker IDs to distinct Edge voices for fallback path."""
        voice_map = {
            1: "en-US-AriaNeural",
            2: "en-US-GuyNeural",
            3: "en-US-JennyNeural",
        }
        return voice_map.get(speaker_id, voice_map[1])

    def _get_caption_color_for_speaker(self, speaker_id: int) -> str:
        """Map speaker IDs to caption colors."""
        color_map = {
            1: "white",
            2: "#FFD54A",  # warm yellow
            3: "#63E6FF",  # cyan
            4: "#FF8AD8",  # pink
        }
        return color_map.get(speaker_id, "white")

    def _load_whisper_model(self):
        """Lazy-load Whisper model used for word-level timestamps."""
        if self.whisper_disabled:
            return None
        if self.whisper_model is not None:
            return self.whisper_model

        try:
            import whisper
            model_name = os.getenv("CAPTION_WHISPER_MODEL", "base")
            print(f"    [CAPTION] Loading Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name)
            return self.whisper_model
        except Exception as e:
            print(f"    [INFO] Whisper unavailable for timed captions: {e}")
            self.whisper_disabled = True
            return None

    def _extract_whisper_word_timings(self, audio_path: str) -> List[Dict]:
        """Extract word timings from audio using Whisper."""
        model = self._load_whisper_model()
        if model is None:
            return []

        try:
            result = model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en",
            )
            timings: List[Dict] = []
            for seg in result.get("segments", []) or []:
                for w in seg.get("words", []) or []:
                    word = (w.get("word") or "").strip()
                    start = float(w.get("start", 0.0) or 0.0)
                    end = float(w.get("end", start) or start)
                    if word and end > start:
                        timings.append({"word": word, "start": start, "duration": end - start})
            return timings
        except Exception as e:
            print(f"    [INFO] Whisper word timing extraction failed: {e}")
            return []

    def _speaker_for_time(self, segment_idx: int, t: float) -> int:
        """Resolve speaker id for a timestamp based on generated chunk spans."""
        spans = self.segment_speaker_spans.get(segment_idx, [])
        if not spans:
            return 1

        for start, end, speaker_id in spans:
            if start <= t < end:
                return speaker_id

        # If timing lands exactly at/after the last boundary, use last speaker.
        return spans[-1][2]

    def _build_word_caption_clips(self, audio_path: str, segment_idx: int, segment_duration: float):
        """Create one-word-at-a-time caption clips using Whisper timings."""
        timings = self._extract_whisper_word_timings(audio_path)
        if not timings:
            return []

        caption_clips = []
        for item in timings:
            word = str(item["word"]).strip()
            start = max(0.0, float(item["start"]))
            duration = max(0.05, float(item["duration"]))

            if not word or start >= segment_duration:
                continue

            if start + duration > segment_duration:
                duration = max(0.05, segment_duration - start)

            speaker_id = self._speaker_for_time(segment_idx, start + (duration * 0.5))
            color = self._get_caption_color_for_speaker(speaker_id)

            word_clip = self._create_caption_clip(
                text=word,
                color=color,
                start=start,
                duration=duration,
            )

            if word_clip is None:
                continue

            caption_clips.append(word_clip)

        return caption_clips

    def _load_caption_font(self, font_size: int):
        """Load a reasonably bold caption font with safe fallbacks."""
        candidates = [
            "arialbd.ttf",
            "Arial Bold.ttf",
            "arial.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, font_size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _caption_color_to_rgb(self, color: str) -> Tuple[int, int, int]:
        """Convert a CSS-like color string into an RGB tuple."""
        if not color:
            return (255, 255, 255)
        if color.startswith("#"):
            value = color.lstrip("#")
            if len(value) == 6:
                try:
                    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
                except Exception:
                    return (255, 255, 255)

        named = {
            "white": (255, 255, 255),
            "yellow": (255, 215, 74),
            "cyan": (99, 230, 255),
            "pink": (255, 138, 216),
        }
        return named.get(color.lower(), (255, 255, 255))

    def _create_caption_clip(self, text: str, color: str, start: float, duration: float):
        """Render caption text with PIL to avoid MoviePy bottom-edge clipping."""
        if not text:
            return None

        font_size = 78
        stroke_width = 5
        horizontal_padding = 32
        vertical_padding = 28
        font = self._load_caption_font(font_size)
        fill_rgb = self._caption_color_to_rgb(color)

        probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=stroke_width,
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        canvas_width = min(self.width - 120, text_width + horizontal_padding * 2)
        canvas_height = text_height + vertical_padding * 2

        image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text(
            (canvas_width // 2, vertical_padding - bbox[1]),
            text,
            font=font,
            fill=(*fill_rgb, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
            anchor="ma",
        )

        word_clip = ImageClip(np.array(image))
        if hasattr(word_clip, "with_start"):
            word_clip = word_clip.with_start(start)
        else:
            word_clip = word_clip.set_start(start)

        if hasattr(word_clip, "with_duration"):
            word_clip = word_clip.with_duration(duration)
        else:
            word_clip = word_clip.set_duration(duration)

        y_pos = max(0, self.height - canvas_height - max(180, int(self.height * 0.10)))
        if hasattr(word_clip, "with_position"):
            word_clip = word_clip.with_position(("center", y_pos))
        else:
            word_clip = word_clip.set_position(("center", y_pos))
        return word_clip

    def _should_use_cog_video(self, description: str) -> bool:
        """Use local CogVideoX generation only for long, detailed scenes."""
        return bool(description and len(description.strip()) > self.cog_scene_char_threshold)

    def _load_cog_video_model(self) -> bool:
        """Lazy-load CogVideoX-2b Diffusers pipeline with model-card optimizations."""
        if self.cog_pipe is not None:
            return True

        try:
            import torch
            from diffusers import CogVideoXPipeline
        except Exception as e:
            print(f"    [INFO] CogVideoX dependencies unavailable: {e}")
            return False

        if not torch.cuda.is_available():
            print("    [INFO] CogVideoX requires CUDA for practical local inference; skipping")
            return False

        try:
            print(f"    [LOAD] Loading CogVideoX pipeline: {self.cog_model_id}")
            self.cog_pipe = CogVideoXPipeline.from_pretrained(
                self.cog_model_id,
                torch_dtype=torch.float16,
            )
            self.cog_pipe.enable_model_cpu_offload()
            if hasattr(self.cog_pipe, "enable_sequential_cpu_offload"):
                self.cog_pipe.enable_sequential_cpu_offload()
            if hasattr(self.cog_pipe, "vae") and hasattr(self.cog_pipe.vae, "enable_slicing"):
                self.cog_pipe.vae.enable_slicing()
            if hasattr(self.cog_pipe, "vae") and hasattr(self.cog_pipe.vae, "enable_tiling"):
                self.cog_pipe.vae.enable_tiling()
            print("    [OK] CogVideoX ready")
            return True
        except Exception as e:
            print(f"    [INFO] Failed to load CogVideoX: {e}")
            self.cog_pipe = None
            return False

    def generate_cog_video_from_description(self, description: str, segment_idx: int, duration: float) -> Optional[str]:
        """Generate local video via CogVideoX-2b Diffusers pipeline."""
        if not self._should_use_cog_video(description):
            return None
        if not self._load_cog_video_model():
            return None

        try:
            import torch
            from diffusers.utils import export_to_video

            print(
                f"    [T2V] Running CogVideoX-2b for detailed scene "
                f"({len(description)} chars, target ~{duration:.1f}s)..."
            )
            generator = torch.Generator(device="cuda").manual_seed(segment_idx + 1)
            frames = self.cog_pipe(
                prompt=description,
                num_videos_per_prompt=1,
                num_inference_steps=self.cog_num_inference_steps,
                num_frames=self.cog_num_frames,
                guidance_scale=self.cog_guidance_scale,
                generator=generator,
            ).frames[0]

            video_path = self.temp_dir / f"cog_{segment_idx:03d}.mp4"
            export_to_video(frames, str(video_path), fps=self.cog_output_fps)
            print(f"    [OK] CogVideoX clip generated: {video_path}")
            return str(video_path)
        except Exception as e:
            print(f"    [INFO] CogVideoX generation failed: {e}")
            return None

    def _fit_video_to_canvas(self, video_clip: VideoFileClip, duration: float) -> VideoFileClip:
        """Crop/resize any source clip to the target vertical canvas and duration."""
        target_ratio = self.width / self.height
        video_ratio = video_clip.w / video_clip.h

        if video_ratio > target_ratio:
            new_width = int(video_clip.h * target_ratio)
            x_center = video_clip.w / 2
            video_clip = video_clip.cropped(
                x1=int(x_center - new_width / 2),
                width=new_width,
            )
        else:
            new_height = int(video_clip.w / target_ratio)
            y_center = video_clip.h / 2
            video_clip = video_clip.cropped(
                y1=int(y_center - new_height / 2),
                height=new_height,
            )

        video_clip = video_clip.resized((self.width, self.height))
        if video_clip.duration < duration:
            repeats = int(duration / video_clip.duration) + 1
            video_clip = concatenate_videoclips([video_clip] * repeats)
        return video_clip.subclipped(0, min(duration, video_clip.duration))

    def _normalize_segment_timings_for_video(
        self,
        segments: List[Dict],
        total_duration: float,
    ) -> List[Dict]:
        """Enforce sequential timing, bounded clip durations, and full duration coverage."""
        min_seg = 0.2
        max_seg = 120.0

        norm: List[Dict] = []
        for seg in segments:
            dur_raw = seg.get("duration")
            if dur_raw is None:
                start_raw = seg.get("start_time")
                end_raw = seg.get("end_time")
                if isinstance(start_raw, (int, float)) and isinstance(end_raw, (int, float)):
                    dur_raw = float(end_raw) - float(start_raw)
            try:
                dur = float(dur_raw)
            except Exception:
                dur = 1.0
            if dur <= 0:
                dur = 1.0
            dur = max(min_seg, min(max_seg, dur))

            seg_copy = dict(seg)
            seg_copy["duration"] = dur
            norm.append(seg_copy)

        if not norm and total_duration > 0:
            norm = [{
                "start_time": 0.0,
                "end_time": min(max_seg, total_duration),
                "duration": max(min_seg, min(max_seg, total_duration)),
                "scene_description": "",
                "ui_elements_html": "",
                "transcript": "",
            }]

        # Trim overflow while respecting minimum duration.
        current_total = sum(float(s.get("duration", 0.0)) for s in norm)
        overflow = max(0.0, current_total - total_duration)
        if overflow > 1e-9:
            for seg in reversed(norm):
                d = float(seg.get("duration", 0.0))
                reducible = max(0.0, d - min_seg)
                take = min(reducible, overflow)
                seg["duration"] = d - take
                overflow -= take
                if overflow <= 1e-9:
                    break

        # Extend with additional segments instead of exceeding max segment length.
        current_total = sum(float(s.get("duration", 0.0)) for s in norm)
        remain = max(0.0, total_duration - current_total)
        while remain > 1e-9:
            chunk = min(max_seg, remain)
            if chunk < min_seg:
                if norm:
                    last_d = float(norm[-1].get("duration", 0.0))
                    spare = max_seg - last_d
                    take = min(spare, remain)
                    if take > 1e-9:
                        norm[-1]["duration"] = last_d + take
                        remain -= take
                        break
                chunk = min_seg
            norm.append({
                "scene_description": "",
                "ui_elements_html": "",
                "transcript": "",
                "duration": chunk,
            })
            remain -= chunk

        # Assign strict sequential start/end times.
        t = 0.0
        for seg in norm:
            d = float(seg.get("duration", 1.0))
            d = max(min_seg, min(max_seg, d))
            seg["duration"] = d
            seg["start_time"] = round(t, 4)
            t += d
            seg["end_time"] = round(t, 4)

        # Force exact final end-time to requested total duration, when possible.
        if norm and total_duration > 0:
            delta = total_duration - float(norm[-1]["end_time"])
            if abs(delta) > 1e-6:
                new_d = float(norm[-1]["duration"]) + delta
                if min_seg <= new_d <= max_seg:
                    norm[-1]["duration"] = new_d
                    norm[-1]["end_time"] = round(total_duration, 4)

        return norm

    def _trim_silence(self, clip):
        """
        Remove silent portions longer than 3 seconds from *clip*, leaving a
        1-second silence buffer on both sides of each cut (2 seconds of silence
        remain in place of each long-silent block).

        Returns the trimmed clip, or the original clip unchanged if no eligible
        silent regions are found.
        """
        audio = clip.audio
        if audio is None:
            print("    [SILENCE] No audio track — skipping silence trim")
            return clip

        SAMPLE_RATE = 100               # Hz used for RMS analysis
        SILENCE_THRESHOLD_RATIO = 0.01  # fraction of peak RMS treated as silent
        MIN_SILENCE_DURATION = 3.0      # seconds — silence must exceed this to be cut
        BUFFER = 1.0                    # seconds of silence kept on each side of a cut

        total_dur = float(clip.duration)

        try:
            arr = audio.to_soundarray(fps=SAMPLE_RATE)
        except Exception as e:
            print(f"    [SILENCE] Could not sample audio: {e}")
            return clip

        if arr.ndim > 1:
            rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=1))
        else:
            rms = np.abs(arr.astype(np.float64))

        peak = float(np.max(rms))
        if peak < 1e-10:
            print("    [SILENCE] Audio is entirely silent — skipping trim")
            return clip

        threshold = peak * SILENCE_THRESHOLD_RATIO
        is_silent = rms < threshold

        dt = 1.0 / SAMPLE_RATE
        n = len(is_silent)
        silent_runs: List[Tuple[float, float]] = []
        in_silence = False
        run_start = 0

        for i in range(n):
            if is_silent[i] and not in_silence:
                in_silence = True
                run_start = i
            elif not is_silent[i] and in_silence:
                in_silence = False
                run_dur = (i - run_start) * dt
                if run_dur > MIN_SILENCE_DURATION:
                    silent_runs.append((run_start * dt, i * dt))
        if in_silence:
            run_dur = (n - run_start) * dt
            if run_dur > MIN_SILENCE_DURATION:
                silent_runs.append((run_start * dt, n * dt))

        if not silent_runs:
            print("    [SILENCE] No silent portions > 3 s found")
            return clip

        # Convert each silent run into a cut interval, leaving BUFFER on each side.
        cuts: List[Tuple[float, float]] = []
        for s, e in silent_runs:
            cut_start = s + BUFFER
            cut_end = min(e - BUFFER, total_dur)
            if cut_end - cut_start > 0.1:
                cuts.append((cut_start, cut_end))

        if not cuts:
            print("    [SILENCE] Silent regions found but too short after buffering")
            return clip

        print(f"    [SILENCE] Cutting {len(cuts)} silent region(s):")
        for i, (cs, ce) in enumerate(cuts):
            print(f"      cut {i + 1}: {cs:.2f}s – {ce:.2f}s ({ce - cs:.2f}s removed)")

        # Build the list of time ranges to keep.
        keep: List[Tuple[float, float]] = []
        prev = 0.0
        for cut_start, cut_end in cuts:
            if cut_start > prev + 0.05:
                keep.append((prev, cut_start))
            prev = cut_end
        if prev < total_dur - 0.05:
            keep.append((prev, total_dur))

        if not keep:
            return clip

        subclips = [
            clip.subclipped(max(0.0, s), min(total_dur, e))
            for s, e in keep
        ]
        return concatenate_videoclips(subclips)

    def load_example(self, json_path: str) -> Dict:
        """Load training example JSON."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def search_pexels_video(self, description: str, segment_idx: int, duration: float) -> Optional[str]:
        """
        Search and download video from Pexels based on scene description.
        
        Args:
            description: Scene description to search for
            segment_idx: Segment index for file naming
            duration: Desired video duration
            
        Returns:
            Path to downloaded video file, or None if failed
        """
        if not self.use_pexels or not self.pexels_api_key:
            return None
        
        # Extract search keywords from description
        # Remove common words and keep key descriptors
        stop_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'close-up', 'shot'}
        keywords = ' '.join([w for w in description.lower().split() if w not in stop_words])[:50]
        
        video_path = self.temp_dir / f"pexels_{segment_idx:03d}.mp4"
        
        try:
            print(f"  [SEARCH] Searching Pexels for: {keywords[:40]}...")
            headers = {"Authorization": self.pexels_api_key}
            url = f"https://api.pexels.com/videos/search?query={keywords}&per_page=5&orientation=portrait"
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                videos = data.get("videos", [])
                
                if videos:
                    # Get first video with HD quality
                    video = videos[0]
                    video_files = video.get("video_files", [])
                    
                    # Prefer HD portrait video
                    hd_file = next((f for f in video_files 
                                   if f.get("quality") in ["hd", "sd"] 
                                   and f.get("width", 0) < f.get("height", 0)), None)
                    
                    if not hd_file and video_files:
                        hd_file = video_files[0]
                    
                    if hd_file:
                        video_url = hd_file.get("link")
                        print(f"    Downloading video ({hd_file.get('width')}x{hd_file.get('height')})...")
                        
                        vid_response = requests.get(video_url, timeout=60)
                        if vid_response.status_code == 200:
                            with open(video_path, 'wb') as f:
                                f.write(vid_response.content)
                            print(f"    [OK] Pexels video downloaded")
                            return str(video_path)
                else:
                    print(f"    [INFO] No Pexels videos found for query")
            else:
                print(f"    [INFO] Pexels API returned status {response.status_code}")
        except Exception as e:
            print(f"    [INFO] Pexels search failed: {e}")
        
        return None
    
    def generate_image_from_description(self, description: str, segment_idx: int) -> str:
        """
        Generate or create placeholder image from scene description.
        
        Args:
            description: Scene description text
            segment_idx: Segment index for file naming
            
        Returns:
            Path to generated image file
        """
        global SD_AVAILABLE
        
        image_path = self.temp_dir / f"scene_{segment_idx:03d}.png"
        
        # Try Stable Diffusion if available (import on-demand)
        if self.sd_pipe is None:
            try:
                from diffusers import StableDiffusionPipeline
                import torch
                if torch.cuda.is_available():
                    print(f"  Loading Stable Diffusion model...")
                    self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16
                    )
                    self.sd_pipe = self.sd_pipe.to("cuda")
                    SD_AVAILABLE = True
                    print(f"  [OK] Stable Diffusion loaded")
            except Exception as e:
                print(f"  [INFO] Stable Diffusion not available, using placeholders")
                SD_AVAILABLE = False
        
        if self.sd_pipe:
            try:
                # Enhance prompt for better quality
                enhanced_prompt = f"{description}, high quality, professional photography, 4k, detailed, sharp focus, cinematic lighting"
                negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature, amateur"
                
                print(f"  Generating image: {description[:60]}...")
                image = self.sd_pipe(
                    enhanced_prompt,
                    negative_prompt=negative_prompt,
                    height=self.height,
                    width=self.width,
                    num_inference_steps=50,  # Increased from 30 for better quality
                    guidance_scale=8.5  # Increased from 7.5 for more prompt adherence
                ).images[0]
                image.save(image_path)
                print(f"  [OK] Image generated")
                return str(image_path)
            except Exception as e:
                print(f"  [WARN] SD generation failed: {_safe_exception_text(e)}, using placeholder")
        
        # Fallback: Create placeholder with text
        img = Image.new('RGB', (self.width, self.height), color=(40, 40, 60))
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 40)
            font_small = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Draw scene description wrapped
        margin = 80
        y_position = self.height // 2 - 100
        
        # Word wrap the description
        words = description.split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font_small)
            if bbox[2] - bbox[0] < self.width - 2 * margin:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw wrapped text
        for line in lines[:5]:  # Max 5 lines
            bbox = draw.textbbox((0, 0), line, font=font_small)
            text_width = bbox[2] - bbox[0]
            x = (self.width - text_width) // 2
            draw.text((x, y_position), line, fill=(200, 200, 220), font=font_small)
            y_position += 45
        
        # Draw "SEGMENT X" at top
        segment_text = f"SEGMENT {segment_idx + 1}"
        bbox = draw.textbbox((0, 0), segment_text, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(((self.width - text_width) // 2, 100), segment_text, fill=(255, 255, 255), font=font)
        
        img.save(image_path)
        return str(image_path)
    
    def render_ui_elements_html(self, html_content: str, segment_idx: int, scene_description: str = '') -> Optional[str]:
        """
        Render UI elements HTML as overlay image using content-aware templates.
        
        Args:
            html_content: HTML string with UI elements
            segment_idx: Segment index for file naming
            scene_description: Scene description for app type detection
            
        Returns:
            Path to rendered overlay PNG with transparency, or None if empty
        """
        if not html_content or not html_content.strip():
            return None
        
        overlay_path = self.temp_dir / f"overlay_{segment_idx:03d}.png"
        
        # Parse HTML to extract UI elements
        import re
        ui_elements = []
        
        # Extract div elements
        div_pattern = r'<div(?:\s+class="([^"]*)")?>(.*?)</div>'
        div_matches = re.findall(div_pattern, html_content, re.DOTALL)
        
        for classes, content in div_matches:
            content = content.strip()
            if not content:
                continue
            
            # Parse position from classes
            position = 'center'
            if 'pos-top-left' in classes:
                position = 'top-left'
            elif 'pos-top-right' in classes:
                position = 'top-right'
            elif 'pos-top' in classes:
                position = 'top'
            elif 'pos-bottom-left' in classes:
                position = 'bottom-left'
            elif 'pos-bottom-right' in classes:
                position = 'bottom-right'
            elif 'pos-bottom' in classes:
                position = 'bottom'
            
            # Parse style
            style = 'white' if 'style-white' in classes else 'default'
            
            ui_elements.append({
                'content': content,
                'position': position,
                'style': style,
                'type': 'text'
            })
        
        # Extract span elements (icons)
        # Handle both <span class="..." aria-label="..."> and <span aria-label="..." class="...">
        span_pattern = r'<span\s+([^>]*?)>(.*?)</span>'
        span_matches = re.findall(span_pattern, html_content, re.DOTALL)
        
        for attrs, label in span_matches:
            # Extract aria-label from attributes
            aria_match = re.search(r'aria-label="([^"]*)"', attrs)
            if aria_match:
                aria_label = aria_match.group(1)
                
                # Extract position from class attribute
                class_match = re.search(r'class="([^"]*)"', attrs)
                classes = class_match.group(1) if class_match else ""
                
                position = 'center'
                if 'pos-top-left' in classes:
                    position = 'top-left'
                elif 'pos-top-right' in classes:
                    position = 'top-right'
                elif 'pos-top' in classes:
                    position = 'top'
                elif 'pos-bottom-left' in classes:
                    position = 'bottom-left'
                elif 'pos-bottom-right' in classes:
                    position = 'bottom-right'
                elif 'pos-bottom' in classes:
                    position = 'bottom'
                
                ui_elements.append({
                    'content': aria_label,
                    'position': position,
                    'style': 'icon',
                    'type': 'icon'
                })
        
        if not ui_elements:
            return None
        
        # Use template renderer for realistic UI
        try:
            template_img = render_ui_template(
                ui_elements,
                scene_description=scene_description,
                width=self.width,
                height=self.height
            )
            if template_img:
                template_img = template_img.convert('RGBA')
                template_img.save(overlay_path)
                return str(overlay_path)
        except Exception as e:
            print(f"    [INFO] Template rendering failed: {e}, using simple layout")
        
        # Fallback: Create simple stacked layout with better styling
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use better fonts if available
            font_large = ImageFont.truetype("arial.ttf", 60)
            font_medium = ImageFont.truetype("arial.ttf", 45)
            font_small = ImageFont.truetype("arial.ttf", 35)
        except:
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large
        
        # Group elements by position
        from collections import defaultdict
        position_groups = defaultdict(list)
        
        for elem in ui_elements:
            position_groups[elem['position']].append(elem)
        
        # Styling configuration
        def get_style_color(style_str):
            """Get color and extra props based on style class."""
            style_map = {
                'white': {'color': (255, 255, 255, 255), 'outline': False},
                'bold': {'color': (255, 255, 255, 255), 'outline': True, 'thickness': 3},
                'yellow': {'color': (255, 255, 100, 255), 'outline': False},
                'blue': {'color': (100, 200, 255, 255), 'outline': False},
                'green': {'color': (100, 255, 200, 255), 'outline': False},
            }
            return style_map.get(style_str, style_map['white'])
        
        # Draw each position group
        for position_key, elements in position_groups.items():
            # Starting position for this group
            if position_key == 'top':
                x_start = self.width // 2
                y_start = 100
                anchor = 'mt'  # middle-top
            elif position_key == 'top-left':
                x_start = 60
                y_start = 100
                anchor = 'lt'  # left-top
            elif position_key == 'top-right':
                x_start = self.width - 60
                y_start = 100
                anchor = 'rt'  # right-top
            elif position_key == 'bottom':
                x_start = self.width // 2
                y_start = self.height - 150
                anchor = 'mb'  # middle-bottom
            elif position_key == 'bottom-left':
                x_start = 60
                y_start = self.height - 150
                anchor = 'lb'  # left-bottom
            elif position_key == 'bottom-right':
                x_start = self.width - 60
                y_start = self.height - 150
                anchor = 'rb'  # right-bottom
            else:  # center
                x_start = self.width // 2
                y_start = self.height // 2
                anchor = 'mm'
            
            y_offset = y_start
            
            # Draw each element with enhanced styling
            for elem in elements:
                # Select font based on style
                if elem['style'] in ['bold']:
                    current_font = font_large
                elif elem['type'] == 'icon':
                    current_font = font_medium
                else:
                    current_font = font_medium
                
                # Get color config
                style_config = get_style_color(elem['style'])
                text_color = style_config['color']
                
                # Clean up text - remove [Sound: ...] markers if present
                text = elem['content']
                if text.startswith('[Sound:'):
                    continue  # Skip audio description markers
                
                # Draw with outline/shadow for better visibility
                outline_range = style_config.get('thickness', 2)
                shadow_color = (0, 0, 0, 180)
                
                # Draw shadow
                for adj_x in [-2, -1, 0, 1, 2]:
                    for adj_y in [-2, -1, 0, 1, 2]:
                        if adj_x != 0 or adj_y != 0:
                            draw.text(
                                (x_start + adj_x, y_offset + adj_y),
                                text,
                                fill=shadow_color,
                                font=current_font,
                                anchor=anchor
                            )
                
                # Draw main text
                draw.text(
                    (x_start, y_offset),
                    text,
                    fill=text_color,
                    font=current_font,
                    anchor=anchor
                )
                
                y_offset += 70
        
        img.save(overlay_path)
        return str(overlay_path)
    
    def generate_audio_from_transcript(self, transcript: str, segment_idx: int) -> Optional[str]:
        """
        Generate TTS audio from transcript text.
        
        Args:
            transcript: Text to speak
            segment_idx: Segment index for file naming
            
        Returns:
            Path to generated audio file, or None if empty/TTS unavailable
        """
        if not transcript or not transcript.strip():
            return None
        
        audio_path = self.temp_dir / f"audio_{segment_idx:03d}.mp3"
        speaker_chunks = self._split_transcript_by_speaker(transcript)
        self.segment_speaker_spans[segment_idx] = []
        if not speaker_chunks:
            return None
        
        # Try ElevenLabs first if available and enabled
        if self.use_elevenlabs and ELEVENLABS_AVAILABLE and self.elevenlabs_api_key:
            try:
                print("  [TTS] Generating ElevenLabs TTS (multi-speaker)...")
                client = ElevenLabs(api_key=self.elevenlabs_api_key)

                chunk_paths: List[Path] = []
                for chunk_idx, (speaker_id, text) in enumerate(speaker_chunks):
                    voice_id = self._get_elevenlabs_voice_for_speaker(speaker_id)
                    chunk_path = self.temp_dir / f"audio_{segment_idx:03d}_{chunk_idx:03d}.mp3"
                    audio_generator = client.text_to_speech.convert(
                        text=text,
                        voice_id=voice_id,
                        model_id="eleven_v3",
                        voice_settings=VoiceSettings(
                            stability=0.5,
                            similarity_boost=0.75,
                            style=0.0,
                            use_speaker_boost=True
                        ),
                        output_format="mp3_44100_128"
                    )
                    with open(chunk_path, 'wb') as f:
                        for piece in audio_generator:
                            f.write(piece)
                    chunk_paths.append(chunk_path)

                chunk_durations: List[float] = []
                if len(chunk_paths) == 1:
                    single_clip = AudioFileClip(str(chunk_paths[0]))
                    chunk_durations = [float(single_clip.duration or 0.0)]
                    single_clip.close()
                    shutil.copyfile(chunk_paths[0], audio_path)
                else:
                    from moviepy.audio.AudioClip import concatenate_audioclips
                    clips = [AudioFileClip(str(p)) for p in chunk_paths]
                    chunk_durations = [float(c.duration or 0.0) for c in clips]
                    combined = concatenate_audioclips(clips)
                    combined.write_audiofile(str(audio_path), logger=None)
                    combined.close()
                    for c in clips:
                        c.close()

                # Build speaker timing spans for caption coloring.
                t = 0.0
                spans: List[Tuple[float, float, int]] = []
                for idx2, (speaker_id, _) in enumerate(speaker_chunks):
                    dur = chunk_durations[idx2] if idx2 < len(chunk_durations) else 0.0
                    end_t = t + max(0.0, dur)
                    spans.append((t, end_t, speaker_id))
                    t = end_t
                self.segment_speaker_spans[segment_idx] = spans

                print(f"  [OK] ElevenLabs TTS generated")
                return str(audio_path)
            except Exception as e:
                print(f"  [INFO] ElevenLabs failed ({_safe_exception_text(e, 50)}), falling back to Edge TTS")
        
        # Fallback to Edge TTS
        if not EDGE_TTS_AVAILABLE:
            print(f"  [INFO] No TTS available (install edge-tts or elevenlabs)")
            return None
        
        try:
            import asyncio

            async def generate_tts_chunks() -> List[Path]:
                out_paths: List[Path] = []
                for chunk_idx, (speaker_id, text) in enumerate(speaker_chunks):
                    chunk_path = self.temp_dir / f"audio_{segment_idx:03d}_{chunk_idx:03d}.mp3"
                    voice = self._get_edge_voice_for_speaker(speaker_id)
                    # Strip bracket annotation tags (tone labels etc.) — Edge TTS
                    # reads them literally rather than interpreting them.
                    clean_text = re.sub(r"\[[^\]]{1,120}\]", "", text).strip()
                    communicate = edge_tts.Communicate(clean_text or text, voice)
                    await communicate.save(str(chunk_path))
                    out_paths.append(chunk_path)
                return out_paths

            chunk_paths = asyncio.run(generate_tts_chunks())
            chunk_durations: List[float] = []
            if len(chunk_paths) == 1:
                single_clip = AudioFileClip(str(chunk_paths[0]))
                chunk_durations = [float(single_clip.duration or 0.0)]
                single_clip.close()
                shutil.copyfile(chunk_paths[0], audio_path)
            else:
                from moviepy.audio.AudioClip import concatenate_audioclips
                clips = [AudioFileClip(str(p)) for p in chunk_paths]
                chunk_durations = [float(c.duration or 0.0) for c in clips]
                combined = concatenate_audioclips(clips)
                combined.write_audiofile(str(audio_path), logger=None)
                combined.close()
                for c in clips:
                    c.close()

            # Build speaker timing spans for caption coloring.
            t = 0.0
            spans: List[Tuple[float, float, int]] = []
            for idx2, (speaker_id, _) in enumerate(speaker_chunks):
                dur = chunk_durations[idx2] if idx2 < len(chunk_durations) else 0.0
                end_t = t + max(0.0, dur)
                spans.append((t, end_t, speaker_id))
                t = end_t
            self.segment_speaker_spans[segment_idx] = spans

            print(f"  [OK] Generated TTS audio")
            return str(audio_path)
        except Exception as e:
            print(f"  [WARN] TTS generation failed: {_safe_exception_text(e)}")
            return None

    def _load_stable_audio_model(self):
        """
        Lazy-load the Stable Audio Open 1.0 model on first use.
        """
        if self.stable_audio_pipe is None:
            if not STABLE_AUDIO_AVAILABLE:
                print("  [WARNING] Stable Audio not available")
                return False
            
            try:
                # Import here to avoid heavy imports at module load time
                import torch
                from diffusers import StableAudioPipeline
                import sys
                
                print(f"  [LOAD] Loading Stable Audio model (first run may be slow)...", flush=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"  [LOAD] Device: {device}, torch.cuda.available: {torch.cuda.is_available()}", flush=True)
                
                # Load with float32 first (more stable), then optionally convert
                print(f"  [LOAD] Loading from pretrained...", flush=True)
                sys.stdout.flush()
                self.stable_audio_pipe = StableAudioPipeline.from_pretrained(
                    "stabilityai/stable-audio-open-1.0",
                    torch_dtype=torch.float32  # Use float32 for stability
                )
                
                print(f"  [LOAD] Moving to device: {device}...", flush=True)
                sys.stdout.flush()
                self.stable_audio_pipe = self.stable_audio_pipe.to(device)
                
                # StableAudioPipeline doesn't implement eval(); components manage their own modes.
                print(f"  [OK] Stable Audio loaded on {device}", flush=True)
                return True
            except Exception as e:
                print(f"  [ERROR] Failed to load Stable Audio model: {e}")
                self.stable_audio_pipe = None
                return False
        else:
            # Model already loaded; assume it's valid
            return True
    
    def generate_sfx_from_description(
        self,
        description: str,
        segment_idx: int,
        duration: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate a sound effect clip from an audio description using Stable Audio Open 1.0.
        This is a local, free alternative to cloud-based SFX APIs.

        Args:
            description: Text describing the sound effect
            segment_idx: Index used for naming the output file
            duration: Optional target duration (seconds). Valid range: 0.5-47 seconds.
                If None, defaults to 5 seconds.

        Returns:
            Path to generated audio file or None if it failed.
        """
        print(f"  [SFX_FUNC] enter idx={segment_idx} desc='{description[:40]}' duration={duration}")
        if not description or not description.strip():
            print("  [SFX_FUNC] no description, returning None")
            return None
        if not STABLE_AUDIO_AVAILABLE:
            print("  [SFX_FUNC] STABLE_AUDIO_AVAILABLE false")
            return None
        
        # Clamp duration to valid range for Stable Audio (0.5 - 47 seconds)
        if duration is None:
            duration = 5.0
        else:
            duration = max(0.5, min(47.0, duration))

        sfx_path = self.temp_dir / f"sfx_{segment_idx:03d}.wav"
        try:
            # Load model on first use
            if not self._load_stable_audio_model():
                print("  [SFX_FUNC] model load failed, returning None")
                return None
            
            print(f"  [SFX] Generating sound effect: '{description[:50]}...' ({duration:.1f}s)")
            
            import torch  # Lazy import
            
            # Get device from the VAE (or any component). StableAudioPipeline doesn't have unet.
            device = next(self.stable_audio_pipe.vae.parameters()).device
            generator = torch.Generator(device).manual_seed(segment_idx)
            
            # Generate audio (50 steps for faster inference, trades quality for speed)
            print(f"    [GENERATE] Running inference (50 steps)...")
            
            # Ensure no gradients during inference
            with torch.no_grad():
                audio = self.stable_audio_pipe(
                    description,
                    negative_prompt="Low quality, muffled.",
                    audio_end_in_s=duration,
                    num_inference_steps=50,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                ).audios
            
            # Save audio
            import soundfile as sf  # Lazy import
            output = audio[0].T.float().cpu().numpy()
            sf.write(str(sfx_path), output, self.stable_audio_pipe.vae.sampling_rate)
            print(f"  [OK] SFX generated")
            return str(sfx_path)
        except Exception as e:
            print(f"  [INFO] SFX generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_segment_clip(self, segment: Dict, index: int) -> VideoFileClip:
        """
        Create video clip for a single segment.
        
        Args:
            segment: Segment dict from training example
            index: Segment index
            
        Returns:
            MoviePy video clip
        """
        print(f"\n  Segment {index + 1}:")
        print(f"    Duration: {segment['duration']:.1f}s")
        print(f"    Scene: {segment['scene_description'][:60]}...")
        
        generated_video = self.generate_cog_video_from_description(
            segment['scene_description'],
            index,
            segment['duration'],
        )
        background_video_path = generated_video
        if background_video_path is None:
            background_video_path = self.search_pexels_video(
                segment['scene_description'],
                index,
                segment['duration']
            )

        if background_video_path:
            try:
                video_clip = VideoFileClip(background_video_path)
                bg_clip = self._fit_video_to_canvas(video_clip, segment['duration'])
            except Exception as e:
                print(f"    [WARNING] Failed to process source video: {e}")
                print(f"    Falling back to generated image...")
                background_video_path = None
        
        if not background_video_path:
            # Fallback to generated image
            bg_image_path = self.generate_image_from_description(
                segment['scene_description'],
                index
            )
            bg_clip = ImageClip(bg_image_path).with_duration(segment['duration'])
        
        clips = [bg_clip]
        
        # Add UI elements overlay if present
        if segment.get('ui_elements_html'):
            # Augment HTML with ASID audio description if available
            if ASID_AVAILABLE and asid_model:
                try:
                    # we currently don't have segment video path; if provided, pass it
                    audio_desc = asid_model.describe_audio(segment.get('audio_path', ''))
                    if audio_desc:
                        segment['ui_elements_html'] += f"\n<div class=\"pos-bottom style-white\">{audio_desc}</div>"
                except Exception as e:
                    print(f"    [INFO] ASID audio analysis failed: {e}")

            overlay_path = self.render_ui_elements_html(
                segment['ui_elements_html'],
                index,
                scene_description=segment.get('scene_description', '')
            )
            if overlay_path:
                overlay_clip = ImageClip(overlay_path).with_duration(segment['duration'])
                clips.append(overlay_clip)
                print(f"    [OK] Added UI overlay")
        
        # Add transcript as text overlay or audio
        audio_clip = None
        sfx_clip = None
        if segment.get('transcript') or segment.get('audio_description'):
            audio_path = None
            # Try TTS first if transcript exists
            if segment.get('transcript'):
                audio_path = self.generate_audio_from_transcript(segment['transcript'], index)
                
                if audio_path:
                    # Load audio clip
                    try:
                        audio_clip = AudioFileClip(audio_path)
                    except Exception as e:
                        print(f"    [WARN] Could not load audio: {_safe_exception_text(e)}")

            # If we have an audio description, generate a sound effect
            if segment.get('audio_description'):
                print(f"    [SFX] Description: {segment['audio_description'][:50]}...")
                sfx_path = self.generate_sfx_from_description(
                    segment['audio_description'],
                    index,
                    segment.get('duration'),
                )
                sfx_clip = None
                if sfx_path:
                    print(f"    [SFX] Generated at: {sfx_path}")
                    try:
                        sfx_clip = AudioFileClip(sfx_path)
                        # Keep generated background audio far below narration.
                        # Stable Audio outputs can be very dense/loud, so use a
                        # very conservative bed level under speech.
                        sfx_gain = 0.08 if audio_clip is not None else 0.18
                        try:
                            from moviepy.audio.fx.all import volumex
                            sfx_clip = sfx_clip.fx(volumex, sfx_gain)
                        except ImportError:
                            pass
                        print(f"    [SFX] Loaded ({sfx_clip.duration:.1f}s, gain={sfx_gain:.2f})")
                    except Exception as e:
                        print(f"    [ERROR] Failed to load SFX audio: {e}")
                        sfx_clip = None
                else:
                    print(f"    [INFO] SFX generation returned None")

            # combine clips if necessary
            if audio_clip and sfx_clip:
                from moviepy.audio.AudioClip import CompositeAudioClip
                audio_clip = CompositeAudioClip([audio_clip, sfx_clip])
            elif sfx_clip and not audio_clip:
                audio_clip = sfx_clip

            # Word-level captions from Whisper timings, color-coded per speaker.
            if audio_path and segment.get('transcript'):
                try:
                    word_caption_clips = self._build_word_caption_clips(
                        audio_path=audio_path,
                        segment_idx=index,
                        segment_duration=float(segment['duration']),
                    )
                    if word_caption_clips:
                        clips.extend(word_caption_clips)
                        print(f"    [OK] Added {len(word_caption_clips)} word-timed captions")
                    else:
                        print("    [INFO] No Whisper word timings; skipping captions for this segment")
                except Exception as e:
                    print(f"    [INFO] Timed captions failed: {e}")
        
        # Composite all layers
        final_clip = CompositeVideoClip(clips, size=(self.width, self.height))
        
        # Attach audio if generated
        if audio_clip:
            final_clip = final_clip.with_audio(audio_clip)
        
        return final_clip
    
    def generate_video(self, example_json_path: str, output_path: str):
        """
        Generate complete video from training example JSON.
        
        Args:
            example_json_path: Path to training example JSON file
            output_path: Path for output video file
        """
        print(f"\n[VIDEO] Generating video from: {example_json_path}")
        
        # Load example
        example = self.load_example(example_json_path)
        
        # Get metadata
        title = example['input'].get('title', 'Untitled')
        duration = example['input'].get('duration', 0)
        segments = example['output']['segments']

        # Enforce robust timing constraints before clip creation.
        # Rules:
        # - start_i == end_{i-1}
        # - each segment duration in [0.2, 15]
        # - final end time equals requested input duration
        try:
            duration_f = float(duration)
        except Exception:
            duration_f = 0.0
        if duration_f > 0 and segments:
            segments = self._normalize_segment_timings_for_video(segments, duration_f)
            example['output']['segments'] = segments

        blank_scene_indices = [
            idx + 1
            for idx, seg in enumerate(segments)
            if not str(seg.get('scene_description', '')).strip()
        ]
        if blank_scene_indices:
            raise ValueError(
                "Found empty scene_description in segment(s): "
                + ", ".join(str(i) for i in blank_scene_indices)
                + ". Regenerate the example so scene descriptions are populated."
            )
        
        print(f"  Title: {title}")
        print(f"  Total duration: {duration}s")
        print(f"  Segments: {len(segments)}")
        
        # Show configuration
        print(f"\n  Configuration:")
        print(f"    Local T2V: {self.cog_model_id} for scenes > {self.cog_scene_char_threshold} chars")
        if self.use_pexels and self.pexels_api_key:
            print(f"    Video Source: Pexels (with SD fallback)")
        else:
            print(f"    Video Source: Stable Diffusion only")
        
        if self.use_elevenlabs and self.elevenlabs_api_key:
            print(f"    TTS: ElevenLabs (with Edge TTS fallback)")
        elif EDGE_TTS_AVAILABLE:
            print(f"    TTS: Edge TTS")
        else:
            print(f"    TTS: None available")
        
        # Generate clips for each segment
        clips = []
        for idx, segment in enumerate(segments):
            clip = self.create_segment_clip(segment, idx)
            clips.append(clip)
        
        # Concatenate all segments
        print(f"\n  Concatenating {len(clips)} segments...")
        final_video = concatenate_videoclips(clips)

        # Optionally trim long silent portions from the assembled video.
        if self.trim_silence:
            print(f"\n  [SILENCE] Trimming silent portions > 3 s (1 s buffer each side)...")
            trimmed = self._trim_silence(final_video)
            if trimmed is not final_video:
                final_video.close()
                final_video = trimmed
            print(f"  [OK] Post-trim duration: {final_video.duration:.1f}s")

        # Write output
        print(f"\n  [RENDER] Rendering final video to {output_path}...")
        final_video.write_videofile(
            output_path,
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            bitrate='5000k'
        )
        
        # Cleanup
        for clip in clips:
            clip.close()
        final_video.close()
        
        print(f"\n  [OK] Video saved: {output_path}")
        print(f"     Duration: {final_video.duration:.1f}s")


def main():
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description='Generate video from training example JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate video from example JSON
  python generate_video_from_example.py data/training_dataset/sENByrMHfyk_example.json

  # Specify output path
  python generate_video_from_example.py example.json -o my_video.mp4

  # Custom resolution
  python generate_video_from_example.py example.json --width 720 --height 1280
        """
    )
    
    parser.add_argument(
        'json_file',
        help='Path to training example JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        default='output_video.mp4',
        help='Output video file path (default: output_video.mp4)'
    )
    parser.add_argument(
        '--form',
        choices=['short', 'long'],
        default='short',
        help='Video form: short=vertical 1080x1920 (default), long=horizontal 1920x1080'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Video width in pixels (overrides --form)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Video height in pixels (overrides --form)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    parser.add_argument(
        '--no-pexels',
        action='store_true',
        help='Disable Pexels video search, use only generated images'
    )
    parser.add_argument(
        '--no-elevenlabs',
        action='store_true',
        help='Disable ElevenLabs TTS, use only Edge TTS'
    )
    parser.add_argument(
        '--trim-silence',
        action='store_true',
        help='Cut silent portions longer than 3 seconds from the final video, '
             'leaving a 1-second silence buffer on each side of each cut'
    )

    args = parser.parse_args()

    # Resolve dimensions: --form sets defaults, --width/--height override
    _form_dims = {'short': (1080, 1920), 'long': (1920, 1080)}
    _default_w, _default_h = _form_dims[args.form]
    args.width = args.width if args.width is not None else _default_w
    args.height = args.height if args.height is not None else _default_h

    # Validate input file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    # Create generator
    generator = VideoGenerator(
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_pexels=not args.no_pexels,
        use_elevenlabs=not args.no_elevenlabs,
        trim_silence=args.trim_silence,
    )
    
    # Generate video
    try:
        generator.generate_video(str(json_path), args.output)
    except KeyboardInterrupt:
        print("\n\n[WARN] Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error generating video: {_safe_exception_text(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
