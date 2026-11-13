"""
Video Composer Module
Assembles all media assets into a final short-form video with effects and captions.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import random
from dataclasses import dataclass

# Video editing - MoviePy 2.x compatible
MOVIEPY_AVAILABLE = False
CompositeVideoClip = None
ImageClip = None
AudioFileClip = None
TextClip = None
concatenate_videoclips = None

try:
    # MoviePy 2.x imports directly from moviepy (not moviepy.editor)
    from moviepy import (
        VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
        TextClip, concatenate_videoclips, CompositeAudioClip, vfx
    )
    MOVIEPY_AVAILABLE = True
    print("✅ MoviePy loaded successfully")
except ImportError as e:
    try:
        # Fallback for MoviePy 1.x
        from moviepy.editor import (
            VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
            TextClip, concatenate_videoclips, CompositeAudioClip
        )
        from moviepy.video.fx import all as vfx
        MOVIEPY_AVAILABLE = True
        print("✅ MoviePy (v1.x) loaded successfully")
    except ImportError as e2:
        MOVIEPY_AVAILABLE = False
        print(f"⚠️ MoviePy not available: {e2}")

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np


@dataclass
class VideoConfig:
    """Configuration for video composition."""
    width: int = 1080
    height: int = 1920  # 9:16 vertical
    fps: int = 30
    background_color: tuple = (0, 0, 0)
    caption_style: str = "animated"  # animated, static, karaoke
    transition_type: str = "crossfade"  # crossfade, zoom, slide
    add_background_music: bool = True
    music_volume: float = 0.2


class VideoComposer:
    """
    Composes final video from all media assets with effects, transitions, and captions.
    """
    
    def __init__(self, config: VideoConfig = None):
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy not installed. Install with: pip install moviepy")
        
        self.config = config or VideoConfig()
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)
    
    def compose_video(self, assets: Dict, output_filename: str = "video.mp4") -> str:
        """
        Compose final video from all assets.
        
        Args:
            assets: MediaAssets object with audio, images, video_clips, and captions
            output_filename: Name of output video file
            
        Returns:
            Path to generated video file
        """
        print(f"\n🎬 Composing video: {output_filename}")
        
        # Load audio
        audio_clip = AudioFileClip(assets['audio_path'])
        duration = audio_clip.duration
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Images: {len(assets['images'])}")
        video_clips = assets.get('video_clips', [])
        print(f"  Video clips: {len(video_clips)}")
        if len(video_clips) > 0:
            print(f"    First clip: {video_clips[0]}")
        print(f"  Captions: {len(assets['captions'])} words")
        
        # Create visual timeline (mix of images and video clips)
        # Sync visuals to speech timings for better pacing
        video_clip = self._create_visual_timeline(
            image_paths=assets['images'],
            video_clip_paths=video_clips,
            duration=duration,
            captions=assets['captions']
        )
        
        # Add captions
        video_with_captions = self._add_captions(
            video_clip,
            captions=assets['captions'],
            duration=duration
        )
        
        # Set audio
        final_video = video_with_captions.with_audio(audio_clip)
        
        # Add background music if enabled
        if self.config.add_background_music:
            final_video = self._add_background_music(final_video)
        
        # Export
        output_path = self.output_dir / output_filename
        print(f"  ⏳ Rendering video to {output_path}...")
        
        final_video.write_videofile(
            str(output_path),
            fps=self.config.fps,
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            bitrate='8000k'
        )
        
        # Cleanup
        audio_clip.close()
        video_clip.close()
        final_video.close()
        
        print(f"  ✅ Video saved: {output_path}")
        return str(output_path)
    
    def _create_visual_timeline(self, image_paths: List[str], video_clip_paths: List[str], 
                                duration: float, captions: List[Dict] = None) -> CompositeVideoClip:
        """
        Create the visual timeline from images and video clips with effects.
        Syncs visual changes to speech timing for better engagement.
        
        Args:
            image_paths: List of image file paths
            video_clip_paths: List of video clip file paths
            duration: Total duration in seconds
            captions: Optional list of word timings to sync visuals
            
        Returns:
            CompositeVideoClip with all visuals
        """
        print("  🎨 Creating visual timeline...")
        
        # Combine all media sources
        all_media = []
        for path in image_paths:
            all_media.append(('image', path))
        for path in video_clip_paths:
            all_media.append(('video', path))
        
        if not all_media:
            return self._create_blank_clip(duration)
        
        # Shuffle to mix images and videos throughout
        import random
        random.shuffle(all_media)
        
        # Determine scene change points based on captions
        scene_durations = self._calculate_scene_durations(duration, len(all_media), captions)
        
        clips = []
        current_time = 0
        
        for i, ((media_type, media_path), scene_duration) in enumerate(zip(all_media, scene_durations)):
            try:
                if media_type == 'image':
                    clip = self._create_image_clip(
                        media_path,
                        duration=scene_duration,
                        start_time=current_time,
                        index=i
                    )
                else:  # video
                    clip = self._create_video_clip(
                        media_path,
                        target_duration=scene_duration,
                        start_time=current_time
                    )
                
                clips.append(clip)
                current_time += scene_duration
                
            except Exception as e:
                print(f"    ⚠️ Error processing {media_type} {media_path}: {e}")
        
        if not clips:
            return self._create_blank_clip(duration)
        
        # Combine all clips
        final_clip = CompositeVideoClip(
            clips,
            size=(self.config.width, self.config.height)
        ).with_duration(duration)
        
        return final_clip
    
    def _calculate_scene_durations(self, total_duration: float, num_media: int, 
                                   captions: List[Dict] = None) -> List[float]:
        """
        Calculate how long each media clip should display based on speech timing.
        Changes visuals at natural speech breaks for better engagement.
        
        Args:
            total_duration: Total video duration
            num_media: Number of media items to distribute
            captions: Word timings to sync to
            
        Returns:
            List of durations for each media item
        """
        if not captions or num_media == 0:
            # Equal distribution fallback
            return [total_duration / num_media] * num_media
        
        # Group words into scenes (change visual every 3-5 seconds or at sentence breaks)
        target_scene_duration = total_duration / num_media
        min_scene_duration = 2.0  # At least 2 seconds per clip
        max_scene_duration = 6.0  # At most 6 seconds per clip
        
        scene_durations = []
        current_scene_start = 0
        scenes_created = 0
        
        # Determine scene break points based on natural pauses
        i = 0
        while i < len(captions) and scenes_created < num_media:
            scene_start_time = current_scene_start
            scene_end_time = scene_start_time + target_scene_duration
            
            # Look for a good break point (punctuation or natural pause)
            best_break_idx = i
            best_break_time = captions[i]['start'] + captions[i]['duration']
            
            # Scan ahead for sentence endings or good break points
            for j in range(i, min(i + 20, len(captions))):
                word = captions[j]['word'].strip()
                word_end_time = captions[j]['start'] + captions[j]['duration']
                
                # Stop at sentence endings
                if any(punct in word for punct in ['.', '!', '?']):
                    if word_end_time >= scene_start_time + min_scene_duration:
                        best_break_idx = j + 1
                        best_break_time = word_end_time
                        break
                
                # Or take a comma break if duration is good
                if ',' in word:
                    if (scene_start_time + min_scene_duration <= word_end_time <= 
                        scene_start_time + max_scene_duration):
                        best_break_idx = j + 1
                        best_break_time = word_end_time
            
            # Calculate scene duration
            scene_duration = min(max(best_break_time - scene_start_time, min_scene_duration), 
                                max_scene_duration)
            scene_durations.append(scene_duration)
            
            current_scene_start = scene_start_time + scene_duration
            i = best_break_idx
            scenes_created += 1
        
        # If we have remaining media or scenes, distribute remaining time
        if scenes_created < num_media:
            remaining_duration = total_duration - sum(scene_durations)
            remaining_scenes = num_media - scenes_created
            if remaining_scenes > 0:
                scene_durations.extend([remaining_duration / remaining_scenes] * remaining_scenes)
        
        # Adjust to match total duration exactly
        total_assigned = sum(scene_durations)
        if total_assigned > 0:
            adjustment_factor = total_duration / total_assigned
            scene_durations = [d * adjustment_factor for d in scene_durations]
        
        return scene_durations
    
    def _create_image_clip(self, img_path: str, duration: float, 
                          start_time: float, index: int) -> ImageClip:
        """
        Create an image clip with effects.
        
        Args:
            img_path: Path to image
            duration: Duration to display
            start_time: Start time in video
            index: Image index (for varying effects)
            
        Returns:
            ImageClip with effects applied
        """
        # Load and resize image
        img = Image.open(img_path)
        img = self._resize_and_crop(img, self.config.width, self.config.height)
        
        # Apply enhancement
        if index % 3 == 0:
            # Slight brightness boost
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
        elif index % 3 == 1:
            # Slight contrast boost
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.05)
        
        # Convert to numpy array for MoviePy
        img_array = np.array(img)
        
        # Create clip
        clip = ImageClip(img_array).with_duration(duration).with_start(start_time)
        
        # Apply motion effect (Ken Burns effect)
        if index % 2 == 0:
            clip = self._apply_zoom_effect(clip, zoom_ratio=1.2)
        else:
            clip = self._apply_pan_effect(clip)
        
        return clip
    
    def _create_video_clip(self, video_path: str, target_duration: float, 
                          start_time: float) -> VideoFileClip:
        """
        Create a video clip from a video file, trimmed and resized.
        
        Args:
            video_path: Path to video file
            target_duration: Target duration for this clip
            start_time: Start time in the final video
            
        Returns:
            VideoFileClip ready for composition
        """
        from moviepy import VideoFileClip
        
        # Load video
        clip = VideoFileClip(video_path)
        
        # Trim to target duration if needed
        if clip.duration > target_duration:
            # Take a clip from the middle to avoid intros/outros
            start_offset = (clip.duration - target_duration) / 2
            clip = clip.subclipped(start_offset, start_offset + target_duration)
        
        # Resize and crop to fit 9:16 aspect ratio
        clip = clip.resized(height=self.config.height)
        
        # Center crop if wider than target
        if clip.w > self.config.width:
            x_center = clip.w / 2
            x1 = int(x_center - self.config.width / 2)
            clip = clip.cropped(x1=x1, width=self.config.width)
        
        # Set start time
        clip = clip.with_start(start_time)
        
        return clip
    
    def _resize_and_crop(self, img: Image.Image, target_width: int, 
                        target_height: int) -> Image.Image:
        """Resize and crop image to fit target dimensions."""
        # Calculate aspect ratios
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # Image is wider, crop width
            new_height = target_height
            new_width = int(new_height * img_ratio)
        else:
            # Image is taller, crop height
            new_width = target_width
            new_height = int(new_width / img_ratio)
        
        # Resize
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        img = img.crop((left, top, right, bottom))
        
        return img
    
    def _apply_zoom_effect(self, clip: ImageClip, zoom_ratio: float = 1.2) -> ImageClip:
        """Apply smooth zoom effect (Ken Burns)."""
        # MoviePy 2.x: Dynamic resize effects require different approach
        # Temporarily disabled for compatibility
        return clip
    
    def _apply_pan_effect(self, clip: ImageClip) -> ImageClip:
        """Apply subtle pan effect."""
        def pan(t):
            # Pan horizontally
            progress = t / clip.duration
            offset = int((self.config.width * 0.05) * progress)  # 5% pan
            return (offset, 0)
        
        # This requires a slightly larger image, simplified version
        return clip  # Placeholder, complex pan requires different approach
    
    def _add_captions(self, video_clip: CompositeVideoClip, captions: List[Dict],
                     duration: float) -> CompositeVideoClip:
        """
        Add captions to video with word-level timing.
        
        Args:
            video_clip: Base video clip
            captions: List of caption dictionaries with word, start, duration
            duration: Total video duration
            
        Returns:
            Video with captions overlaid
        """
        print("  📝 Adding captions...")
        
        if self.config.caption_style == "animated":
            return self._add_animated_captions(video_clip, captions, duration)
        elif self.config.caption_style == "karaoke":
            return self._add_karaoke_captions(video_clip, captions, duration)
        else:
            return self._add_static_captions(video_clip, captions, duration)
    
    def _add_animated_captions(self, video_clip: CompositeVideoClip, 
                               captions: List[Dict], duration: float) -> CompositeVideoClip:
        """Add word-by-word animated captions (TikTok style)."""
        
        caption_clips = []
        
        for caption in captions:
            word = caption['word'].strip()
            if not word:
                continue
            
            try:
                # Create text clip
                txt_clip = TextClip(
                    text=word,
                    font_size=80,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,  # Reduced from 3 to prevent edge clipping
                    method='caption',
                    size=(self.config.width - 100, None),
                    margin=(0, 30)  # Vertical margin to prevent stroke bleed clipping
                )
                
                # Position at 65% from top - close to bottom without clipping
                # Testing showed margin=(0,30) prevents stroke bleed clipping
                txt_clip = txt_clip.with_position(('center', int(self.config.height * 0.65)))
                
                # Set timing
                txt_clip = txt_clip.with_start(caption['start']).with_duration(caption['duration'])
                
                # Scale animation disabled for MoviePy 2.x compatibility
                # txt_clip = txt_clip.resize(lambda t: min(1.0, 0.5 + t * 5))
                
                caption_clips.append(txt_clip)
                
            except Exception as e:
                print(f"    ⚠️ Error creating caption for '{word}': {e}")
        
        # Composite all clips with explicit size to prevent clipping
        if caption_clips:
            return CompositeVideoClip(
                [video_clip] + caption_clips,
                size=(self.config.width, self.config.height)
            )
        return video_clip
    
    def _add_karaoke_captions(self, video_clip: CompositeVideoClip, 
                             captions: List[Dict], duration: float) -> CompositeVideoClip:
        """Add karaoke-style captions (highlighting current word)."""
        
        # Group words into phrases (every 3-5 words)
        phrases = []
        current_phrase = []
        phrase_start = 0
        
        for i, caption in enumerate(captions):
            if not current_phrase:
                phrase_start = caption['start']
            
            current_phrase.append(caption)
            
            # Create phrase every 4 words or at end
            if len(current_phrase) >= 4 or i == len(captions) - 1:
                phrase_text = ' '.join([c['word'] for c in current_phrase])
                phrase_duration = (current_phrase[-1]['start'] + current_phrase[-1]['duration']) - phrase_start
                
                phrases.append({
                    'text': phrase_text,
                    'start': phrase_start,
                    'duration': phrase_duration,
                    'words': current_phrase.copy()
                })
                current_phrase = []
        
        # Create caption clips for each phrase
        caption_clips = []
        
        for phrase in phrases:
            try:
                txt_clip = TextClip(
                    text=phrase['text'],
                    font_size=70,
                    color='yellow',
                    stroke_color='black',
                    stroke_width=2,
                    method='caption',
                    size=(self.config.width - 100, None),
                    margin=(0, 30)  # Vertical margin to prevent stroke bleed clipping
                )
                
                txt_clip = txt_clip.with_position(('center', int(self.config.height * 0.65)))
                txt_clip = txt_clip.with_start(phrase['start']).with_duration(phrase['duration'])
                
                caption_clips.append(txt_clip)
                
            except Exception as e:
                print(f"    ⚠️ Error creating phrase caption: {e}")
        
        if caption_clips:
            return CompositeVideoClip(
                [video_clip] + caption_clips,
                size=(self.config.width, self.config.height)
            )
        return video_clip
    
    def _add_static_captions(self, video_clip: CompositeVideoClip, 
                            captions: List[Dict], duration: float) -> CompositeVideoClip:
        """Add simple static captions."""
        # Create full text from all captions
        full_text = ' '.join([c['word'] for c in captions])
        
        try:
            txt_clip = TextClip(
                text=full_text,
                font_size=60,
                color='white',
                font='Arial',
                method='caption',
                size=(self.config.width - 100, None),
                margin=(0, 30)  # Vertical margin to prevent stroke bleed clipping
            )
            
            txt_clip = txt_clip.with_position(('center', int(self.config.height * 0.65)))
            txt_clip = txt_clip.with_duration(duration)
            
            return CompositeVideoClip(
                [video_clip, txt_clip],
                size=(self.config.width, self.config.height)
            )
        except:
            return video_clip
    
    def _add_background_music(self, video_clip: CompositeVideoClip) -> CompositeVideoClip:
        """Add background music to video."""
        # This would integrate with a music library
        # Placeholder for now
        print("  🎵 Background music support coming soon...")
        return video_clip
    
    def _create_blank_clip(self, duration: float) -> ImageClip:
        """Create a blank clip as fallback."""
        blank = Image.new('RGB', (self.config.width, self.config.height), 
                         self.config.background_color)
        return ImageClip(np.array(blank)).with_duration(duration)


# Example usage
if __name__ == "__main__":
    # Example assets
    example_assets = {
        'audio_path': './cache/media/test/voiceover.mp3',
        'audio_duration': 15.0,
        'images': [
            './cache/media/test/images/img_001.jpg',
            './cache/media/test/images/img_002.jpg',
            './cache/media/test/images/img_003.jpg',
        ],
        'captions': [
            {'word': 'This', 'start': 0.0, 'duration': 0.3},
            {'word': 'is', 'start': 0.3, 'duration': 0.2},
            {'word': 'incredible', 'start': 0.5, 'duration': 0.5},
            {'word': 'news!', 'start': 1.0, 'duration': 0.4},
        ]
    }
    
    # Compose video
    config = VideoConfig(caption_style="animated")
    composer = VideoComposer(config)
    
    # This would normally be called with actual assets
    print("Video Composer ready. Use compose_video() with real assets.")
