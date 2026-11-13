"""
Frame extraction utilities
"""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from moviepy import VideoFileClip
from PIL import Image


class FrameExtractor:
    """Extracts frames from video files."""
    
    def extract_frames(
        self,
        video_path: str,
        interval: int = 30,
        max_frames: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to video file
            interval: Extract every N frames
            max_frames: Maximum frames to extract (None = all)
            output_dir: If provided, save frames as images
            
        Returns:
            List of (frame_number, timestamp, frame_image_array)
        """
        clip = VideoFileClip(video_path)
        fps = clip.fps
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        frames_data = []
        frame_count = 0
        
        for frame_num, frame in enumerate(clip.iter_frames()):
            # Sample at intervals
            if frame_num % interval != 0:
                continue
            
            timestamp = frame_num / fps
            
            # Save frame if output directory provided
            if output_dir:
                frame_path = output_dir / f"frame_{frame_num:06d}.jpg"
                Image.fromarray(frame).save(frame_path, quality=85)
            
            frames_data.append((frame_num, timestamp, frame))
            frame_count += 1
            
            # Check max frames limit
            if max_frames and frame_count >= max_frames:
                break
        
        clip.close()
        return frames_data
