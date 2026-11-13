"""
Configuration for Video Labeler
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class LabelerConfig:
    """Configuration for video analysis."""
    
    # Frame sampling
    frame_interval: int = 30  # Analyze every N frames (30 frames = 1 sec at 30fps)
    max_frames: Optional[int] = None  # Max frames to analyze (None = all)
    
    # VLM settings
    vlm_provider: Literal["openai", "anthropic", "local"] = "local"
    vlm_model: str = "AudioVisual-Caption/ASID-Captioner-3B"
    
    # Transcription
    use_whisper: bool = True  # Extract audio transcript
    whisper_model: str = "base"  # tiny, base, small, medium, large
    use_speaker_diarization: bool = False  # Add [SPEAKER_X] tags to transcript segments
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    
    # Analysis depth
    analyze_actions: bool = True  # Track actions/movements between frames
    analyze_objects: bool = True  # Detect objects/characters
    analyze_scenes: bool = True  # Detect scene changes
    generate_summary: bool = True  # Generate overall summary
    
    # Segmentation
    auto_segment: bool = True  # Automatically detect logical segments
    min_segment_duration: float = 2.0  # Minimum segment length in seconds
    scene_change_threshold: float = 0.7  # Similarity threshold for scene changes
    
    # Keywords & metadata
    generate_keywords: bool = True  # Generate searchable keywords per segment
    max_keywords_per_segment: int = 10
    
    # Output
    include_frame_images: bool = False  # Save analyzed frame images
    output_format: Literal["json", "dict"] = "dict"
    
    # Performance
    batch_size: int = 5  # Number of frames to analyze in parallel
    cache_frames: bool = True  # Cache extracted frames to disk
    # Skip UI overlay extraction calls to reduce VLM calls in segment mode.
    # For ASID, this removes the second call per segment and is a major speedup.
    extract_ui: bool = True
    # Optional 4-bit quantization for local SmolVLM path (not ASID path).
    smolvlm_4bit: bool = False
    # Max generation tokens for ASID responses (lower = faster, but less detail).
    asid_max_new_tokens: int = 512
    
    # Fast mode: analyze one representative frame per transcript segment
    # instead of every N frames. Much faster (1 VLM call per segment vs. 2 per frame)
    segment_based_analysis: bool = True
    max_segment_duration: float = 5.0  # Max window size (seconds) before forcing a split
    
    def __post_init__(self):
        """Validate configuration."""
        if self.frame_interval < 1:
            raise ValueError("frame_interval must be >= 1")
        if self.min_segment_duration < 0:
            raise ValueError("min_segment_duration must be >= 0")
        if not 0 <= self.scene_change_threshold <= 1:
            raise ValueError("scene_change_threshold must be between 0 and 1")
