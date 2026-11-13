"""
Video Labeler Package
A reusable library for analyzing videos with VLMs (Vision-Language Models).

Extracts transcripts, analyzes frames, detects segments, and generates searchable metadata.
"""

from .core import VideoLabeler, VideoAnalysis, VideoSegment
from .config import LabelerConfig

__version__ = "0.1.0"
__all__ = ["VideoLabeler", "VideoAnalysis", "VideoSegment", "LabelerConfig"]
