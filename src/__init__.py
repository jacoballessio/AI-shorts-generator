"""
AI Shorts Generator Package
"""

__version__ = "1.0.0"
__author__ = "AI Shorts Generator Team"
__description__ = "Autonomous AI agent for creating viral short-form videos"

# Lazy imports - only import when accessed directly
# This prevents heavy dependencies from loading when importing submodules
# from .research_agent import ResearchAgent, TopicCandidate
# from .content_creator import ContentCreator, VideoScript
# from .media_generator import MediaGenerator, MediaAssets
# from .video_composer import VideoComposer, VideoConfig
# from .main import ShortVideoOrchestrator

__all__ = [
    "ResearchAgent",
    "TopicCandidate",
    "ContentCreator",
    "VideoScript",
    "MediaGenerator",
    "MediaAssets",
    "VideoComposer",
    "VideoConfig",
    "ShortVideoOrchestrator",
]
