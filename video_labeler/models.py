"""
Data models for video analysis results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class FrameAnalysis:
    """Analysis of a single video frame."""
    frame_number: int
    timestamp: float  # seconds
    description: str  # VLM description of what's in the frame
    objects: List[str] = field(default_factory=list)  # Detected objects/characters
    actions: List[str] = field(default_factory=list)  # Actions occurring
    scene_type: Optional[str] = None  # indoor, outdoor, studio, etc.
    mood: Optional[str] = None  # happy, tense, dramatic, etc.
    text_visible: Optional[str] = None  # Any text visible in frame
    ui_elements: List[Dict] = field(default_factory=list)  # UI elements with positions
    ui_html: str = ""  # Styled HTML that recreates UI appearance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "description": self.description,
            "objects": self.objects,
            "actions": self.actions,
            "scene_type": self.scene_type,
            "mood": self.mood,
            "text_visible": self.text_visible,
            "ui_elements": self.ui_elements,
            "ui_html": self.ui_html
        }


@dataclass
class VideoSegment:
    """A logical segment of the video with unified content."""
    segment_id: int
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    
    # Content
    description: str  # What happens in this segment
    transcript: str  # What's said in this segment
    
    # Visual analysis
    key_frames: List[FrameAnalysis] = field(default_factory=list)
    dominant_objects: List[str] = field(default_factory=list)  # Most common objects
    actions_sequence: List[str] = field(default_factory=list)  # Sequence of actions
    
    # Metadata
    scene_type: Optional[str] = None
    mood: Optional[str] = None
    keywords: List[str] = field(default_factory=list)  # Searchable keywords
    tags: List[str] = field(default_factory=list)  # Categories/tags
    
    # Clipping info
    clip_worthy: bool = False  # Is this segment good for a clip?
    quality_score: float = 0.0  # 0-1 score for clip quality
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "description": self.description,
            "transcript": self.transcript,
            "key_frames": [f.to_dict() for f in self.key_frames],
            "dominant_objects": self.dominant_objects,
            "actions_sequence": self.actions_sequence,
            "scene_type": self.scene_type,
            "mood": self.mood,
            "keywords": self.keywords,
            "tags": self.tags,
            "clip_worthy": self.clip_worthy,
            "quality_score": self.quality_score
        }


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio."""
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None
    speaker: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }


@dataclass
class VideoAnalysis:
    """Complete analysis of a video file."""
    video_path: str
    duration: float  # Total duration in seconds
    fps: float  # Frames per second
    resolution: Tuple[int, int]  # (width, height)
    
    # Analysis results
    segments: List[VideoSegment] = field(default_factory=list)
    full_transcript: str = ""
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    
    # Overall metadata
    summary: Optional[str] = None  # AI-generated summary
    global_keywords: List[str] = field(default_factory=list)  # Video-wide keywords
    global_tags: List[str] = field(default_factory=list)
    
    # Statistics
    total_frames_analyzed: int = 0
    analysis_duration: float = 0.0  # How long analysis took
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "fps": self.fps,
            "resolution": list(self.resolution),
            "segments": [s.to_dict() for s in self.segments],
            "full_transcript": self.full_transcript,
            "transcript_segments": [t.to_dict() for t in self.transcript_segments],
            "summary": self.summary,
            "global_keywords": self.global_keywords,
            "global_tags": self.global_tags,
            "total_frames_analyzed": self.total_frames_analyzed,
            "analysis_duration": self.analysis_duration
        }
    
    def to_json(self, filepath: str):
        """Save analysis to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'VideoAnalysis':
        """Load analysis from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct objects
        segments = [
            VideoSegment(
                segment_id=s['segment_id'],
                start_time=s['start_time'],
                end_time=s['end_time'],
                duration=s['duration'],
                description=s['description'],
                transcript=s['transcript'],
                key_frames=[
                    FrameAnalysis(**kf) for kf in s['key_frames']
                ],
                dominant_objects=s['dominant_objects'],
                actions_sequence=s['actions_sequence'],
                scene_type=s.get('scene_type'),
                mood=s.get('mood'),
                keywords=s['keywords'],
                tags=s['tags'],
                clip_worthy=s['clip_worthy'],
                quality_score=s['quality_score']
            )
            for s in data['segments']
        ]
        
        transcript_segments = [
            TranscriptSegment(**t) for t in data['transcript_segments']
        ]
        
        return cls(
            video_path=data['video_path'],
            duration=data['duration'],
            fps=data['fps'],
            resolution=tuple(data['resolution']),
            segments=segments,
            full_transcript=data['full_transcript'],
            transcript_segments=transcript_segments,
            summary=data.get('summary'),
            global_keywords=data['global_keywords'],
            global_tags=data['global_tags'],
            total_frames_analyzed=data['total_frames_analyzed'],
            analysis_duration=data['analysis_duration']
        )
    
    def find_segments_by_keyword(self, keyword: str) -> List[VideoSegment]:
        """Find segments containing a specific keyword."""
        keyword_lower = keyword.lower()
        return [
            seg for seg in self.segments
            if keyword_lower in seg.description.lower()
            or keyword_lower in seg.transcript.lower()
            or any(keyword_lower in kw.lower() for kw in seg.keywords)
        ]
    
    def get_clip_worthy_segments(self, min_score: float = 0.7) -> List[VideoSegment]:
        """Get segments marked as clip-worthy above a quality threshold."""
        return [
            seg for seg in self.segments
            if seg.clip_worthy and seg.quality_score >= min_score
        ]
    
    def get_segments_by_timerange(self, start: float, end: float) -> List[VideoSegment]:
        """Get all segments that overlap with a time range."""
        return [
            seg for seg in self.segments
            if not (seg.end_time < start or seg.start_time > end)
        ]
