"""
Video segmentation - detecting logical segments in video
"""

from typing import List
import numpy as np

from .models import VideoSegment, FrameAnalysis, TranscriptSegment


class SegmentDetector:
    """Detects logical segments in video based on visual and audio cues."""
    
    def __init__(
        self,
        min_duration: float = 2.0,
        threshold: float = 0.7
    ):
        """
        Initialize segment detector.
        
        Args:
            min_duration: Minimum segment duration in seconds
            threshold: Scene change detection threshold (0-1)
        """
        self.min_duration = min_duration
        self.threshold = threshold
    
    def detect_segments(
        self,
        frame_analyses: List[FrameAnalysis],
        transcript_segments: List[TranscriptSegment],
        video_duration: float
    ) -> List[VideoSegment]:
        """
        Detect logical segments in the video.
        
        Args:
            frame_analyses: List of analyzed frames
            transcript_segments: List of transcript segments
            video_duration: Total video duration
            
        Returns:
            List of VideoSegment objects
        """
        if not frame_analyses:
            return []
        
        # Detect segment boundaries
        boundaries = self._detect_boundaries(frame_analyses, transcript_segments)
        
        # Ensure we have start and end
        if 0 not in boundaries:
            boundaries.insert(0, 0)
        if video_duration not in boundaries:
            boundaries.append(video_duration)
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            duration = end_time - start_time
            
            # Skip segments that are too short
            if duration < self.min_duration:
                continue
            
            # Create segment
            segment = VideoSegment(
                segment_id=len(segments),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                description="",  # Will be filled in later
                transcript=""    # Will be filled in later
            )
            
            segments.append(segment)
        
        # Generate descriptions for segments
        segments = self._generate_segment_descriptions(
            segments,
            frame_analyses
        )
        
        return segments
    
    def _detect_boundaries(
        self,
        frame_analyses: List[FrameAnalysis],
        transcript_segments: List[TranscriptSegment]
    ) -> List[float]:
        """Detect segment boundaries based on visual and audio changes."""
        
        boundaries = []
        
        # 1. Detect scene changes (visual)
        for i in range(1, len(frame_analyses)):
            prev_frame = frame_analyses[i-1]
            curr_frame = frame_analyses[i]
            
            # Check for scene type change
            if (prev_frame.scene_type and curr_frame.scene_type and
                prev_frame.scene_type != curr_frame.scene_type):
                boundaries.append(curr_frame.timestamp)
                continue
            
            # Check for significant object change
            # Convert objects to strings to handle any non-hashable types
            prev_objects = set(str(obj) if not isinstance(obj, str) else obj 
                             for obj in prev_frame.objects)
            curr_objects = set(str(obj) if not isinstance(obj, str) else obj 
                             for obj in curr_frame.objects)
            
            # Calculate Jaccard similarity
            if prev_objects or curr_objects:
                intersection = len(prev_objects & curr_objects)
                union = len(prev_objects | curr_objects)
                similarity = intersection / union if union > 0 else 0
                
                # If similarity is below threshold, it's a new scene
                if similarity < self.threshold:
                    boundaries.append(curr_frame.timestamp)
        
        # 2. Use transcript pauses/topic changes
        # Long pauses in speech often indicate segment boundaries
        for i in range(1, len(transcript_segments)):
            prev_seg = transcript_segments[i-1]
            curr_seg = transcript_segments[i]
            
            pause_duration = curr_seg.start_time - prev_seg.end_time
            
            # If there's a pause > 2 seconds, it's likely a boundary
            if pause_duration > 2.0:
                boundaries.append(curr_seg.start_time)
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    def _generate_segment_descriptions(
        self,
        segments: List[VideoSegment],
        frame_analyses: List[FrameAnalysis]
    ) -> List[VideoSegment]:
        """Generate descriptions for each segment."""
        
        for segment in segments:
            # Get frames in this segment
            segment_frames = [
                f for f in frame_analyses
                if segment.start_time <= f.timestamp <= segment.end_time
            ]
            
            if not segment_frames:
                segment.description = "No visual data available"
                continue
            
            # Combine frame descriptions
            # Use the middle frame's description as primary
            mid_idx = len(segment_frames) // 2
            primary_desc = segment_frames[mid_idx].description
            
            # Add context from other frames if they differ significantly
            all_descs = [f.description for f in segment_frames]
            unique_elements = self._extract_unique_elements(all_descs)
            
            if unique_elements:
                segment.description = f"{primary_desc}. {unique_elements}"
            else:
                segment.description = primary_desc
        
        return segments
    
    def _extract_unique_elements(self, descriptions: List[str]) -> str:
        """Extract unique narrative elements from multiple descriptions."""
        
        # Simple approach: look for unique key phrases
        # Could be enhanced with NLP
        
        if len(descriptions) <= 1:
            return ""
        
        # Get unique sentences
        all_sentences = []
        for desc in descriptions:
            sentences = [s.strip() for s in desc.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        # Count occurrences
        from collections import Counter
        sentence_counts = Counter(all_sentences)
        
        # Get sentences that appear in minority of descriptions
        unique = [
            sent for sent, count in sentence_counts.items()
            if count == 1 and len(sent) > 20
        ]
        
        if unique:
            return " ".join(unique[:2])  # Max 2 unique elements
        
        return ""
