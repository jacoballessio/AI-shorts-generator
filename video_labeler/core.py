"""
Core Video Labeler implementation
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile

from moviepy import VideoFileClip
import numpy as np
from PIL import Image

from .config import LabelerConfig
from .models import (
    VideoAnalysis, VideoSegment, FrameAnalysis, 
    TranscriptSegment
)
from .frame_extractor import FrameExtractor
from .transcript_extractor import TranscriptExtractor
from .vlm_analyzer import VLMAnalyzer
from .segmentation import SegmentDetector
from .ocr_extractor import OCRExtractor


class VideoLabeler:
    """
    Main class for analyzing videos with VLM and transcription.
    
    Usage:
        config = LabelerConfig(frame_interval=30, use_whisper=True)
        labeler = VideoLabeler(config)
        analysis = labeler.analyze_video("path/to/video.mp4")
        
        # Search for content
        cooking_segments = analysis.find_segments_by_keyword("cooking")
        
        # Get clip-worthy segments
        clips = analysis.get_clip_worthy_segments(min_score=0.8)
    """
    
    def __init__(self, config: Optional[LabelerConfig] = None):
        """
        Initialize the video labeler.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or LabelerConfig()
        
        # Initialize components
        self.frame_extractor = FrameExtractor()
        self.transcript_extractor = TranscriptExtractor(
            model=self.config.whisper_model if self.config.use_whisper else None,
            use_speaker_diarization=self.config.use_speaker_diarization,
            diarization_model=self.config.diarization_model,
        )
        self.vlm_analyzer = VLMAnalyzer(
            provider=self.config.vlm_provider,
            model=self.config.vlm_model,
            smolvlm_4bit=self.config.smolvlm_4bit,
            asid_max_new_tokens=self.config.asid_max_new_tokens,
        )
        self.segment_detector = SegmentDetector(
            min_duration=self.config.min_segment_duration,
            threshold=self.config.scene_change_threshold
        )
        
        # Initialize OCR extractor
        import torch
        use_gpu = torch.cuda.is_available() if self.config.vlm_provider == "local" else False
        self.ocr_extractor = OCRExtractor(use_gpu=use_gpu)
        
        print(f"[VIDEO] VideoLabeler initialized")
        print(f"   VLM: {self.config.vlm_provider}/{self.config.vlm_model}")
        print(f"   Whisper: {self.config.whisper_model if self.config.use_whisper else 'disabled'}")
        print(f"   OCR: enabled (GPU: {use_gpu})")
    
    def analyze_video(
        self, 
        video_path: str,
        output_dir: Optional[str] = None
    ) -> VideoAnalysis:
        """
        Analyze a video file completely.
        
        Args:
            video_path: Path to video file
            output_dir: Optional directory to save intermediate files
            
        Returns:
            VideoAnalysis object with complete analysis
        """
        start_time = time.time()
        
        print(f"\n[ANALYZE] Analyzing video: {video_path}")
        
        # Create output directory if needed
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="video_labeler_"))
        
        # Step 1: Extract video metadata
        print("[METADATA] Extracting video metadata...")
        video_info = self._extract_video_info(video_path)
        
        # Step 2: Extract transcript
        transcript_segments = []
        full_transcript = ""
        if self.config.use_whisper:
            print("[WHISPER] Extracting transcript with Whisper...")
            transcript_segments, full_transcript = self.transcript_extractor.extract(
                video_path,
                output_dir / "transcript.json"
            )
            print(f"   [OK] Extracted {len(transcript_segments)} transcript segments")
        
        # Step 3+4+5+6: Analyze video - either fast segment-based or full frame-by-frame
        if self.config.segment_based_analysis:
            # Fast path: 1 combined VLM call per transcript segment
            print("[VLM] Analyzing video by segments (fast mode)...")
            if self.config.vlm_provider == "local":
                import torch
                device = "CUDA" if torch.cuda.is_available() else "CPU"
                n_segs = max(1, len(transcript_segments)) if transcript_segments else int(video_info['duration'] / 5) + 1
                est_total = n_segs * (2.5 if device == "CUDA" else 12.0)
                print(f"   Device: {device}")
                print(f"   Estimated time: ~{est_total:.0f}s ({n_segs} segments, 1 call each)")
            frame_analyses, segments = self._analyze_by_segments(
                video_path, transcript_segments, video_info
            )
            print(f"   [OK] Analyzed {len(segments)} segments")
        else:
            # Legacy path: analyze every N frames individually
            print(f"[FRAMES] Extracting frames (every {self.config.frame_interval} frames)...")
            frames_data = self.frame_extractor.extract_frames(
                video_path,
                interval=self.config.frame_interval,
                max_frames=self.config.max_frames,
                output_dir=output_dir / "frames" if self.config.include_frame_images else None
            )
            print(f"   [OK] Extracted {len(frames_data)} frames")

            print("[VLM] Analyzing frames with VLM...")
            if self.config.vlm_provider == "local":
                import torch
                device = "CUDA" if torch.cuda.is_available() else "CPU"
                est_time_per_frame = 1.5 if device == "CUDA" else 8.0
                est_total = est_time_per_frame * len(frames_data)
                print(f"   Device: {device}")
                print(f"   Estimated time: ~{est_total:.0f}s ({est_time_per_frame:.1f}s per frame)")
                if device == "CPU":
                    print(f"   [WARNING] Running on CPU - this will be slow!")

            frame_analyses = self._analyze_frames_batch(
                frames_data, transcript_segments, video_info['fps']
            )
            print(f"   [OK] Analyzed {len(frame_analyses)} frames")

            print("[SEGMENTS] Detecting logical segments...")
            segments = self.segment_detector.detect_segments(
                frame_analyses, transcript_segments, video_info['duration']
            )

            print("[ENRICH] Enriching segments...")
            segments = self._enrich_segments(segments, frame_analyses, transcript_segments)
            print(f"   [OK] Created {len(segments)} segments")
        
        # Step 7: Generate summary if requested
        summary = None
        if self.config.generate_summary:
            print("[SUMMARY] Generating video summary...")
            summary = self._generate_summary(segments, full_transcript)
        
        # Step 8: Extract global keywords/tags
        global_keywords = []
        global_tags = []
        if self.config.generate_keywords:
            print("[KEYWORDS] Extracting global keywords...")
            global_keywords, global_tags = self._extract_global_metadata(segments)
        
        # Build final analysis
        analysis = VideoAnalysis(
            video_path=video_path,
            duration=video_info['duration'],
            fps=video_info['fps'],
            resolution=video_info['resolution'],
            segments=segments,
            full_transcript=full_transcript,
            transcript_segments=transcript_segments,
            summary=summary,
            global_keywords=global_keywords,
            global_tags=global_tags,
            total_frames_analyzed=len(frame_analyses),
            analysis_duration=time.time() - start_time
        )
        
        print(f"\n[OK] Analysis complete in {analysis.analysis_duration:.1f}s")
        print(f"   Segments: {len(segments)}")
        print(f"   Transcript: {len(full_transcript)} characters")
        print(f"   Keywords: {len(global_keywords)}")
        
        # Save analysis
        if output_dir:
            analysis_path = output_dir / "analysis.json"
            analysis.to_json(str(analysis_path))
            print(f"[SAVE] Saved analysis to {analysis_path}")
            
            # Also save raw frame analyses for pipeline use.
            # When segment_based_analysis is active each FrameAnalysis corresponds
            # to exactly one VideoSegment, so we include the window boundaries here
            # so that dataset_builder can reconstruct proper start/end for each clip.
            frames_path = output_dir / "frames_analysis.json"
            frames_data_dict = [
                {
                    'timestamp': fa.timestamp,
                    'description': fa.description,
                    'frame_number': fa.frame_number,
                    'objects': fa.objects,
                    'actions': fa.actions,
                    'text_visible': fa.text_visible,
                    'scene_type': fa.scene_type,
                    'ui_elements': fa.ui_elements or [],
                    'ui_html': getattr(fa, 'ui_html', ''),
                    # Window bounds (same as timestamp if not segment-based)
                    'segment_start': segments[i].start_time if i < len(segments) else fa.timestamp,
                    'segment_end': segments[i].end_time if i < len(segments) else fa.timestamp,
                }
                for i, fa in enumerate(frame_analyses)
            ]
            import json
            with open(frames_path, 'w', encoding='utf-8') as f:
                json.dump(frames_data_dict, f, indent=2)
            print(f"[SAVE] Saved frame analyses to {frames_path}")
        
        return analysis
    
    def _extract_video_info(self, video_path: str) -> dict:
        """Extract basic video metadata."""
        clip = VideoFileClip(video_path)
        info = {
            'duration': clip.duration,
            'fps': clip.fps,
            'resolution': (clip.w, clip.h)
        }
        clip.close()
        return info
    
    def _analyze_by_segments(
        self,
        video_path: str,
        transcript_segments: List[TranscriptSegment],
        video_info: dict
    ) -> Tuple[List['FrameAnalysis'], List['VideoSegment']]:
        """
        Fast segment-based analysis: sample one representative frame per segment,
        make one combined VLM call (scene + UI) per segment.
        """
        import time
        from moviepy import VideoFileClip
        
        duration = video_info['duration']

        # ── Step 1: detect visual scene cuts ─────────────────────────────
        print("   Detecting visual scene cuts...", end=" ", flush=True)
        scene_cuts = self._detect_scene_cuts(video_path)

        # ── Step 2: build transcript-based windows ────────────────────────
        if transcript_segments:
            # Group transcript segments into windows.
            # Rules (checked in order):
            #  1. If adding this segment would exceed max_segment_duration → close the current window first.
            #  2. If the gap to the next segment is tiny (< 1 s, i.e. continuous speech) → extend.
            #  3. If the current window is still below min_segment_duration → extend (don't leave stubs).
            #  4. Otherwise → close and start a new window.
            max_win = getattr(self.config, 'max_segment_duration', 5.0)
            windows = []
            buf_start = transcript_segments[0].start_time
            buf_end = transcript_segments[0].end_time
            for seg in transcript_segments[1:]:
                gap = seg.start_time - buf_end
                would_be_duration = seg.end_time - buf_start
                if would_be_duration > max_win:
                    # Close current window before it gets too long
                    windows.append((buf_start, buf_end))
                    buf_start = seg.start_time
                    buf_end = seg.end_time
                elif gap < 1.0:
                    # Continuous speech — extend
                    buf_end = seg.end_time
                elif buf_end - buf_start < self.config.min_segment_duration:
                    # Too short — extend
                    buf_end = seg.end_time
                else:
                    windows.append((buf_start, buf_end))
                    buf_start = seg.start_time
                    buf_end = seg.end_time
            windows.append((buf_start, buf_end))
            # Clamp to video bounds
            windows = [(max(0, s), min(duration, e)) for s, e in windows if e > s]
            # Ensure full coverage: first segment starts at 0, last ends at video end
            if windows:
                windows[0] = (0.0, windows[0][1])
                windows[-1] = (windows[-1][0], duration)
        else:
            window_size = 5.0
            windows = [
                (t, min(t + window_size, duration))
                for t in [i * window_size for i in range(int(duration / window_size) + 1)]
                if t < duration
            ]

        # ── Step 3: split windows at visual scene cuts ────────────────────
        if scene_cuts:
            windows = self._split_windows_at_cuts(
                windows, scene_cuts,
                min_duration=self.config.min_segment_duration
            )
            print(f"   Windows after scene-cut split: {len(windows)}")

        clip = VideoFileClip(video_path)
        frame_analyses = []
        segments = []
        use_asid = getattr(self.vlm_analyzer, 'is_asid', False)
        # For ASID we don't need the outer clip object (it extracts its own temp clips)
        if use_asid:
            clip.close()
            clip = None
        
        for i, (win_start, win_end) in enumerate(windows):
            mid_time = (win_start + win_end) / 2.0
            mid_time = max(0.0, min(mid_time, duration - 0.05))
            
            print(f"   Segment {i+1}/{len(windows)} ({win_start:.1f}s–{win_end:.1f}s)...", end=" ", flush=True)
            t0 = time.time()
            
            # Collect transcript for this window.
            # Assign each transcript segment to the window that contains its
            # START TIME (exclusive upper bound so each segment belongs to
            # exactly one window).  The last window uses <= to catch segments
            # that end right at the video boundary.
            is_last = (i == len(windows) - 1)
            context = " ".join(
                s.text for s in transcript_segments
                if s.start_time >= win_start and (
                    s.start_time < win_end if not is_last else s.start_time <= win_end
                )
            )
            
            if use_asid:
                # ASID: feed the actual video segment (audio + video)
                scene_desc, ui_html, ui_elements = self.vlm_analyzer.analyze_clip(
                    video_path,
                    win_start,
                    win_end,
                    transcript_context=context,
                    extract_ui=self.config.extract_ui,
                )
            else:
                # SmolVLM: extract a representative midpoint frame
                frame = clip.get_frame(mid_time)
                scene_desc, ui_html, ui_elements = self.vlm_analyzer.analyze_frame_combined(
                    frame, i, mid_time, transcript_context=context
                )
            
            elapsed = time.time() - t0
            print(f"[OK] ({elapsed:.1f}s)")
            
            # Build FrameAnalysis
            fa = FrameAnalysis(
                frame_number=i,
                timestamp=mid_time,
                description=scene_desc,
                ui_elements=ui_elements,
                ui_html=ui_html,
            )
            frame_analyses.append(fa)
            
            # Build VideoSegment directly (same start-time assignment as context)
            seg_transcript = " ".join(
                s.text for s in transcript_segments
                if s.start_time >= win_start and (
                    s.start_time < win_end if not is_last else s.start_time <= win_end
                )
            )
            seg = VideoSegment(
                segment_id=i,
                start_time=win_start,
                end_time=win_end,
                duration=win_end - win_start,
                description=scene_desc,
                transcript=seg_transcript,
                key_frames=[fa],
                clip_worthy=True,
                quality_score=0.7,
            )
            segments.append(seg)
        
        if clip:
            clip.close()
        return frame_analyses, segments

    def _analyze_frames_batch(
        self,
        frames_data: List[Tuple[int, float, np.ndarray]],
        transcript_segments: List[TranscriptSegment],
        fps: float
    ) -> List[FrameAnalysis]:
        """Analyze frames in batches with VLM."""
        
        frame_analyses = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(frames_data), batch_size):
            batch = frames_data[i:i+batch_size]
            print(f"   Analyzing frames {i+1}-{min(i+batch_size, len(frames_data))} of {len(frames_data)}...")
            
            # Get corresponding transcript for context
            batch_analyses = []
            for idx, (frame_num, timestamp, frame_image) in enumerate(batch, 1):
                # Show which frame we're on within the batch
                batch_progress = f"{i+idx}/{len(frames_data)}"
                print(f"      Frame {batch_progress} (timestamp {timestamp:.1f}s)...", end=" ", flush=True)
                
                # Find transcript around this timestamp
                context_transcript = self._get_transcript_context(
                    timestamp, 
                    transcript_segments
                )
                
                # Get previous frame description for continuity
                previous_desc = ""
                if frame_analyses:  # If we have any previous frames
                    previous_desc = frame_analyses[-1].description
                
                # Analyze frame
                import time
                start_time = time.time()
                
                # Run VLM analysis with previous frame context
                analysis = self.vlm_analyzer.analyze_frame(
                    frame_image,
                    frame_num,
                    timestamp,
                    transcript_context=context_transcript,
                    analyze_actions=self.config.analyze_actions,
                    analyze_objects=self.config.analyze_objects,
                    previous_frame_description=previous_desc
                )
                
                # Generate styled HTML for UI elements
                ui_html = self.vlm_analyzer.generate_ui_html_from_frame(
                    frame_image,
                    frame_num,
                    timestamp,
                    transcript_context=context_transcript,
                    previous_ui_html=getattr(frame_analyses[-1] if frame_analyses else None, 'ui_html', '')
                )
                
                # Store UI HTML in analysis object (we'll need to extend the model)
                analysis.ui_html = ui_html
                
                # OCR disabled - VLM should read text directly
                # (OCR was producing garbled results for watermarks)
                
                # Handle "Same shot" responses by referencing previous full description
                if analysis.description.lower().strip() in ['same shot', 'same shot.', 'same', 'identical']:
                    if frame_analyses:
                        analysis.description = f"[Continued: {frame_analyses[-1].description[:100]}...]"
                
                elapsed = time.time() - start_time
                print(f"[OK] ({elapsed:.1f}s)")
                
                batch_analyses.append(analysis)
            
            frame_analyses.extend(batch_analyses)
            
            # Show batch completion progress
            completed = len(frame_analyses)
            total = len(frames_data)
            progress_pct = (completed / total) * 100
            print(f"   [OK] Batch complete: {completed}/{total} frames ({progress_pct:.0f}%)")
            print()
        
        return frame_analyses

    def _detect_scene_cuts(self, video_path: str, threshold: float = 27.0) -> List[float]:
        """
        Return a sorted list of scene-cut timestamps (in seconds) using PySceneDetect's
        ContentDetector (histogram-difference based, fast, no ML required).
        """
        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            # detect() returns list of (start_timecode, end_timecode) tuples
            cuts = []
            for i, (start_tc, _) in enumerate(scene_list):
                t = start_tc.get_seconds()
                if t > 0.1:  # skip the very first frame
                    cuts.append(t)
            print(f"   [SCENE] Detected {len(cuts)} visual scene cuts")
            return sorted(cuts)
        except Exception as e:
            print(f"   [SCENE] Scene detection failed ({e}), skipping")
            return []

    def _split_windows_at_cuts(
        self,
        windows: List[tuple],
        cuts: List[float],
        min_duration: float = 1.0
    ) -> List[tuple]:
        """
        Split transcript-based windows at visual scene cut points.
        Only splits if both resulting sub-windows are >= min_duration.
        """
        result = []
        for win_start, win_end in windows:
            # Find cuts that fall inside this window (with a small margin)
            inner_cuts = [c for c in cuts if win_start + min_duration < c < win_end - min_duration]
            if not inner_cuts:
                result.append((win_start, win_end))
                continue
            # Split at each cut
            prev = win_start
            for cut in inner_cuts:
                result.append((prev, cut))
                prev = cut
            result.append((prev, win_end))
        return result

    def _get_transcript_context(
        self,
        timestamp: float,
        transcript_segments: List[TranscriptSegment],
        window: float = 5.0
    ) -> str:
        """Get transcript text around a timestamp."""
        relevant = [
            seg.text for seg in transcript_segments
            if abs(seg.start_time - timestamp) <= window
        ]
        return " ".join(relevant) if relevant else ""
    
    def _enrich_segments(
        self,
        segments: List[VideoSegment],
        frame_analyses: List[FrameAnalysis],
        transcript_segments: List[TranscriptSegment]
    ) -> List[VideoSegment]:
        """Enrich segments with detailed analysis."""
        
        for segment in segments:
            # Find frames in this segment
            segment.key_frames = [
                f for f in frame_analyses
                if segment.start_time <= f.timestamp <= segment.end_time
            ]
            
            # Find transcript in this segment
            segment_transcripts = [
                t.text for t in transcript_segments
                if segment.start_time <= t.start_time <= segment.end_time
            ]
            segment.transcript = " ".join(segment_transcripts)
            
            # Extract dominant objects
            all_objects = []
            for frame in segment.key_frames:
                all_objects.extend(frame.objects)
            # Convert to strings to handle non-hashable types (dicts, etc)
            all_objects_str = [str(obj) if not isinstance(obj, str) else obj for obj in all_objects]
            segment.dominant_objects = list(set(all_objects_str))
            
            # Extract action sequence
            all_actions = []
            for frame in segment.key_frames:
                all_actions.extend(frame.actions)
            segment.actions_sequence = all_actions
            
            # Determine scene type and mood from frames
            if segment.key_frames:
                # Use most common scene type and mood
                scene_types = [f.scene_type for f in segment.key_frames if f.scene_type]
                moods = [f.mood for f in segment.key_frames if f.mood]
                
                if scene_types:
                    segment.scene_type = max(set(scene_types), key=scene_types.count)
                if moods:
                    segment.mood = max(set(moods), key=moods.count)
            
            # Generate keywords if enabled
            if self.config.generate_keywords:
                segment.keywords = self._generate_keywords_for_segment(segment)
            
            # Score clip quality
            segment.quality_score = self._score_segment_quality(segment)
            segment.clip_worthy = segment.quality_score >= 0.6
        
        return segments
    
    def _generate_keywords_for_segment(self, segment: VideoSegment) -> List[str]:
        """Generate searchable keywords for a segment."""
        keywords = set()
        
        # Add objects
        keywords.update(segment.dominant_objects)
        
        # Add actions
        keywords.update(segment.actions_sequence)
        
        # Extract key words from description
        # Simple approach - could use LLM for better extraction
        desc_words = segment.description.lower().split()
        keywords.update([w for w in desc_words if len(w) > 4])
        
        # Limit to max keywords
        return list(keywords)[:self.config.max_keywords_per_segment]
    
    def _score_segment_quality(self, segment: VideoSegment) -> float:
        """Score segment quality for clipping (0-1)."""
        score = 0.5  # Base score
        
        # Has visual variety
        if len(segment.key_frames) > 2:
            score += 0.1
        
        # Has speech
        if segment.transcript and len(segment.transcript) > 20:
            score += 0.2
        
        # Has action
        if len(segment.actions_sequence) > 0:
            score += 0.1
        
        # Good duration (not too short, not too long)
        if 3 <= segment.duration <= 15:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_summary(
        self,
        segments: List[VideoSegment],
        full_transcript: str
    ) -> str:
        """Generate an overall summary of the video."""
        # Use VLM to summarize
        segments_desc = "\n".join([
            f"- {seg.description} ({seg.duration:.1f}s)"
            for seg in segments
        ])
        
        prompt = f"""Summarize this video in 2-3 sentences based on the following segments:

{segments_desc}

Transcript: {full_transcript[:500]}...

Provide a concise, engaging summary."""
        
        summary = self.vlm_analyzer.generate_summary(prompt)
        return summary
    
    def _extract_global_metadata(
        self,
        segments: List[VideoSegment]
    ) -> Tuple[List[str], List[str]]:
        """Extract global keywords and tags."""
        all_keywords = set()
        all_objects = set()
        all_actions = set()
        
        for seg in segments:
            all_keywords.update(seg.keywords)
            # Convert to strings to handle non-hashable types
            all_objects.update(str(obj) if not isinstance(obj, str) else obj for obj in seg.dominant_objects)
            all_actions.update(str(act) if not isinstance(act, str) else act for act in seg.actions_sequence)
        
        # Keywords are combination of segment keywords
        global_keywords = list(all_keywords)[:20]  # Top 20
        
        # Tags are high-level categories
        tags = self._categorize_content(all_objects, all_actions)
        
        return global_keywords, tags
    
    def _categorize_content(
        self,
        objects: set,
        actions: set
    ) -> List[str]:
        """Categorize content into high-level tags."""
        tags = []
        
        # Simple rule-based categorization
        # Could use LLM for better categorization
        
        people_keywords = {'person', 'people', 'man', 'woman', 'child', 'face'}
        if any(k in objects for k in people_keywords):
            tags.append('people')
        
        food_keywords = {'food', 'cooking', 'eating', 'meal', 'kitchen'}
        if any(k in objects or k in actions for k in food_keywords):
            tags.append('cooking')
        
        sport_keywords = {'sport', 'running', 'jumping', 'playing', 'ball'}
        if any(k in actions for k in sport_keywords):
            tags.append('sports')
        
        nature_keywords = {'outdoor', 'nature', 'tree', 'sky', 'water'}
        if any(k in objects for k in nature_keywords):
            tags.append('nature')
        
        tech_keywords = {'computer', 'phone', 'screen', 'technology'}
        if any(k in objects for k in tech_keywords):
            tags.append('technology')
        
        return tags
