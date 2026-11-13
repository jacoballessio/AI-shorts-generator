# Integrating Video Labeler with AI Shorts Generator

This document explains how to use the `video_labeler` package within the AI shorts generator project.

## Use Case

The video labeler solves the problem of **matching video clips to script segments**. When you have:
- Article videos that you don't know the content of
- YouTube clips extracted from trending content
- Any source video that needs to be matched to narration

The labeler can analyze the video, understand what's happening, and help you find the right clips for each script segment.

## Integration Example

### 1. Analyze Source Videos

```python
from video_labeler import VideoLabeler, LabelerConfig

# Configure for fast analysis (good for multiple videos)
config = LabelerConfig(
    frame_interval=60,  # Every 2 seconds
    use_whisper=True,   # Get what's being said
    vlm_provider="openai",
    vlm_model="gpt-4o",
    generate_keywords=True,
    auto_segment=True
)

labeler = VideoLabeler(config)

# Analyze an article video
analysis = labeler.analyze_video(
    "article_video.mp4",
    output_dir="./cache/video_analysis"
)

# Save for later use
analysis.to_json("./cache/video_analysis/article_video_analysis.json")
```

### 2. Match Video Segments to Script Segments

```python
from video_labeler import VideoAnalysis
from src.content_creator import VideoScript

# Load script (created by ContentCreator)
script = VideoScript(...)  # Your generated script with segments

# Load video analysis
video_analysis = VideoAnalysis.from_json("./cache/video_analysis/article_video_analysis.json")

# Match each script segment to best video segment
def find_best_video_match(script_segment, video_analysis):
    """Find the best video segment for a script segment."""
    
    # Search by keywords from media_description
    keywords = script_segment.media_description.lower().split()
    
    best_match = None
    best_score = 0
    
    for video_seg in video_analysis.segments:
        # Score based on keyword overlap
        score = 0
        
        # Check description
        desc_lower = video_seg.description.lower()
        for keyword in keywords:
            if keyword in desc_lower:
                score += 2
        
        # Check objects
        for obj in video_seg.dominant_objects:
            for keyword in keywords:
                if keyword in obj.lower():
                    score += 1
        
        # Check transcript
        if video_seg.transcript:
            transcript_lower = video_seg.transcript.lower()
            for keyword in keywords:
                if keyword in transcript_lower:
                    score += 1.5
        
        # Prefer clip-worthy segments
        if video_seg.clip_worthy:
            score *= 1.2
        
        if score > best_score:
            best_score = score
            best_match = video_seg
    
    return best_match, best_score


# Match all script segments
matched_clips = []
for script_seg in script.segments:
    video_seg, confidence = find_best_video_match(script_seg, video_analysis)
    
    if video_seg and confidence > 3:  # Require minimum confidence
        matched_clips.append({
            'script_segment': script_seg,
            'video_segment': video_seg,
            'confidence': confidence,
            'start_time': video_seg.start_time,
            'end_time': video_seg.end_time
        })
        print(f"✅ Matched: {script_seg.text[:50]}...")
        print(f"   Video: {video_seg.description[:50]}... ({video_seg.start_time:.1f}s)")
        print(f"   Confidence: {confidence:.1f}")
    else:
        print(f"⚠️  No good match for: {script_seg.text[:50]}...")
        # Fallback to generated image or stock footage
```

### 3. Extract Matched Clips

```python
from moviepy import VideoFileClip

def extract_clips_from_analysis(matched_clips, source_video_path, output_dir):
    """Extract video clips based on matched segments."""
    
    video = VideoFileClip(source_video_path)
    extracted_clips = []
    
    for i, match in enumerate(matched_clips):
        video_seg = match['video_segment']
        
        # Extract the clip
        clip = video.subclipped(video_seg.start_time, video_seg.end_time)
        
        # Save clip
        clip_path = f"{output_dir}/clip_{i:03d}.mp4"
        clip.write_videofile(
            clip_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        
        extracted_clips.append({
            'path': clip_path,
            'script_segment': match['script_segment'],
            'duration': clip.duration
        })
    
    video.close()
    return extracted_clips
```

### 4. Update MediaGenerator Integration

In `src/media_generator.py`, add video labeling support:

```python
class MediaGenerator:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add video labeler
        from video_labeler import VideoLabeler, LabelerConfig
        labeler_config = LabelerConfig(
            frame_interval=60,
            use_whisper=True,
            vlm_provider=os.getenv("LLM_PROVIDER", "openai")
        )
        self.video_labeler = VideoLabeler(labeler_config)
    
    async def _match_video_clips_to_segments(
        self,
        script_segments,
        source_videos
    ):
        """Match source videos to script segments using video labeler."""
        
        # Analyze all source videos
        video_analyses = []
        for video_path in source_videos:
            # Check cache first
            cache_path = f"./cache/{Path(video_path).stem}_analysis.json"
            
            if os.path.exists(cache_path):
                analysis = VideoAnalysis.from_json(cache_path)
            else:
                analysis = self.video_labeler.analyze_video(video_path)
                analysis.to_json(cache_path)
            
            video_analyses.append(analysis)
        
        # Match each script segment
        matched_clips = []
        for script_seg in script_segments:
            best_match = None
            best_score = 0
            
            for analysis in video_analyses:
                match, score = self._find_best_match(script_seg, analysis)
                if score > best_score:
                    best_score = score
                    best_match = (match, analysis.video_path)
            
            if best_match:
                matched_clips.append({
                    'segment': best_match[0],
                    'source_video': best_match[1],
                    'script_segment': script_seg
                })
        
        return matched_clips
```

## Performance Considerations

### Caching

Always cache video analyses:

```python
import hashlib
from pathlib import Path

def get_analysis_cache_path(video_path):
    """Generate cache path for video analysis."""
    # Use file hash to detect changes
    with open(video_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    cache_dir = Path("./cache/video_analysis")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    return cache_dir / f"{Path(video_path).stem}_{file_hash}.json"

# Use it
cache_path = get_analysis_cache_path("video.mp4")
if cache_path.exists():
    analysis = VideoAnalysis.from_json(str(cache_path))
else:
    analysis = labeler.analyze_video("video.mp4")
    analysis.to_json(str(cache_path))
```

### Batch Processing

For multiple videos, process in parallel:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def analyze_videos_batch(video_paths, labeler):
    """Analyze multiple videos in parallel."""
    
    def analyze_one(path):
        return labeler.analyze_video(path)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        loop = asyncio.get_event_loop()
        analyses = await asyncio.gather(*[
            loop.run_in_executor(executor, analyze_one, path)
            for path in video_paths
        ])
    
    return analyses
```

### Fast Mode for Preview

For quick preview without full analysis:

```python
fast_config = LabelerConfig(
    frame_interval=90,  # Every 3 seconds
    use_whisper=False,  # Skip transcription
    generate_summary=False,
    max_frames=20,  # Analyze max 20 frames
    batch_size=10
)

quick_labeler = VideoLabeler(fast_config)
preview = quick_labeler.analyze_video("video.mp4")
```

## Cost Optimization

VLM API calls can be expensive. Optimize costs:

1. **Cache Everything** - Never analyze the same video twice
2. **Reduce Frame Interval** - `frame_interval=60` or `90` is usually enough
3. **Use Cheaper Models** - GPT-4o-mini for simple videos
4. **Batch Similar Videos** - Analyze related videos together
5. **Set Max Frames** - Limit analysis to first N frames for long videos

## Future Enhancements

- **Local VLM Support**: Use LLaVA or SmolVLM to eliminate API costs
- **Better Matching Algorithm**: Use embeddings for semantic matching
- **Action Interpolation**: Estimate actions between frames with LLM
- **Multi-track Composition**: Overlay multiple matched clips
- **Real-time Analysis**: Analyze videos as they download

## Example Full Workflow

```python
from src.content_creator import ContentCreator
from src.media_generator import MediaGenerator
from video_labeler import VideoLabeler, LabelerConfig

# 1. Create script with segments
creator = ContentCreator()
script = creator.create_script(topic, duration=45)

# 2. Analyze source videos
labeler = VideoLabeler(LabelerConfig(frame_interval=60))
video_analysis = labeler.analyze_video("article_video.mp4")
video_analysis.to_json("cache/analysis.json")

# 3. Match segments
matched = []
for script_seg in script.segments:
    best_match, score = find_best_video_match(script_seg, video_analysis)
    if score > 3:
        matched.append((script_seg, best_match))

# 4. Extract clips
clips = extract_matched_clips(matched, "article_video.mp4")

# 5. Compose final video
composer.compose(script, clips, captions)
```

This integration allows the AI shorts generator to intelligently use video content from any source, even when you don't know what's in the videos beforehand!
