# Video Labeler

A reusable Python package for analyzing videos using Vision-Language Models (VLMs) and speech recognition.

**🆓 100% FREE & LOCAL** - No API costs! Runs entirely on your machine using SmolVLM.

## Features

- 🎬 **Video Frame Analysis**: Analyze frames using local VLM (SmolVLM) or cloud APIs (GPT-4V, Claude)
- 🆓 **Fully Local**: Zero API costs with SmolVLM - your videos never leave your machine
- 🎤 **Transcript Extraction**: Extract accurate transcripts with word-level timestamps using Whisper
- ✂️ **Automatic Segmentation**: Detect logical segments based on visual and audio cues
- 🏷️ **Content Labeling**: Generate keywords, tags, and descriptions for searchable metadata
- 🎯 **Action Detection**: Track actions and movements between frames with LLM estimation
- 📊 **Quality Scoring**: Automatically score segments for clip-worthiness
- 💾 **Serialization**: Save and load analyses as JSON
- 🔍 **Content Search**: Find segments by keywords, time range, or content

## Quick Start (Local - FREE!)

```bash
# Install dependencies
pip install torch torchvision transformers accelerate
pip install moviepy pillow numpy openai-whisper
```

```python
from video_labeler import VideoLabeler, LabelerConfig

# Use local SmolVLM (free, no API keys needed!)
config = LabelerConfig(
    vlm_provider="local",  # Fully local
    vlm_model="HuggingFaceTB/SmolVLM-Instruct",
    use_whisper=True
)

labeler = VideoLabeler(config)

# Analyze a video (first run downloads model ~2GB)
analysis = labeler.analyze_video("path/to/video.mp4")

# Print results
print(f"Summary: {analysis.summary}")
print(f"Segments: {len(analysis.segments)}")

# Find specific content
cooking_segments = analysis.find_segments_by_keyword("cooking")

# Get clip-worthy segments
clips = analysis.get_clip_worthy_segments(min_score=0.7)
```

**See [SMOLVLM_SETUP.md](SMOLVLM_SETUP.md) for complete installation guide and optimization tips!**

### Easy Installation (Recommended)

**Windows:**
```bash
# Run the installer
install_windows.bat
```

**Linux/Mac:**
```bash
# Make executable and run
chmod +x install_linux.sh
./install_linux.sh
```

The installer will:
- ✅ Install PyTorch (with CUDA if you have GPU)
- ✅ Install all dependencies
- ✅ Test your installation
- ✅ Download SmolVLM (~2GB on first run)

## Installation

### Option 1: Local (Recommended - FREE!)

```bash
# PyTorch with CUDA (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install transformers accelerate moviepy pillow numpy openai-whisper
```

### Option 2: Cloud APIs (Costs money)

```bash
# Basic installation
pip install moviepy pillow numpy

# With OpenAI or Anthropic
pip install moviepy pillow numpy openai anthropic openai-whisper
```

## Configuration

### Local VLM (Default - FREE!)

```python
from video_labeler import LabelerConfig

config = LabelerConfig(
    # Frame sampling
    frame_interval=30,          # Analyze every 30 frames (1s at 30fps)
    max_frames=None,            # Analyze all frames
    
    # VLM settings (defaults to local!)
    vlm_provider="local",                           # Local SmolVLM
    vlm_model="HuggingFaceTB/SmolVLM-Instruct",    # Free & fast
    
    # Transcription (also local!)
    use_whisper=True,
    whisper_model="base",       # tiny, base, small, medium, large
    
    # Analysis depth
    analyze_actions=True,       # Track actions between frames
    analyze_objects=True,       # Detect objects/characters
    analyze_scenes=True,        # Detect scene changes
    generate_summary=True,      # Generate overall summary
    
    # Segmentation
    auto_segment=True,
    min_segment_duration=2.0,   # Minimum 2 seconds per segment
    scene_change_threshold=0.7, # 0-1 similarity threshold
    
    # Keywords & metadata
    generate_keywords=True,
    max_keywords_per_segment=10,
    
    # Performance
    batch_size=5,               # Process 5 frames in parallel
    include_frame_images=False, # Save frame images
)

labeler = VideoLabeler(config)
```

### Cloud APIs (Optional - Costs Money)

If you prefer cloud APIs or need higher quality:

```python
# OpenAI GPT-4V
config = LabelerConfig(
    vlm_provider="openai",
    vlm_model="gpt-4o",
    api_key="your-openai-key"  # or set OPENAI_API_KEY env var
)

# Anthropic Claude Vision
config = LabelerConfig(
    vlm_provider="anthropic",
    vlm_model="claude-3-5-sonnet-20241022",
    api_key="your-anthropic-key"  # or set ANTHROPIC_API_KEY env var
)
```

### Provider Comparison

| Feature | Local (SmolVLM) | OpenAI GPT-4V | Claude Vision |
|---------|----------------|---------------|---------------|
| **Cost** | 🆓 FREE | 💰 ~$0.50/video | 💰 ~$1.00/video |
| **Speed (GPU)** | ⚡ Fast (~1s/frame) | 🐢 Slow (API latency) | 🐢 Slow (API latency) |
| **Privacy** | 🔒 100% local | ☁️ Cloud only | ☁️ Cloud only |
| **Quality** | ⭐⭐⭐⭐ Very good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent |
| **Setup** | Install PyTorch | API key | API key |
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Rate limits** | ✅ None | ⚠️ Yes | ⚠️ Yes |

**Recommendation**: Start with SmolVLM (local) for development and scale. Only use cloud APIs if you need the absolute highest quality or don't have a GPU.

## Architecture

```
VideoLabeler
├── FrameExtractor       # Extracts frames from video
├── TranscriptExtractor  # Extracts audio transcript with Whisper
├── VLMAnalyzer         # Analyzes frames with VLM
└── SegmentDetector     # Detects logical segments
```

## Data Models

### VideoAnalysis
Complete analysis of a video with all metadata and segments.

```python
analysis.video_path          # Original video path
analysis.duration            # Total duration (seconds)
analysis.fps                 # Frames per second
analysis.segments            # List of VideoSegment objects
analysis.full_transcript     # Complete transcript text
analysis.summary             # AI-generated summary
analysis.global_keywords     # Video-wide keywords
```

### VideoSegment
A logical segment of the video with unified content.

```python
segment.segment_id           # Unique ID
segment.start_time           # Start time (seconds)
segment.end_time             # End time (seconds)
segment.duration             # Duration (seconds)
segment.description          # What happens in this segment
segment.transcript           # What's said in this segment
segment.key_frames           # List of FrameAnalysis objects
segment.dominant_objects     # Most common objects
segment.actions_sequence     # Sequence of actions
segment.keywords             # Searchable keywords
segment.clip_worthy          # Boolean: good for clipping?
segment.quality_score        # 0-1 quality score
```

### FrameAnalysis
Analysis of a single video frame.

```python
frame.frame_number           # Frame number in video
frame.timestamp              # Timestamp (seconds)
frame.description            # VLM description
frame.objects                # Detected objects/characters
frame.actions                # Actions occurring
frame.scene_type             # indoor, outdoor, studio, etc.
frame.mood                   # happy, tense, dramatic, etc.
frame.text_visible           # Any visible text
```

## Use Cases

### 1. Content Discovery
```python
# Find all cooking segments in a video
cooking = analysis.find_segments_by_keyword("cooking")

# Find action-packed moments
action_segments = [
    seg for seg in analysis.segments
    if len(seg.actions_sequence) > 3
]
```

### 2. Automatic Clipping
```python
from moviepy import VideoFileClip

# Get best segments
best = analysis.get_clip_worthy_segments(min_score=0.8)

# Extract as clips
video = VideoFileClip("video.mp4")
for i, seg in enumerate(best[:5]):
    clip = video.subclipped(seg.start_time, seg.end_time)
    clip.write_videofile(f"clip_{i}.mp4")
```

### 3. Video Summarization
```python
# Get key moments
key_moments = analysis.segments[::3]  # Every 3rd segment

# Create summary video
from moviepy import concatenate_videoclips
video = VideoFileClip("video.mp4")
clips = [
    video.subclipped(seg.start_time, seg.end_time)
    for seg in key_moments
]
summary_video = concatenate_videoclips(clips)
summary_video.write_videofile("summary.mp4")
```

### 4. Content Search & Indexing
```python
# Build searchable index
for segment in analysis.segments:
    index_entry = {
        'video_id': 'abc123',
        'segment_id': segment.segment_id,
        'time_range': (segment.start_time, segment.end_time),
        'keywords': segment.keywords,
        'description': segment.description,
        'transcript': segment.transcript
    }
    # Add to search database...
```

### 5. Save & Reuse Analysis
```python
# Analyze once
analysis = labeler.analyze_video("video.mp4")
analysis.to_json("analysis.json")

# Load later (no re-analysis needed)
from video_labeler import VideoAnalysis
analysis = VideoAnalysis.from_json("analysis.json")

# Use immediately
clips = analysis.get_clip_worthy_segments()
```

## API Reference

### VideoLabeler

**Methods:**
- `analyze_video(video_path, output_dir=None)` - Analyze a video file
  - Returns: `VideoAnalysis` object
  
### VideoAnalysis

**Methods:**
- `find_segments_by_keyword(keyword)` - Find segments containing keyword
- `get_clip_worthy_segments(min_score=0.7)` - Get quality clips
- `get_segments_by_timerange(start, end)` - Get segments in time range
- `to_json(filepath)` - Save analysis to JSON
- `from_json(filepath)` - Load analysis from JSON (class method)

## Performance Tips

1. **Frame Interval**: Higher interval = faster but less detailed
   - `frame_interval=30` (1s) - Good balance
   - `frame_interval=60` (2s) - Fast mode
   - `frame_interval=15` (0.5s) - High detail

2. **Whisper Model**: Larger = more accurate but slower
   - `tiny` - Fast, less accurate
   - `base` - Good balance (recommended)
   - `medium/large` - Slow, very accurate

3. **Batch Processing**: Increase `batch_size` for faster VLM calls
   - Default: 5 frames
   - High-RAM system: 10-20 frames

4. **Caching**: Keep `cache_frames=True` for repeated analysis

## Future Enhancements

- [ ] Local VLM support (LLaVA, SmolVLM)
- [ ] GPU acceleration for frame processing
- [ ] Multi-video batch processing
- [ ] Advanced action interpolation between frames
- [ ] Scene detection using embeddings
- [ ] Emotional analysis and sentiment tracking
- [ ] Object tracking across frames
- [ ] Support for live streams

## Requirements

- Python 3.10+
- moviepy >= 2.0.0
- Pillow
- numpy
- openai (for GPT-4 Vision)
- anthropic (for Claude Vision)
- openai-whisper (for transcription)

## License

MIT License - Use freely in your projects

## Contributing

Contributions welcome! This is designed to be a reusable library.

## Example Output

```json
{
  "video_path": "cooking_video.mp4",
  "duration": 120.5,
  "segments": [
    {
      "segment_id": 0,
      "start_time": 0.0,
      "end_time": 15.3,
      "description": "Chef introduces ingredients on counter",
      "transcript": "Today we're making pasta carbonara...",
      "keywords": ["cooking", "pasta", "ingredients", "chef"],
      "clip_worthy": true,
      "quality_score": 0.85
    },
    ...
  ],
  "summary": "A cooking tutorial showing how to make pasta carbonara...",
  "global_keywords": ["cooking", "pasta", "italian", "recipe"]
}
```

## Support

For issues or questions, open an issue on the repository.
