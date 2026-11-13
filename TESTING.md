# Testing Video Labeler with example.mp4

## ✅ What You Have

You now have **3 test scripts** ready to analyze your `example.mp4` video:

### 1. Quick Test (Recommended for first try)
**File:** `quick_test.py`  
**Time:** ~1 minute  
**What it does:**
- Analyzes 10 frames from the video
- Uses local SmolVLM (free!)
- Skips transcription for speed
- Shows basic video analysis results

**Run it:**
```bash
python quick_test.py
```

### 2. Full Analysis
**File:** `test_example_video.py`  
**Time:** ~5 minutes  
**What it does:**
- Analyzes entire video (every 2 seconds)
- Uses local SmolVLM (free!)
- Extracts full transcript with Whisper
- Detects segments automatically
- Generates keywords and quality scores
- Finds clip-worthy moments
- Saves comprehensive JSON report

**Run it:**
```bash
python test_example_video.py
```

### 3. Interactive Helper (Easiest)
**File:** `test_video_labeler.bat` (Windows) or `test_video_labeler.sh` (Linux/Mac)  
**What it does:**
- Shows menu to choose quick or full test
- Automatically runs the right script

**Run it:**
```bash
# Windows
test_video_labeler.bat

# Linux/Mac
chmod +x test_video_labeler.sh
./test_video_labeler.sh
```

## 📋 What You'll Get

After running the tests, you'll see:

### Quick Test Output
```
✅ Quick test configuration:
   Max frames: 10
   Whisper: disabled (for speed)
   Segmentation: disabled (for speed)

📊 Video: example.mp4
   Duration: 257.1s
   Frames analyzed: 10
   Analysis time: 42.3s

📝 Summary:
   A person speaking to the camera...

🔑 Keywords: person, talking, indoor, face, video

🖼️  Frame Analyses:
1. Frame at 0.0s:
   A person with short brown hair speaking to the camera
   Objects: person, face, background
```

### Full Analysis Output
```
📊 VIDEO INFO
Duration: 257.1 seconds
Frames analyzed: 129
Analysis time: 184.5s

📝 SUMMARY
[AI-generated summary of the entire video]

🔑 KEYWORDS
person, talking, speaking, video, indoor, face, camera, [...]

🎤 FULL TRANSCRIPT
[Complete transcript of all speech in the video]

✂️  SEGMENTS (12 segments found)
Segment 1: 0.0s - 23.4s (23.4s)
   Description: Person introducing the topic
   Transcript: "Hey everyone, today we're going to..."
   Keywords: introduction, person, speaking
   Quality score: 0.85
   Clip-worthy: ✅ Yes

[... more segments ...]

🎬 CLIP-WORTHY SEGMENTS (5 found)
1. 34.2s - 56.8s (score: 0.92)
   Most engaging moment with strong visuals

💾 SAVED TO: example_analysis.json
```

## 📁 Output Files

After running tests, you'll have:

- `quick_test_analysis.json` - Quick test results
- `example_analysis.json` - Full analysis results
- `analysis_cache/` - Cached frame extractions (for faster re-runs)

## 🔍 Using the Results

### In Python

```python
from video_labeler.models import VideoAnalysis

# Load saved analysis
analysis = VideoAnalysis.from_json("example_analysis.json")

# Find specific content
cooking = analysis.find_segments_by_keyword("cooking")
talking = analysis.find_segments_by_keyword("talking")

# Get best moments
clips = analysis.get_clip_worthy_segments(min_score=0.7)

# Extract a specific segment
segment = analysis.segments[0]
print(f"Start: {segment.start_time}s")
print(f"End: {segment.end_time}s")
print(f"Description: {segment.description}")

# Get segments in time range
morning_clips = analysis.get_segments_by_timerange(0, 60)
```

### For Video Editing

The analysis tells you exactly where to cut:

```python
# Get the best 5 clips
top_clips = analysis.get_clip_worthy_segments(min_score=0.8)

# Extract them with MoviePy
from moviepy import VideoFileClip

video = VideoFileClip("example.mp4")

for i, clip in enumerate(top_clips[:5], 1):
    extracted = video.subclipped(clip.start_time, clip.end_time)
    extracted.write_videofile(f"clip_{i}.mp4")

video.close()
```

## 🎯 Next Steps

1. **Run quick_test.py** to verify everything works
2. **Run test_example_video.py** for complete analysis
3. **Review the JSON output** to see all detected information
4. **Integrate with your workflow** using the video_labeler package

### Integration Examples

See these files for integration ideas:
- `video_labeler/INTEGRATION.md` - Integrate with AI shorts generator
- `video_labeler/examples.py` - More usage examples
- `video_labeler/README.md` - Full API documentation

## 🚀 Performance Tips

### For Fast Testing (CPU)
```python
config = LabelerConfig(
    frame_interval=120,  # Every 4 seconds
    max_frames=10,       # Limit frames
    use_whisper=False    # Skip transcription
)
```

### For Best Quality (GPU)
```python
config = LabelerConfig(
    frame_interval=30,   # Every 1 second
    whisper_model="medium",  # Better transcription
    analyze_actions=True,
    analyze_scenes=True
)
```

### For Production (Balanced)
```python
config = LabelerConfig(
    frame_interval=60,    # Every 2 seconds
    whisper_model="base", # Good quality, fast
    batch_size=10         # Process multiple frames
)
```

## 📊 Understanding Quality Scores

The quality_score (0-1) indicates how "clip-worthy" a segment is based on:
- **Visual appeal** - Interesting composition, good lighting
- **Action density** - Things happening on screen
- **Speech quality** - Clear, engaging narration
- **Scene stability** - Minimal blurriness or camera shake

**0.9+** - Excellent, definitely use  
**0.7-0.9** - Good, likely useful  
**0.5-0.7** - Okay, review manually  
**< 0.5** - Low quality, probably skip

## ❓ Troubleshooting

### "No module named 'video_labeler'"
```bash
# Make sure you're in the right directory
cd Z:\Projects\AI-shorts-generator

# Or install the package
cd video_labeler
pip install -e .
```

### "CUDA out of memory"
```python
# Reduce batch size
config = LabelerConfig(batch_size=1)
```

### "Too slow on CPU"
```python
# Use quick test settings
config = LabelerConfig(
    frame_interval=120,
    max_frames=10,
    use_whisper=False
)
```

## 📚 Documentation

- **[README.md](README.md)** - Main project overview
- **[video_labeler/README.md](video_labeler/README.md)** - Package documentation
- **[video_labeler/SMOLVLM_SETUP.md](video_labeler/SMOLVLM_SETUP.md)** - SmolVLM setup guide
- **[video_labeler/GETTING_STARTED.md](video_labeler/GETTING_STARTED.md)** - Getting started guide

## 🎉 Ready!

Everything is set up and ready to test. Just run:

```bash
python quick_test.py
```

Happy testing! 🚀
