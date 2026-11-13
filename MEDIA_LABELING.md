# Media Labeling Integration Guide

This document explains how the AI Shorts Generator integrates video and image labeling to intelligently use media from source articles.

## Overview

The media labeling system:
1. **Extracts media** from articles/posts (images and videos)
2. **Labels images** using VLM (Vision Language Models)
3. **Labels videos** using the video_labeler package (full analysis with transcription and segmentation)
4. **Passes labeled media** to the LLM during script generation
5. **LLM assigns media** to script segments based on content matching
6. **TTS timing** determines when each media appears in the final video

## Workflow

```
Article/Post
    ↓
Extract Media URLs (images, videos)
    ↓
Download Media
    ↓
Label Images with VLM ────┐
    +                      │
Label Videos with         │
video_labeler             │
    ↓                     │
Labeled Media Dataset ────┘
    ↓
Pass to LLM for Script Generation
    ↓
LLM Assigns Media to Script Segments
    ↓
Generate TTS with word timing
    ↓
Video Composer syncs media with TTS timing
    ↓
Final Video
```

## Components

### 1. MediaLabeler (`src/media_labeler.py`)

Handles all media labeling:

```python
from media_labeler import MediaLabeler

labeler = MediaLabeler(cache_dir="./cache/media_labels")

# Label all media from an article
labeled_media = await labeler.label_media_from_article(
    media_urls={
        'images': ['url1.jpg', 'url2.jpg'],
        'videos': ['video1.mp4'],
        'youtube': []
    },
    article_id="unique_article_id"
)
```

**Image Labeling:**
- Uses SmolVLM (local, free)
- Generates description of image content
- Extracts keywords
- Fast (~2-3 seconds per image)

**Video Labeling:**
- Uses video_labeler package
- Full transcript extraction (Whisper)
- Scene detection and segmentation
- Identifies clip-worthy moments
- Extracts keywords, objects, actions
- Slower (~2-5 minutes per video depending on length)

### 2. LabeledMedia Object

Each labeled media item contains:

```python
{
    "media_type": "image" or "video",
    "url": "original_url",
    "local_path": "/path/to/cached/file",
    "description": "VLM description of content",
    "keywords": ["keyword1", "keyword2", ...],
    
    # For videos only:
    "duration": 257.28,
    "clip_worthy_segments": [
        {
            "segment_id": 0,
            "start_time": 10.0,
            "end_time": 20.0,
            "description": "...",
            "clip_worthy": true,
            "quality_score": 0.8
        }
    ],
    "summary": "Overall video summary"
}
```

### 3. Script Generation with Media

The ContentCreator now accepts labeled media:

```python
from content_creator import ContentCreator

creator = ContentCreator(provider="anthropic")

script = creator.create_script(
    topic=topic_dict,
    duration=45,
    style="engaging",
    tone="informative",
    labeled_media=labeled_media  # Pass labeled media here
)
```

The LLM receives information about all available media and assigns them to script segments:

```json
{
  "segments": [
    {
      "text": "OpenAI just got disrupted by DeepSeek...",
      "type": "hook",
      "media_assignment": "MEDIA 1",  // Uses first labeled media
      "assigned_media_path": "/path/to/image1.jpg"
    },
    {
      "text": "Here's what happened...",
      "type": "content",
      "media_assignment": "MEDIA 2",  // Uses second labeled media (video)
      "clip_start": 10.5,
      "clip_end": 20.0,
      "assigned_media_path": "/path/to/video1.mp4"
    },
    {
      "text": "This changes everything...",
      "type": "content",
      "media_assignment": "GENERATE",  // No matching media, will generate
      "sd_prompt": "futuristic AI datacenter...",
      "search_query": "AI technology future"
    }
  ]
}
```

### 4. ScriptSegment with Media Assignment

Each script segment now includes:

```python
@dataclass
class ScriptSegment:
    text: str
    type: str
    media_description: str
    media_type: str
    
    # New media assignment fields:
    media_assignment: str  # "GENERATE", "MEDIA 1", "MEDIA 2", etc.
    assigned_media_path: str  # Resolved path to media file
    clip_start: float  # For video clips
    clip_end: float  # For video clips
    
    sd_prompt: str  # Used if media_assignment is "GENERATE"
    search_query: str  # Used if media_assignment is "GENERATE"
    duration_estimate: float
```

## Example Usage

### Basic Integration

```python
import asyncio
from research_agent import ResearchAgent
from media_labeler import MediaLabeler
from content_creator import ContentCreator

async def create_video_with_article_media():
    # 1. Research topic
    research = ResearchAgent()
    topics = await research.research_topics(num_topics=10)
    best_topic = topics[0]
    
    # 2. Label media from article
    labeler = MediaLabeler()
    media_urls = best_topic.metadata.get('media_urls', {})
    
    labeled_media = await labeler.label_media_from_article(
        media_urls=media_urls,
        article_id=f"article_{best_topic.title[:20]}"
    )
    
    # 3. Generate script with labeled media
    creator = ContentCreator(provider="anthropic")
    script = creator.create_script(
        topic=best_topic.to_dict(),
        duration=45,
        labeled_media=labeled_media
    )
    
    # 4. Continue with media generation and video composition...
    # The video composer will use assigned_media_path when available
```

### Checking Media Assignments

```python
# After script generation, check what media was assigned
for i, segment in enumerate(script.segments):
    print(f"Segment {i+1}: {segment.text[:50]}...")
    
    if segment.assigned_media_path:
        print(f"  ✅ Using: {segment.assigned_media_path}")
        if segment.clip_start is not None:
            print(f"  Clip: {segment.clip_start:.1f}s - {segment.clip_end:.1f}s")
    else:
        print(f"  🎨 Will generate {segment.media_type}")
        print(f"  Search: {segment.search_query}")
```

## Configuration

### Enable/Disable Media Labeling

In your config:

```python
config = {
    'media': {
        'enable_article_media': True,  # Enable media labeling from articles
        'max_images': 5,  # Limit number of images to label
        'max_videos': 3,  # Limit number of videos to label
        'cache_dir': './cache/media_labels'
    }
}
```

### Video Labeler Settings

Customize video analysis in `media_labeler.py`:

```python
self.video_labeler_config = LabelerConfig(
    frame_interval=120,  # Analyze every 4 seconds
    max_frames=None,  # Analyze entire video
    use_whisper=True,  # Extract transcript
    whisper_model="base",  # Whisper model size
    auto_segment=True,  # Detect segments
    batch_size=5
)
```

## Performance Considerations

- **Image labeling**: ~2-3 seconds per image (local VLM)
- **Video labeling**: ~2-5 minutes per video depending on length
- **Caching**: Labeled media is cached, subsequent runs are instant
- **Parallel processing**: Images and videos are labeled in parallel
- **Limits**: Default limits are 5 images and 3 videos per article

## Benefits

1. **Authentic Content**: Uses actual media from the source article
2. **Context Matching**: LLM intelligently matches media to script content
3. **Reduced Generation**: Less need to generate/search for stock media
4. **Better Quality**: Original articles often have high-quality, relevant media
5. **Clip Extraction**: For videos, extracts only the best clip-worthy segments

## Limitations

1. **Processing Time**: Video labeling takes 2-5 minutes per video
2. **Download Required**: All media must be downloaded to cache
3. **LLM Dependency**: Requires GPT-4 or Claude for intelligent assignment
4. **GPU Recommended**: Video labeling is much faster with GPU (RTX 4090)

## Troubleshooting

### Media not being assigned

Check if:
- Media URLs are valid and downloadable
- Labeled media is being passed to `create_script()`
- Keywords match between media and script content

### Video labeling is slow

- Ensure GPU is being used (check `video_labeler_config`)
- Reduce `frame_interval` for faster analysis (less detailed)
- Use smaller Whisper model: `whisper_model="tiny"`

### Out of memory

- Reduce `max_videos` in config
- Reduce `max_frames` in video labeler config
- Process videos sequentially instead of parallel

## Testing

Run the test script:

```bash
python test_media_labeling.py
```

This demonstrates the full workflow with sample media URLs.
