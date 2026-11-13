# AI-Shorts-Generator

Multi-Modal AI Video Creation Platform

## Overview

The AI-Shorts-Generator is a comprehensive platform with three distinct AI-powered approaches for creating engaging short-form videos optimized for platforms like YouTube Shorts, Instagram Reels, and TikTok. Each approach serves different use cases and workflows.

> **📖 See [APPROACHES.md](APPROACHES.md) for detailed comparison of all three approaches.**
> 
> **🚀 Quick start? See [QUICKSTART.md](QUICKSTART.md) to get running in 5 minutes.**
> 
> **📊 Project status? See [STATUS.md](STATUS.md) for current implementation status.**

---

## 🤖 Approach 1: Autonomous Content Creation (NEW!)

**Fully autonomous AI agent that creates original viral content from scratch.**

The AI autonomously researches trending topics on the internet, identifies interesting news, drama, and viral content, then automatically creates unique, engaging short-form videos with minimal human intervention.

### Key Features

- **🔍 Autonomous Research Agent**: Automatically scrapes and analyzes trending topics from multiple sources (Reddit, Twitter/X, Google Trends, news sites)
- **🤖 AI Content Creation**: Generates unique, engaging scripts using advanced language models (Claude/GPT) that add value and maintain accuracy
- **🎙️ Text-to-Speech**: High-quality AI voice generation for narration
- **📝 Auto-Captioning**: Dynamic, animated captions synchronized with audio
- **🖼️ Multi-Source Imagery**: 
  - Web scraping for relevant, trending images
  - Local Stable Diffusion generation for unique visuals
- **🎬 Intelligent Video Composition**: Automatic editing with transitions, effects, and pacing optimized for engagement
- **🖥️ Web UI**: User-friendly interface for monitoring, customization, and manual intervention
- **📊 Trend Analysis**: AI evaluates content potential before creation

### Quick Start
```bash
python ui/app.py
# Navigate to http://localhost:7860
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup.

---

## 🎬 Approach 2: Multi-Video Clip Composer (EXPERIMENTAL)

**AI-powered creative storytelling from multiple raw video clips using Vision-Language Models.**

Give the AI a collection of raw video clips (e.g., cat videos, cooking clips, sports footage) and describe what you want. The AI uses VLMs to understand the content, intelligently clips and sequences them, and creates creative compilations with AI-generated narratives.

### Key Features

- **👁️ VLM Analysis**: Uses Vision-Language Models to understand video content
- **✂️ Intelligent Clipping**: Automatically identifies and extracts interesting moments
- **📖 Creative Storytelling**: Generates narratives that connect disparate clips
- **🎭 Style Adaptation**: Can create comedy, drama, educational content from the same clips
- **🔄 Multi-Source**: Handles various video formats and qualities

### Use Cases
- "Take these 20 cat videos and make a weird horror story"
- "Turn these recipe clips into a cooking competition"
- "Make a dramatic sports montage from these clips"

**Status**: Coming soon to `experimental/multi_clip_composer.py`

---

## 📹 Approach 3: Video Summarization (ORIGINAL)

**AI-powered analysis and condensing of long-form videos into engaging shorts.**

The original concept: analyze lengthy videos, identify the most compelling moments and key highlights, and generate condensed short-form edits with seamless transitions and visual enhancements.

### Key Features

- **Video Summarization**: Analyze and summarize lengthy videos using speech recognition, natural language processing, and computer vision
- **Highlight Extraction**: Extract key moments and compile them into condensed videos
- **Customization**: Specify desired length, detail level, and tone/style
- **Automated Captioning and Tagging**: Generate captions and tags for accessibility and discoverability
- **Platform Integration**: Tailored output for various platforms

### Original Experimental Code

See `experimental/` directory:
- `generate_shorts_v1.py` - Basic video summarization
- `generate_shorts_v2.py` - Enhanced with audio merging
- `transcription.txt` - Example transcription output

---

---

## Architecture (Approach 1: Autonomous Creation)

```
┌─────────────────────────────────────────────────────────────────┐
│                           Web UI (Gradio)                       │
│  Start Generation | View Progress | Configure Settings          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Main Orchestrator                          │
│  Coordinates the entire pipeline and manages workflow           │
└─────┬────────┬────────┬────────┬────────────────────────────────┘
      │        │        │        │
      ▼        ▼        ▼        ▼
┌──────────┬─────────┬────────┬──────────────┐
│ Research │ Content │ Media  │    Video     │
│  Agent   │ Creator │ Gen    │  Composer    │
└──────────┴─────────┴────────┴──────────────┘
```

### System Components

1. **Research Agent** (`src/research_agent.py`)
   - Scrapes trending topics from Reddit, Twitter/X, Google Trends
   - Analyzes news sources for viral potential
   - Ranks topics by engagement metrics and relevance

2. **Content Creator** (`src/content_creator.py`)
   - Generates engaging scripts using Claude/GPT
   - Creates hooks, narrative arcs, and CTAs
   - Ensures accuracy and fact-checking
   - Optimizes for retention and engagement

3. **Media Generator** (`src/media_generator.py`)
   - **TTS Module**: Converts script to natural speech (ElevenLabs, Coqui TTS, or Edge TTS)
- **Audiovisual captioning**: Optional ASID‑Captioner‑3B model can analyze videos to produce rich audio/music/sound‑effect descriptions. Requires additional dependencies (`transformers`, `qwen_omni_utils`) and a GPU.
   - **Image Scraper**: Fetches relevant images from web (Pexels, Unsplash, Google Images)
   - **Stable Diffusion**: Generates custom images locally using prompts from script
   - **Caption Generator**: Creates word-level captions with timing

4. **Video Composer** (`src/video_composer.py`)
   - Assembles audio, images, and captions
   - Applies transitions, zooms, and ken burns effects
   - Adds background music and sound effects
   - Exports in optimal format for each platform (9:16 aspect ratio)

5. **Web UI** (`ui/app.py`)
   - Real-time progress monitoring
   - Manual topic input option
   - Style and tone customization
   - Preview and editing capabilities

## Installation (Approach 1: Autonomous Creation)

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for Stable Diffusion, optional but recommended)
- FFmpeg
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-shorts-generator.git
cd AI-shorts-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (Claude, ElevenLabs, etc.)
# Optional: to enable audiovisual descriptions with ASID Captioner, install the
# required Python packages (torch, transformers, qwen-omni-utils) and you can
# optionally set `ASID_MODEL_ID` in `.env` to a different Hugging Face model
# identifier (defaults to AudioVisual-Caption/ASID-Captioner-3B). No API key is
# required; the feature gracefully degrades if dependencies are missing.


# Download Stable Diffusion model (optional, for local image generation)
# python scripts/download_sd_model.py
```

## Usage (Approach 1: Autonomous Creation)

### Web UI (Recommended)

```bash
python ui/app.py
```

Then navigate to `http://localhost:7860` in your browser.

### Command Line

```bash
# Fully autonomous mode (AI picks topic)
python src/main.py --mode auto

# Manual topic mode
python src/main.py --mode manual --topic "AI breakthrough in 2026"

# Custom settings
python src/main.py --mode auto --duration 60 --style dramatic --voice-provider elevenlabs
```

## Configuration

Edit `config/settings.py` or use environment variables:

```python
# Research Settings
RESEARCH_SOURCES = ["reddit", "twitter", "google_trends", "news_api"]
TRENDING_THRESHOLD = 0.7  # Minimum viral potential score

# Content Settings
LLM_PROVIDER = "anthropic"  # or "openai"
LLM_MODEL = "claude-3-5-sonnet-20241022"
VOICE_PROVIDER = "elevenlabs"  # or "coqui", "edge"

# Media Settings
USE_STABLE_DIFFUSION = True
SD_MODEL_PATH = "./models/stable-diffusion-v1-5"
IMAGE_SOURCES = ["pexels", "unsplash", "stable_diffusion"]

# Video Settings
VIDEO_DURATION = 45  # seconds
ASPECT_RATIO = "9:16"  # vertical for shorts
FPS = 30
CAPTION_STYLE = "animated"
```

## Project Structure

```
AI-shorts-generator/
├── src/                        # Approach 1: Autonomous content creation
│   ├── main.py                 # Main orchestrator
│   ├── research_agent.py       # Trend research and topic discovery
│   ├── content_creator.py      # Script generation and narrative design
│   ├── media_generator.py      # TTS, images, stable diffusion
│   ├── video_composer.py       # Video assembly and editing
│   └── utils/
│       ├── web_scraper.py      # Web scraping utilities
│       ├── image_fetcher.py    # Image download and processing
│       └── audio_processor.py  # Audio manipulation
├── ui/
│   └── app.py                  # Gradio web interface
├── config/
│   └── settings.py             # Configuration management
├── experimental/               # All experimental approaches
│   ├── generate_shorts_v1.py   # Approach 3: Original video summarization
│   ├── generate_shorts_v2.py   # Approach 3: Enhanced version
│   ├── transcription.txt       # Example output
│   └── multi_clip_composer/    # Approach 2: Coming soon
├── models/                     # Local AI models (Stable Diffusion)
├── output/                     # Generated videos
├── cache/                      # Temporary files and cache
├── requirements.txt
├── .env.example
├── QUICKSTART.md
└── README.md
```

## 🎥 Video Labeler Package (NEW!)

**Fully local video analysis with SmolVLM - 100% free, no API costs!**

The `video_labeler` package is a powerful, reusable tool for analyzing videos using Vision-Language Models and speech recognition. It can:

- **Analyze video content** frame-by-frame using local AI (SmolVLM)
- **Extract transcripts** with word-level timestamps (Whisper)
- **Detect segments** automatically based on scene changes and speech
- **Generate keywords** and searchable metadata
- **Score quality** to find the best clip-worthy moments
- **100% Local** - No API costs, runs entirely on your machine

### Quick Test

Test the video labeler with the included example video:

```bash
# Quick test (analyzes 10 frames, < 1 minute)
python quick_test.py

# Full analysis (with transcription and segmentation, ~5 minutes)
python test_example_video.py
```

### Installation

```bash
# Windows
cd video_labeler
install_windows.bat

# Linux/Mac
cd video_labeler
chmod +x install_linux.sh
./install_linux.sh
```

### Usage Example

```python
from video_labeler import VideoLabeler

# Uses local SmolVLM by default (free!)
labeler = VideoLabeler()

# Analyze a video
analysis = labeler.analyze_video("your_video.mp4")

# Find clip-worthy segments
clips = analysis.get_clip_worthy_segments(min_score=0.7)

# Search for specific content
cooking_clips = analysis.find_segments_by_keyword("cooking")
```

### Documentation

- **[video_labeler/README.md](video_labeler/README.md)** - Full API documentation
- **[video_labeler/SMOLVLM_SETUP.md](video_labeler/SMOLVLM_SETUP.md)** - Installation and optimization guide
- **[video_labeler/INTEGRATION.md](video_labeler/INTEGRATION.md)** - Integration with AI shorts generator
- **[video_labeler/GETTING_STARTED.md](video_labeler/GETTING_STARTED.md)** - Quick start guide

### Cost Comparison

| Feature | Local (SmolVLM) | Cloud APIs |
|---------|-----------------|------------|
| Cost per video | **$0 (FREE!)** | $0.50 - $2.00 |
| Privacy | 100% local | Cloud only |
| Speed (with GPU) | Fast | API latency |
| Offline capable | ✅ Yes | ❌ No |

**Perfect for:** Analyzing article videos, finding B-roll footage, matching clips to narration, and building searchable video libraries.

## Roadmap

### Approach 1: Autonomous Content Creation ✅
- [x] Project structure and architecture
- [x] Research agent implementation
- [x] Content creator with LLM integration
- [x] TTS integration (multiple providers)
- [x] Image fetching and processing
- [x] Stable Diffusion integration (optional)
- [x] Video composition engine
- [x] Caption generation and synchronization
- [x] Web UI with Gradio
- [ ] Background music and sound effects
- [ ] Multi-platform optimization
- [ ] Analytics and feedback loop
- [ ] Batch processing and scheduling
- [ ] Cloud deployment options

### Approach 2: Multi-Video Clip Composer 🚧
- [x] **VLM integration** (SmolVLM local, GPT-4V, Claude Vision)
- [x] **Video scene analysis** with vision-language models
- [x] **Video labeler package** - Reusable video analysis tool
- [x] **Transcript extraction** with Whisper
- [x] **Automatic segmentation** based on scenes and speech
- [x] **Keyword generation** and quality scoring
- [ ] Intelligent clip extraction and matching to script segments
- [ ] Multi-clip sequencing algorithm
- [ ] Creative narrative generation
- [ ] Style-based compilation (comedy, drama, educational)
- [ ] Batch video processing
- [ ] UI for clip upload and management

### Approach 3: Video Summarization (Original) ✅
- [x] Basic implementation (v1)
- [x] Audio extraction and transcription
- [x] AI-powered summarization
- [x] Caption overlay
- [ ] Advanced highlight detection
- [ ] Multiple video platform support
- [ ] Quality optimization

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Anthropic Claude for content generation
- Stability AI for Stable Diffusion
- OpenAI Whisper for audio processing
- MoviePy and FFmpeg for video composition
