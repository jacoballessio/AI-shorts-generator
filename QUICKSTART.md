# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-shorts-generator.git
cd AI-shorts-generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and add your API keys

You can also optionally configure the ASID Captioner model by setting
`ASID_MODEL_ID` or just leave it disabled; the code will skip captioning if the
package dependencies (`torch`, `transformers`, `qwen-omni-utils`) are not
installed.
# At minimum, you need ONE of these:
# - ANTHROPIC_API_KEY (recommended)
# - OPENAI_API_KEY
```

**Get API Keys:**
- Anthropic Claude: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys
- Reddit API: https://www.reddit.com/prefs/apps
- News API: https://newsapi.org/
- Pexels: https://www.pexels.com/api/

### 3. Run the Web UI

```bash
python ui/app.py
```

Then open your browser to: `http://localhost:7860`

### 4. Generate Your First Video

**Option A: Web UI (Recommended)**
1. Go to the "🤖 Autonomous Generation" tab
2. Click "🚀 Generate Video"
3. Wait 2-5 minutes
4. Download your video!

**Option B: Command Line**
```bash
# Autonomous mode (AI picks topic)
python src/main.py --mode auto

# Manual topic mode
python src/main.py --mode manual --topic "Your Topic Here"

# Custom settings
python src/main.py --mode auto --duration 60 --style dramatic
```

## 📖 Usage Examples

### Autonomous Generation
The AI will research trending topics and create a video:
```bash
python src/main.py --mode auto --duration 45 --style engaging
```

### Manual Topic
Specify your own topic:
```bash
python src/main.py --mode manual --topic "AI Breakthrough in 2026" --description "Details about the breakthrough..."
```

### Batch Generation
Generate multiple videos at once:
```bash
python src/main.py --mode batch --count 5
```

## ⚙️ Configuration

### Video Settings
Edit `config/settings.py` or set environment variables:

```bash
# Video duration (seconds)
DEFAULT_DURATION=45

# Video style
DEFAULT_STYLE=engaging  # engaging, dramatic, educational, humorous

# Narrative tone
DEFAULT_TONE=informative  # informative, casual, professional, excited
```

### Caption Styles
- `animated` - TikTok-style word-by-word (default)
- `karaoke` - Highlighted phrases
- `static` - Simple subtitles

### TTS Options
- `edge` - Free, good quality (default, no API key needed)
- `elevenlabs` - Premium, best quality (requires API key)

## 🎨 Customization

### Change Voice
In `.env`:
```bash
EDGE_TTS_VOICE=en-US-GuyNeural  # Male voice
# or
EDGE_TTS_VOICE=en-GB-SoniaNeural  # British accent
```

### Enable Stable Diffusion
```bash
USE_STABLE_DIFFUSION=true
SD_MODEL_PATH=runwayml/stable-diffusion-v1-5
```
Note: Requires GPU for good performance

### Research Sources
```bash
RESEARCH_SOURCES=reddit,google_trends,news
```

## 🐛 Troubleshooting

### "No LLM API key found"
- Add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to `.env`

### "MoviePy error"
- Install FFmpeg: https://ffmpeg.org/download.html
- Windows: `winget install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### Slow video generation
- First run downloads Stable Diffusion models (one-time)
- Without GPU, image generation is slower
- Disable Stable Diffusion: `USE_STABLE_DIFFUSION=false`

### Image generation failing
- Add `PEXELS_API_KEY` to `.env`
- Or enable Stable Diffusion

### Reddit research not working
- Add Reddit API credentials to `.env`
- Or disable: `RESEARCH_SOURCES=google_trends,news`

## 📊 Expected Performance

**Typical Generation Time:**
- Research: 10-30 seconds
- Script creation: 5-10 seconds
- Media generation: 30-120 seconds
  - TTS: 5-10 seconds
  - Stable Diffusion images: 10-60 seconds (GPU) or 2-5 minutes (CPU)
  - Web images: 10-20 seconds
- Video composition: 20-60 seconds

**Total: 2-5 minutes per video**

## 🔥 Tips for Viral Content

1. **Use Autonomous Mode** - AI finds what's actually trending
2. **Keep it Short** - 30-60 seconds performs best
3. **Dramatic Style** - Works well for news and trending topics
4. **Excited Tone** - Increases engagement
5. **Enable Stable Diffusion** - Unique images perform better than stock

## 📚 Next Steps

- Check out [README.md](README.md) for full documentation
- Explore the Web UI features
- Experiment with different styles and tones
- Set up scheduling for automated posting
- Check `experimental/` for legacy scripts

## 🆘 Need Help?

- Check the [Issues](https://github.com/yourusername/AI-shorts-generator/issues)
- Read the full [README.md](README.md)
- Review configuration in `config/settings.py`

## 🎉 You're Ready!

Start generating viral short-form videos with AI!

```bash
python ui/app.py
```
