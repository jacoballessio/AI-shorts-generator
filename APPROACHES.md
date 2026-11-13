# AI Shorts Generator - Three Approaches

This project now supports **three different approaches** to AI-powered short-form video creation. Each serves different use cases and workflows.

---

## 📊 Comparison Table

| Feature | Approach 1: Autonomous | Approach 2: Multi-Clip | Approach 3: Summarization |
|---------|------------------------|------------------------|---------------------------|
| **Input** | None (AI finds topics) | Multiple video clips | Single long video |
| **AI Role** | Full autonomy | Creative compiler | Analyzer & condenser |
| **Primary Tech** | LLM + TTS + Image Gen | VLM + LLM | Whisper + LLM + CV |
| **Output Style** | Original content | Creative compilation | Highlight reel |
| **Use Case** | Viral content creation | Remix & storytelling | Long-to-short conversion |
| **Status** | ✅ Production Ready | 🚧 Experimental | ✅ Working (v1, v2) |

---

## 🤖 Approach 1: Autonomous Content Creation

**"Let AI create viral shorts from scratch"**

### What It Does
- AI researches trending topics online
- Generates engaging scripts
- Creates TTS voiceover
- Fetches/generates images
- Composes professional video

### When to Use
- You want to ride trending topics
- Minimal manual input desired
- Creating content at scale
- News, facts, educational shorts

### Example
```bash
python src/main.py --mode auto --style dramatic
```
Output: 45-second video about a trending AI breakthrough

### Location
- Main code: `src/`
- UI: `ui/app.py`
- Docs: `README.md`, `QUICKSTART.md`

---

## 🎭 Approach 2: Multi-Clip Composer (VLM)

**"Turn random clips into creative stories"**

### What It Does
- Accepts multiple raw video files
- Uses VLMs to understand content
- Extracts interesting moments
- Creates narrative connections
- Compiles themed videos

### When to Use
- You have lots of raw footage
- Want creative remixes
- Making compilations with narrative
- Experimental/artistic projects

### Example
```bash
# (Coming soon)
python experimental/multi_clip_composer.py \
  --clips "cat_*.mp4" \
  --theme "Create a horror story where cats are plotting"
```

### Location
- Code: `experimental/multi_clip_composer.py`
- Status: **Under development**

---

## 📹 Approach 3: Video Summarization

**"Condense long videos into shorts"**

### What It Does
- Transcribes long video
- AI summarizes key points
- Extracts highlight clips
- Adds captions
- Creates short version

### When to Use
- You have long-form content
- Want to create teasers
- Podcast → clips
- Lecture → summaries

### Example
```bash
python experimental/generate_shorts_v2.py input_video.mp4 --summary_length 30
```

### Location
- Code: `experimental/generate_shorts_v1.py`, `v2.py`
- Status: **Working** (basic implementation)

---

## 🚀 Which Approach Should You Use?

### Choose Approach 1 if:
- ✅ You want fully autonomous content
- ✅ You need trending/viral topics
- ✅ You want original created content
- ✅ You have API keys for LLMs

### Choose Approach 2 if:
- ✅ You have raw video clips
- ✅ You want creative remixes
- ✅ You enjoy experimental features
- ✅ You want AI-driven storytelling

### Choose Approach 3 if:
- ✅ You have long videos to condense
- ✅ You want highlight reels
- ✅ You need simple summarization
- ✅ You prefer tried & tested methods

---

## 💡 Can They Work Together?

**Yes!** Here are some combination ideas:

1. **Approach 1 → 3**: Generate long content autonomously, then summarize to different lengths
2. **Approach 2 → 1**: Use multi-clip to create base content, then regenerate with new narration
3. **Approach 3 → 2**: Summarize multiple long videos, then compile summaries into themed collection

---

## 🔮 Future Roadmap

### Approach 1 (Autonomous)
- [ ] Background music library
- [ ] Multi-platform optimization
- [ ] Scheduled/batch generation
- [ ] Analytics integration

### Approach 2 (Multi-Clip)
- [ ] GPT-4 Vision integration
- [ ] Claude Vision integration
- [ ] Scene detection algorithms
- [ ] Creative narrative engine
- [ ] Style presets (comedy, drama, horror)

### Approach 3 (Summarization)
- [ ] Advanced highlight detection
- [ ] Multi-video support
- [ ] Speaker diarization
- [ ] Chapter detection

---

## 📦 Installation

All approaches use the same base installation:

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

**Required for all:**
- Python 3.10+
- FFmpeg

**Approach 1 additionally needs:**
- ANTHROPIC_API_KEY or OPENAI_API_KEY
- Optional: PEXELS_API_KEY, REDDIT_CLIENT_ID

**Approach 2 additionally needs:**
- GPT-4 Vision or Claude Vision API access
- OpenCV (included)

**Approach 3 needs:**
- Basic setup only

---

## 🎯 Quick Start by Approach

### Approach 1: Autonomous
```bash
python ui/app.py
# Click "Generate Video" in the web UI
```

### Approach 2: Multi-Clip (Coming Soon)
```bash
python experimental/multi_clip_composer.py --help
```

### Approach 3: Summarization
```bash
python experimental/generate_shorts_v2.py your_video.mp4
```

---

## 📚 More Information

- Full documentation: [README.md](README.md)
- Quick setup: [QUICKSTART.md](QUICKSTART.md)
- Examples: [examples.py](examples.py)

---

**Choose your approach and start creating!** 🎬
