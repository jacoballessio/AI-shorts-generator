# Common Issues & Solutions

## Import Errors

### ❌ "cannot import name 'FLAX_WEIGHTS_NAME' from 'transformers.utils'"

**Cause**: Version incompatibility between `transformers` and `diffusers` packages with Python 3.13

**Solution**: Disable Stable Diffusion in `.env`:
```bash
USE_STABLE_DIFFUSION=false
```

Or try downgrading packages:
```bash
pip install transformers==4.35.0 diffusers==0.25.0
```

**Impact**: SD image generation disabled, web images and placeholders used instead

---

### ❌ "No module named 'moviepy.editor'"

**Cause**: MoviePy not installed or using wrong version

**Solution**:
```bash
pip install moviepy
```

If still failing, try:
```bash
pip install moviepy==1.0.3
```

**Note**: MoviePy is **required** for video composition in Approach 1

---

### ⚠️ "PRAW in asynchronous environment" warning

**Cause**: Using synchronous `praw` instead of async version

**Solution**: Already fixed! Install asyncpraw:
```bash
pip install asyncpraw
```

**Impact**: Just a warning, but asyncpraw is faster and proper for async code

---

## Research Issues

### ❌ "Found 0 high-potential topics"

**Causes**:
1. Viral threshold too high (0.6+)
2. No API keys configured
3. Rate limiting

**Solutions**:

**Lower the threshold** in `.env`:
```bash
VIRAL_THRESHOLD=0.4  # or even 0.3 for more results
```

**Check API keys** are set:
```bash
# At least one is needed for research:
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
NEWS_API_KEY=your_key
```

**Disable failing sources**:
```bash
# If Reddit auth fails, use only web sources:
RESEARCH_SOURCES=google_trends,news
```

**Expected output**:
```
✅ Found 10-30 topics with threshold 0.4
✅ Found 3-10 topics with threshold 0.6
```

---

## API Key Issues

### ❌ "No LLM API key found"

**Solution**: Add to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
# OR
OPENAI_API_KEY=sk-...
```

Get keys from:
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys

---

### ⚠️ Reddit API not working

**Solution**: 

1. Create Reddit app: https://www.reddit.com/prefs/apps
2. Click "create app" or "create another app"
3. Select "script" type
4. Add to `.env`:
```bash
REDDIT_CLIENT_ID=your_14_char_id
REDDIT_CLIENT_SECRET=your_27_char_secret
```

**Alternative**: Disable Reddit research:
```bash
RESEARCH_SOURCES=google_trends,news
```

---

## Video Generation Issues

### ❌ FFmpeg not found

**Solution**: Install FFmpeg:

**Windows**:
```bash
winget install ffmpeg
# or
choco install ffmpeg
```

**macOS**:
```bash
brew install ffmpeg
```

**Linux**:
```bash
sudo apt install ffmpeg
```

**Verify**:
```bash
ffmpeg -version
```

---

### ⚠️ Video generation is slow

**Causes**:
1. Using CPU instead of GPU for Stable Diffusion
2. Downloading models first time
3. Large video resolution

**Solutions**:

**Disable SD** if no GPU:
```bash
USE_STABLE_DIFFUSION=false
```

**Use web images only**:
```bash
PEXELS_API_KEY=your_key
USE_STABLE_DIFFUSION=false
```

**Lower resolution** (not recommended):
```python
# In config/settings.py
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
```

**Expected times**:
- With GPU: 2-3 minutes per video
- CPU only: 5-10 minutes per video

---

## Gradio UI Issues

### ⚠️ "theme parameter moved to launch()" warning

**Status**: Fixed in latest code, cosmetic warning only

**Solution**: Pull latest changes or ignore (doesn't affect functionality)

---

### ❌ UI won't start

**Diagnostic**:
```bash
python test_imports.py
```

Should show all ✓ marks.

**If imports fail**:
```bash
pip install -r requirements.txt --force-reinstall
```

---

## Memory Issues

### ❌ "Out of memory" during Stable Diffusion

**Solution**: Disable SD or use smaller model:
```bash
USE_STABLE_DIFFUSION=false
```

Or reduce batch size in code (advanced).

---

## Dependencies

### Quick reinstall all dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Minimal working setup (no Stable Diffusion):
```bash
pip install anthropic openai asyncpraw requests beautifulsoup4 pillow moviepy gradio edge-tts python-dotenv
```

---

## Verification Commands

Test individual components:

```bash
# Test LLM
python -c "from anthropic import Anthropic; print('Anthropic OK')"

# Test MoviePy
python -c "from moviepy.editor import VideoFileClip; print('MoviePy OK')"

# Test Gradio
python -c "import gradio; print(f'Gradio {gradio.__version__}')"

# Test all imports
python test_imports.py
```

---

## Getting Help

1. Check [STATUS.md](STATUS.md) for current implementation status
2. Review [APPROACHES.md](APPROACHES.md) for different use cases
3. See [QUICKSTART.md](QUICKSTART.md) for setup guide
4. Run diagnostics: `python test_imports.py`

---

## Known Working Configuration

```
Python: 3.10-3.12 (3.13 has diffusers issues)
Gradio: 6.6.0
MoviePy: 1.0.3
asyncpraw: 7.7.1
anthropic: 0.18.0+
edge-tts: 6.1.9+

With USE_STABLE_DIFFUSION=false
```

---

## Quick Fixes Summary

**Most common issues**:
```bash
# 1. Lower viral threshold
echo "VIRAL_THRESHOLD=0.4" >> .env

# 2. Disable Stable Diffusion
echo "USE_STABLE_DIFFUSION=false" >> .env

# 3. Install asyncpraw
pip install asyncpraw

# 4. Verify MoviePy
pip install moviepy --upgrade

# 5. Check API keys exist
cat .env | grep API_KEY
```

**Then restart UI**:
```bash
python ui/app.py
```
