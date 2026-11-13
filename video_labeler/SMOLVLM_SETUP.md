# SmolVLM Setup Guide

Complete guide for setting up **fully local** video analysis with SmolVLM.

## What is SmolVLM?

SmolVLM is a small (2B parameters), efficient vision-language model from Hugging Face that runs entirely on your local machine. Perfect for:
- ✅ **Free** - No API costs
- ✅ **Private** - Your videos never leave your machine
- ✅ **Fast** - With GPU, processes frames quickly
- ✅ **Good quality** - Despite small size, produces excellent results

## System Requirements

### Minimum (CPU Only)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB for model weights
- **OS**: Windows, Linux, or macOS
- **Python**: 3.10+

### Recommended (GPU)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or 12.1
- **RAM**: 16GB
- **Storage**: ~5GB for model

## Installation

### 1. Install PyTorch with CUDA (GPU)

For Windows with NVIDIA GPU:
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU only (slower):
```bash
pip install torch torchvision
```

For Linux/Mac, see: https://pytorch.org/get-started/locally/

### 2. Install Required Dependencies

```bash
pip install transformers accelerate pillow
```

### 3. Install Video Labeler Dependencies

```bash
pip install moviepy openai-whisper
```

## Quick Test

Test if SmolVLM works:

```python
from video_labeler import VideoLabeler, LabelerConfig

# Configure for local VLM
config = LabelerConfig(
    vlm_provider="local",
    vlm_model="HuggingFaceTB/SmolVLM-Instruct",
    use_whisper=True,
    frame_interval=30
)

# This will download the model on first run (~2-3GB)
labeler = VideoLabeler(config)
print("✅ SmolVLM loaded successfully!")

# Test on a video
analysis = labeler.analyze_video("test_video.mp4")
print(f"Analyzed {len(analysis.segments)} segments")
```

## Model Download

On first run, SmolVLM will automatically download:
- **Model**: `HuggingFaceTB/SmolVLM-Instruct` (~2GB)
- **Processor**: Tokenizer and image processor (~100MB)

Models are cached in:
- **Windows**: `C:\Users\<username>\.cache\huggingface\`
- **Linux/Mac**: `~/.cache/huggingface/`

## Performance Benchmarks

Frame analysis speed (single frame):

| Hardware | Speed | Notes |
|----------|-------|-------|
| RTX 4090 | ~0.5s | Excellent for real-time |
| RTX 3060 | ~1.5s | Good for batch processing |
| CPU (i7) | ~8s | Usable but slow |
| CPU (i5) | ~15s | Very slow |

For a 60-second video at 30 frames interval:
- **GPU (RTX 3060)**: ~30-60 seconds
- **CPU**: 4-8 minutes

## Optimization Tips

### 1. Batch Processing
```python
config = LabelerConfig(
    frame_interval=60,  # Every 2 seconds instead of 1
    batch_size=10,       # Process 10 frames in memory
    max_frames=50        # Limit total frames
)
```

### 2. GPU Memory Management

For 6GB VRAM:
```python
config = LabelerConfig(
    vlm_provider="local",
    vlm_model="HuggingFaceTB/SmolVLM-Instruct",
    batch_size=5  # Smaller batches
)
```

For 12GB+ VRAM:
```python
config = LabelerConfig(
    batch_size=20  # Larger batches = faster
)
```

### 3. Free Up VRAM When Done

```python
import torch

# After analysis
labeler = None
torch.cuda.empty_cache()
```

### 4. Use Half Precision (GPU Only)

Already enabled by default for CUDA:
```python
# Automatically uses torch.float16 on GPU
# Uses torch.float32 on CPU
```

## Troubleshooting

### Out of Memory (OOM)

**GPU OOM**:
```python
# Reduce batch size
config = LabelerConfig(batch_size=1)
```

**CPU OOM**:
```python
# Reduce frame interval and max frames
config = LabelerConfig(
    frame_interval=90,
    max_frames=20
)
```

### Slow Performance

**For CPU users**:
```python
# Ultra-fast mode
config = LabelerConfig(
    frame_interval=120,  # Every 4 seconds
    use_whisper=False,   # Skip transcription
    generate_summary=False,
    max_frames=15
)
```

### Model Download Issues

If download fails:
```python
# Manually download
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

model_name = "HuggingFaceTB/SmolVLM-Instruct"
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    _attn_implementation="eager"
)
```

**Note**: SmolVLM is based on Idefics3 architecture and requires `trust_remote_code=True`.

### CUDA Not Available

Check CUDA installation:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Alternative Models

SmolVLM is recommended, but you can try other models:

### Qwen2-VL (Larger, better quality)
```python
config = LabelerConfig(
    vlm_provider="local",
    vlm_model="Qwen/Qwen2-VL-2B-Instruct"  # ~4GB, better results
)
```

### LLaVA-NeXT (Alternative)
```python
config = LabelerConfig(
    vlm_provider="local",
    vlm_model="llava-hf/llava-v1.6-mistral-7b-hf"  # ~13GB, very good
)
```

## Production Usage

### Batch Processing Multiple Videos

```python
import torch
from video_labeler import VideoLabeler, LabelerConfig

config = LabelerConfig(
    vlm_provider="local",
    frame_interval=60,
    batch_size=10
)

labeler = VideoLabeler(config)

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

for video in videos:
    print(f"Analyzing {video}...")
    analysis = labeler.analyze_video(video, output_dir=f"./cache/{video}")
    analysis.to_json(f"./cache/{video}_analysis.json")
    
    # Clear CUDA cache between videos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("✅ All videos analyzed!")
```

### Pre-load Model for Server

```python
# Load model once at startup
labeler = VideoLabeler(LabelerConfig(vlm_provider="local"))

# Use for multiple requests
def analyze_video_endpoint(video_path):
    return labeler.analyze_video(video_path)
```

## Cost Comparison

### Local (SmolVLM)
- **Setup cost**: $0 (free)
- **Per-video cost**: $0 (free)
- **Hardware**: One-time GPU purchase if needed

### Cloud APIs
- **Setup cost**: $0
- **Per-video cost**: $0.10 - $1.00+ depending on frames
- **100 videos**: $10 - $100+
- **1000 videos**: $100 - $1000+

**Break-even**: If you plan to analyze 50+ videos, local is worth it!

## GPU Recommendations

Best value GPUs for video labeling:

| GPU | VRAM | Price | Speed | Notes |
|-----|------|-------|-------|-------|
| RTX 3060 | 12GB | ~$300 | Good | Best value |
| RTX 4060 Ti | 16GB | ~$450 | Better | Future-proof |
| RTX 4070 | 12GB | ~$550 | Fast | Overkill but fast |

## Next Steps

1. **Test on sample video**: Try with a short video first
2. **Optimize settings**: Adjust `frame_interval` and `batch_size`
3. **Cache analyses**: Save JSON to avoid re-analyzing
4. **Integrate with AI shorts**: Use analyzed videos in your pipeline

## Support

- **Hugging Face**: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct
- **Issues**: Report on GitHub
- **Documentation**: See README.md

## Example Output

With SmolVLM, you get the same quality output as cloud APIs:

```json
{
  "description": "Person cooking pasta in a large pot on stove",
  "objects": ["person", "pot", "stove", "pasta", "kitchen"],
  "actions": ["cooking", "stirring", "pouring"],
  "scene_type": "indoor",
  "mood": "focused"
}
```

100% local, 100% private, 100% free! 🎉
