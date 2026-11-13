# YouTuber Training Pipeline - Directory Structure

This directory contains all components for training creator-specific video generation models.

## 📁 Directory Structure

```
AI-shorts-generator/
├── src/
│   ├── data_collection/          # Data collection components
│   │   ├── __init__.py
│   │   ├── youtube_scraper.py    # Download videos + transcripts
│   │   ├── video_labeler.py      # Label frames with SmolVLM
│   │   ├── source_detective.py   # Detect research sources
│   │   ├── source_scraper.py     # Fetch source content
│   │   └── qa_interface.py       # Manual QA tool (Gradio)
│   │
│   ├── training/                  # Model training components
│   │   ├── __init__.py
│   │   ├── dataset_builder.py    # Build training dataset
│   │   └── train_lora.py         # Fine-tune with LoRA
│   │
│   └── inference/                 # Model serving (future)
│       └── __init__.py
│
├── data/                          # Data storage (created at runtime)
│   ├── raw_videos/               # Downloaded videos + metadata
│   ├── labels/                   # Frame labels from SmolVLM
│   ├── sources/                  # Scraped source content
│   └── training_dataset/         # Final training examples
│
├── models/                        # Trained models (created at runtime)
│   └── {creator}_lora/           # LoRA weights per creator
│
├── pipeline_orchestrator.py      # Main pipeline script
├── test_pipeline.py              # Component tests
├── QUICKSTART_TRAINING.md        # Quick start guide
└── YOUTUBER_TRAINING_PIPELINE.md # Full documentation with diagrams
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install yt-dlp newspaper3k pytesseract PyPDF2 peft bitsandbytes datasets wandb accelerate
```

### 2. Test Components
```bash
python test_pipeline.py --component all
```

### 3. Process a Video
```bash
python pipeline_orchestrator.py --video "https://www.youtube.com/watch?v=VIDEO_ID"
```

### 4. Process a Channel
```bash
python pipeline_orchestrator.py --channel "https://www.youtube.com/@veritasium" --max-videos 50
```

### 5. Train Model
```bash
# Edit src/training/train_lora.py and uncomment training code
python src/training/train_lora.py
```

## 📊 Component Overview

### Data Collection

| Component | Purpose | Output |
|-----------|---------|--------|
| **YouTubeScraper** | Downloads videos, transcripts, metadata | `data/raw_videos/{video_id}.json` |
| **VideoLabeler** | Extracts visual descriptions with SmolVLM | `data/labels/{video_id}.json` |
| **SourceDetective** | Finds citations, URLs, on-screen sources | List of detected sources |
| **SourceScraper** | Fetches full article/paper content | `data/sources/{video_id}_sources.json` |
| **QAInterface** | Manual review and correction UI | Corrected source files |

### Training

| Component | Purpose | Output |
|-----------|---------|--------|
| **DatasetBuilder** | Combines all data into training format | `data/training_dataset/dataset.json` |
| **YouTuberTrainer** | Fine-tunes LLM with LoRA | `models/{creator}_lora/` |

## 🔄 Pipeline Flow

```
1. YouTube URL
   ↓
2. YouTubeScraper → Video file + Metadata
   ↓
3. VideoLabeler → Frame descriptions
   ↓
4. SourceDetective → Detected sources
   ↓
5. SourceScraper → Source content
   ↓
6. DatasetBuilder → Training example
   ↓
7. YouTuberTrainer → Fine-tuned model
```

## 📖 Documentation

- **[QUICKSTART_TRAINING.md](../QUICKSTART_TRAINING.md)** - Get started in 1 day
- **[YOUTUBER_TRAINING_PIPELINE.md](../YOUTUBER_TRAINING_PIPELINE.md)** - Full technical documentation
- **[diagrams/](../diagrams/)** - Visual pipeline diagrams (PNG)

## 🧪 Testing

Test individual components:
```bash
python test_pipeline.py --component scraper      # Test YouTube downloading
python test_pipeline.py --component labeler      # Test frame labeling
python test_pipeline.py --component detective    # Test source detection
python test_pipeline.py --component source-scraper  # Test content scraping
python test_pipeline.py --component dataset      # Test dataset building
```

## 💡 Tips

1. **Start small:** Test with 1 video before processing 50
2. **Check quality:** Use QA interface to verify sources
3. **GPU memory:** Reduce batch size if OOM errors occur
4. **Incremental training:** Train on 25 → test → collect more → retrain

## 🐛 Troubleshooting

**Video download fails:**
```bash
pip install -U yt-dlp
```

**OCR not working:**
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

**GPU memory error:**
- Reduce batch size in training config
- Use smaller base model (8B instead of 70B)

**Source detection misses citations:**
- Use QA interface to add manually
- Adjust confidence thresholds
- Check transcript quality

## 📈 Expected Performance

| Dataset Size | Voice Match | Training Time | Quality |
|--------------|-------------|---------------|---------|
| 25 videos | 70-75% | 30-60 min | Basic |
| 50 videos | 80-85% | 1-2 hours | Good |
| 100 videos | 85-90% | 2-3 hours | Excellent |
| 500+ videos | 90-95% | 6-8 hours | Professional |

## 🎯 Next Steps

After training:
1. Test generation quality
2. Integrate with main pipeline
3. Deploy as API
4. Train additional creators
5. Mix-and-match styles

---

**Ready to start?** See [QUICKSTART_TRAINING.md](../QUICKSTART_TRAINING.md) for step-by-step instructions!
