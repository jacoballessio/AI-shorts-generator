# YouTuber Training Pipeline - Quick Start Guide

## 🚀 Get Started in 1 Day

This guide walks you through collecting training data and fine-tuning a model to generate videos in a specific YouTuber's style.

---

## Prerequisites

1. **Install new dependencies:**
```bash
pip install yt-dlp newspaper3k pytesseract PyPDF2 peft bitsandbytes datasets wandb accelerate
```

2. **Tesseract OCR** (for on-screen text detection):
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

3. **GPU with 24GB+ VRAM** for training (or use cloud GPU)
   - Local: RTX 4090, A5000, A6000
   - Cloud: RunPod, Lambda Labs, Vast.ai (~$0.50/hour for A40)

---

## 🎯 Step-by-Step Workflow

### **Phase 1: Data Collection (4-6 hours)**

#### 1. Process a Single Video (Test Run)
```bash
python pipeline_orchestrator.py --video "https://www.youtube.com/watch?v=VIDEO_ID"
```

This will:
- ✅ Download video + transcript
- ✅ Label frames with SmolVLM
- ✅ Detect sources (citations, URLs, on-screen text)
- ✅ Scrape source content
- ✅ Build training example

**Output:** `data/training_dataset/VIDEO_ID_example.json`

#### 2. Process Entire Channel (50-100 videos)
```bash
# Choose a creator who cites sources (easier to start)
python pipeline_orchestrator.py --channel "https://www.youtube.com/@veritasium" --max-videos 50
```

**Time estimate:** ~5-10 minutes per video = 4-8 hours for 50 videos

#### 3. Manual QA (Optional but Recommended)
```bash
python src/data_collection/qa_interface.py
```

- Review detected sources
- Correct mistakes
- Approve quality

---

### **Phase 2: Model Training (2-4 hours)**

#### 1. Train LoRA Model
```python
from src.training.train_lora import YouTuberTrainer

trainer = YouTuberTrainer(
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    output_dir="models/veritasium_lora"
)

trainer.load_model_and_tokenizer()
trainer.setup_lora(rank=64, alpha=128)

trainer.train(
    dataset_path="data/training_dataset/youtuber_dataset.json",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    use_wandb=True  # Track training metrics
)
```

**Or run as script:**
```bash
# Edit train_lora.py to uncomment the training code
python src/training/train_lora.py
```

**Time estimate:** 
- 50 examples: ~1 hour
- 100 examples: ~2-3 hours
- On cloud A100: faster (~30 mins for 100 examples)

**Cost estimate:**
- Cloud GPU (A100): ~$2/hour × 2 hours = **$4-5 total**
- Local GPU: Free (just electricity)

---

### **Phase 3: Inference Integration (30 mins)**

#### 1. Load Fine-Tuned Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "models/veritasium_lora/final")
tokenizer = AutoTokenizer.from_pretrained("models/veritasium_lora/final")
```

#### 2. Generate Scripts
```python
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a video script generator trained on Veritasium's style.

<|start_header_id|>user<|end_header_id|>

Create a 60s video script on quantum computing.

Sources:
Source 1: Article about recent quantum breakthrough...

Generate a JSON script with segments.

<|start_header_id|>assistant<|end_header_id|>

"""

outputs = model.generate(
    tokenizer(prompt, return_tensors="pt").to("cuda"),
    max_new_tokens=2000,
    temperature=0.7
)

script = tokenizer.decode(outputs[0])
```

#### 3. Integrate with Existing Pipeline
- Modify `src/content_creator.py` to add `provider="youtuber"`
- Load fine-tuned model instead of Claude/GPT
- Generate scripts in creator's style

---

## 📊 Expected Results

### After 50 Training Examples:
- ✅ Basic voice/tone match (70-80%)
- ✅ Reasonable pacing
- ⚠️ May need manual editing

### After 100+ Training Examples:
- ✅ Strong voice match (85-90%)
- ✅ Accurate timing and pacing
- ✅ Proper source usage
- ✅ Minimal editing needed

---

## 🎛️ Component Overview

### Data Collection Components

| Component | File | Purpose |
|-----------|------|---------|
| **YouTube Scraper** | `src/data_collection/youtube_scraper.py` | Download videos + transcripts |
| **Video Labeler** | `src/data_collection/video_labeler.py` | Extract visual descriptions with SmolVLM |
| **Source Detective** | `src/data_collection/source_detective.py` | Find citations, URLs, on-screen text |
| **Source Scraper** | `src/data_collection/source_scraper.py` | Fetch article/paper content |
| **QA Interface** | `src/data_collection/qa_interface.py` | Manual review UI |

### Training Components

| Component | File | Purpose |
|-----------|------|---------|
| **Dataset Builder** | `src/training/dataset_builder.py` | Format training examples |
| **LoRA Trainer** | `src/training/train_lora.py` | Fine-tune model |

### Main Orchestrator

| Component | File | Purpose |
|-----------|------|---------|
| **Pipeline Orchestrator** | `pipeline_orchestrator.py` | Run complete pipeline |

---

## 🔧 Troubleshooting

### Video Download Issues
```bash
# Update yt-dlp
pip install -U yt-dlp

# Test download manually
yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID"
```

### OCR Not Working
```bash
# Verify tesseract installation
tesseract --version

# Set path in environment (Windows)
set PATH=%PATH%;C:\Program Files\Tesseract-OCR
```

### GPU Memory Issues
- Reduce batch size: `batch_size=2` or `batch_size=1`
- Use smaller model: Llama 3.1 8B instead of 70B
- Enable gradient checkpointing in training args

### Source Detection Missing Citations
- Check transcript quality (YouTube auto-captions can be poor)
- Use manual QA interface to add missing sources
- Adjust confidence thresholds in `source_detective.py`

---

## 💡 Tips for Best Results

1. **Start with citation-heavy creators:**
   - Veritasium, Kurzgesagt, CGP Grey
   - They explicitly reference sources

2. **Quality over quantity:**
   - 50 high-quality examples > 100 mediocre ones
   - Use QA interface to verify sources

3. **Incremental training:**
   - Train on 25 videos → test → collect 25 more → retrain
   - Iterate based on results

4. **Mix video lengths:**
   - Include short (2-3 min) and long (10+ min) videos
   - Model learns different pacing patterns

---

## 🚀 Cloud Training Options

### RunPod (Recommended)
```bash
# Rent A40 GPU (~$0.50/hour)
1. Go to runpod.io
2. Select "Community Cloud" → A40 (48GB)
3. Deploy PyTorch template
4. Upload code and data
5. Run training
```

### Lambda Labs
```bash
# A100 GPU (~$1.10/hour)
1. lambda labs → GPU Cloud
2. Select A100 instance
3. SSH into instance
4. Run training
```

### Vast.ai (Cheapest)
```bash
# Find deals as low as $0.20/hour
1. Search for A40/A100
2. Filter by reliability score
3. Deploy and train
```

---

## 📈 Next Steps After Training

1. **Test generation quality:**
   - Compare outputs to original creator
   - Check voice consistency
   - Verify timing accuracy

2. **Iterate:**
   - Collect more examples from underrepresented topics
   - Retrain with expanded dataset
   - Fine-tune hyperparameters

3. **Production deployment:**
   - Quantize model to 4-bit for faster inference
   - Deploy as API endpoint
   - Integrate with video generation pipeline

4. **Multi-creator support:**
   - Train separate LoRA for each creator
   - Switch models based on user selection
   - Mix-and-match styles

---

## 🎉 You're Ready!

Run your first video through the pipeline:
```bash
python pipeline_orchestrator.py --video "https://www.youtube.com/watch?v=YOUR_VIDEO"
```

Questions? Check the implementation files - they all have working examples commented out!
