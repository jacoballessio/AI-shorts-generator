# YouTuber-Style Video Generation Training Pipeline

## Overview
This document outlines a system for training an LLM to create videos in the style of specific YouTubers by learning from their existing content and reverse-engineering their source material.

---

## 1. Complete Training Pipeline

![Training Pipeline](diagrams/1_training_pipeline.png)

<details>
<summary>View Mermaid Code</summary>

```mermaid
graph TB
    subgraph "Training Data Collection"
        A[Existing YouTube Videos] --> B[Video Labeler]
        B --> C[Video Metadata<br/>- Transcript<br/>- Visual clips with descriptions<br/>- Timestamps<br/>- Duration]
        
        A --> D[Source Detective]
        D --> E[Extract Sources<br/>- Mentioned articles<br/>- Implied research<br/>- Referenced papers<br/>- News links]
        
        E --> F[Source Scraper]
        F --> G[Source Content<br/>- Full article text<br/>- Key points<br/>- Images/media<br/>- Context]
    end
    
    subgraph "Training Dataset Creation"
        C --> H[Training Examples]
        G --> H
        
        H --> I["INPUT:<br/>- Source material<br/>- Target duration<br/>- Creator style<br/><br/>OUTPUT:<br/>- Full transcript<br/>- Clip descriptions<br/>- Timestamps<br/>- Visual cues"]
    end
    
    subgraph "Model Training"
        I --> J[Fine-tune LLM<br/>e.g., Llama 3.1 70B]
        J --> K[YouTuber-Specific Model<br/>Learns:<br/>- Creator's voice/tone<br/>- Pacing patterns<br/>- Visual storytelling<br/>- Hook structures]
    end
    
    subgraph "Inference Pipeline"
        L[New Source Material<br/>- Articles<br/>- Research<br/>- News] --> K
        M[Config<br/>- Duration: 60s<br/>- Style: Veritasium] --> K
        
        K --> N[Generated Script<br/>JSON output<br/>- Transcript segments<br/>- Clip descriptions<br/>- Timing]
        
        N --> O[Media Generation]
        O --> P[Search/Generate Clips]
        O --> Q[Search/Generate Images]
        
        P --> R[Video Composer]
        Q --> R
        N --> R
        
        R --> S[Final Video<br/>In creator's style]
    end
    
    style J fill:#ff6b6b
    style K fill:#ff6b6b
    style S fill:#51cf66
```
</details>

---

## 2. Training Example Breakdown

![Training Example](diagrams/2_training_example.png)

<details>
<summary>View Mermaid Code</summary>

```mermaid
graph LR
    subgraph "Training Example Structure"
        A[Source Material] --> B{Model}
        C[Metadata] --> B
        
        B --> D[Predicted Output]
        
        E[Ground Truth<br/>Actual Video] -.Compare.-> F[Loss Calculation]
        D -.-> F
        
        F --> G[Backpropagation]
        G --> B
    end
    
    subgraph "Source Material Input"
        A1["Article 1: Full text<br/>Article 2: Summary<br/>Research: Key findings<br/>News: Headlines + context<br/><br/>Total tokens: ~3000"]
    end
    
    subgraph "Metadata Input"
        C1["Duration: 8m 34s<br/>Creator: Veritasium<br/>Topic: Physics<br/>Target audience: General<br/>Upload date: 2024"]
    end
    
    subgraph "Expected Output Format"
        D1["segments: [<br/>  {<br/>    timestamp: 0-15s<br/>    text: 'Hook transcript...'<br/>    visual: 'Dramatic experiment'<br/>    clip_type: 'video'<br/>    pacing: 'fast'<br/>  },<br/>  {<br/>    timestamp: 15-45s<br/>    text: 'Explanation...'<br/>    visual: 'Animation of concept'<br/>    clip_type: 'animation'<br/>    references: [Article 1]<br/>  },<br/>  ...<br/>]"]
    end
    
    subgraph "What Model Learns"
        L1["1. Voice Pattern<br/>- Sentence structure<br/>- Vocabulary choices<br/>- Tone markers<br/><br/>2. Content Strategy<br/>- Hook placement<br/>- Information density<br/>- Source citation style<br/><br/>3. Visual Timing<br/>- When to use clips vs images<br/>- Transition patterns<br/>- B-roll frequency<br/><br/>4. Pacing<br/>- Words per second<br/>- Pause placement<br/>- Segment lengths"]
    end
    
    A1 -.Input.-> A
    C1 -.Input.-> C
    D1 -.Output.-> D
    
    style B fill:#ff6b6b
    style L1 fill:#51cf66
```
</details>

---

## 3. Source Detective Pipeline

![Source Detective](diagrams/3_source_detective.png)

<details>
<summary>View Mermaid Code</summary>

```mermaid
graph TB
    subgraph "Source Detective Pipeline"
        A[YouTube Video] --> B[Extract Transcript<br/>Whisper or YT API]
        A --> C[Video Labeler<br/>SmolVLM analysis]
        
        B --> D[NLP Analysis]
        D --> E["Explicit mentions:<br/>- 'According to Nature...'<br/>- 'Study published in...'<br/>- URL overlays<br/>- Description links"]
        
        D --> F["Implicit references:<br/>- Paraphrased facts<br/>- Statistics cited<br/>- Technical terms<br/>- Event references"]
        
        C --> G["Visual clues:<br/>- Paper screenshots<br/>- Website captures<br/>- News headlines<br/>- Research graphics"]
        
        E --> H[URL Extractor]
        F --> I[Fact Checker]
        G --> J[OCR + Image Search]
        
        H --> K[Source URLs]
        I --> K
        J --> K
        
        K --> L[Content Scraper]
        L --> M["Source Database:<br/>- Original URL<br/>- Full text<br/>- Publication date<br/>- Key excerpts<br/>- Media assets<br/>- Credibility score"]
    end
    
    subgraph "Source Matching Algorithm"
        M --> N[Match sources to<br/>transcript segments]
        B --> N
        
        N --> O["Annotated Transcript:<br/>Each segment tagged with:<br/>- Source IDs<br/>- Confidence score<br/>- Quote vs paraphrase<br/>- Fact-check status"]
    end
    
    subgraph "Training Data Output"
        O --> P["Training Example:<br/><br/>INPUT = Sources<br/>OUTPUT = Transcript + Visuals<br/><br/>Linkage preserved"]
    end
    
    style A fill:#ffd43b
    style M fill:#51cf66
    style P fill:#ff6b6b
```
</details>

---

## 4. Implementation Roadmap

![Implementation Roadmap](diagrams/4_implementation_roadmap.png)

<details>
<summary>View Mermaid Code</summary>

```mermaid
graph TB
    A[Select Target Creator] --> B[Scrape 50-100 videos]
    B --> C[Process videos]
    C --> D[Manual QA]
    D --> E[Dataset ready]
    
    E --> F[Choose base model]
    F --> G[Fine-tune with LoRA]
    G --> H[Training config]
    H --> I[Validation]
    
    I --> J[Deploy model]
    J --> K[API endpoint]
    K --> L[Integration]
    
    L --> M[Expected Performance]
    
    style E fill:#ffd43b
    style J fill:#ff6b6b
    style M fill:#51cf66
```
</details>

---

## Key Components Explained

### 1. Video Labeler
Uses your existing SmolVLM system to:
- Extract visual descriptions frame-by-frame
- Identify clip-worthy segments
- Detect on-screen text and graphics
- Generate timestamps for all visual elements

### 2. Source Detective (Critical Component)
Reverse-engineers what research the creator used:
- **Explicit mentions**: "According to a study in Nature..."
- **Visual clues**: Screenshots of papers, websites, graphs (OCR)
- **Implicit references**: Paraphrased facts, statistics, technical terms
- **Description links**: URLs in video description

### 3. Training Dataset Format

Each example maps:
```
INPUT:
- Source 1: [Full article text from Nature.com]
- Source 2: [Research paper summary]
- Source 3: [News article with context]
- Metadata: duration=514s, creator="Veritasium", topic="Physics"

OUTPUT:
- Segment 1: {text: "Hook...", visual: "Experiment", clip_type: "video", time: 0-15s}
- Segment 2: {text: "Content...", visual: "Animation", references: [Source 1], time: 15-45s}
- ...
```

### 4. What Makes This Work

**Patterns the model learns:**
- **Voice fingerprint**: Sentence structures, vocabulary, tone markers unique to creator
- **Content strategy**: How they transform dry research into engaging narratives
- **Visual timing**: When to use clips vs images, transition patterns
- **Pacing**: Words per second, pause placement, segment lengths
- **Hook structures**: Opening patterns that grab attention
- **Citation style**: How they reference sources (if at all)

### 5. Implementation Challenges

**Hardest part: Source Detective**
- Need NLP to detect citations and paraphrases
- OCR for on-screen text
- Reverse image search for screenshots
- Manual QA to verify source matches

**Solutions:**
- Start with creators who explicitly cite sources (easier)
- Use LLMs to suggest likely sources based on content
- Build confidence scores (high confidence = direct quote, low = inferred)
- Manual review for first 50-100 examples

### 6. Resource Requirements

**Training:**
- Base model: Llama 3.1 70B or Mixtral 8x22B
- Method: LoRA fine-tuning (4-8GB VRAM overhead)
- Cost: ~$50-100 on cloud GPU (RunPod, Lambda Labs)
- Time: 1-2 weeks for 100 examples

**Inference:**
- 4-bit quantized model (~40GB)
- RTX 4090 or A6000 sufficient
- Generation time: ~30 seconds per script
- Can run locally or via API

### 7. Expected Results

**After 100 training examples:**
- ✅ Voice/tone matches 80-90%
- ✅ Pacing and timing accurate
- ✅ Visual selection appropriate
- ⚠️ May need manual editing for polish

**After 500+ examples:**
- ✅ Near-perfect voice match
- ✅ Minimal editing needed
- ✅ Can handle multiple creator styles
- ✅ Production-ready quality

---

## Next Steps

1. **Prototype Source Detective** with 10 videos
2. **Manual annotation** to validate approach
3. **Build training pipeline** for 50-100 videos
4. **Fine-tune model** with LoRA
5. **Evaluate outputs** against held-out examples
6. **Integrate** with current video generation system

This approach transforms generic AI content into creator-specific videos by learning from how real YouTubers transform research into engaging narratives.
