# Local LLM Fallback Setup

When cloud LLMs (OpenAI/Anthropic) refuse to generate scripts about certain topics (copyright, sensitive content), the system can fall back to a local uncensored model.

## Why Use a Local LLM?

- **No censorship**: Cloud LLMs often refuse topics like Pokémon, brand names, or copyrighted content
- **Privacy**: Scripts generated entirely on your machine
- **Cost**: No API costs for these generations
- **VRAM management**: System automatically loads/unloads model to avoid conflicts with Stable Diffusion

## Installation

### 1. Install llama-cpp-python

```powershell
# For GPU acceleration (CUDA)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CPU only (slower)
pip install llama-cpp-python
```

### 2. Download a Quantized Model

Recommended models (8B parameters, 4-6GB):

**Option A: Llama 3.1 8B Instruct (Best quality)**
```powershell
# Download from HuggingFace
# https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

# Get the Q4_K_M quantization (good balance of quality/size)
# File: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (~4.5GB)
```

**Option B: Llama 2 7B Chat**
```powershell
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# File: llama-2-7b-chat.Q4_K_M.gguf
```

**Option C: Mistral 7B Instruct**
```powershell
# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# File: mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 3. Configure .env

Add the path to your downloaded model:

```env
LOCAL_LLM_PATH=Z:/Models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

## How It Works

1. **Primary attempt**: Script generation tries your configured LLM provider (OpenAI/Anthropic)
2. **Refusal detection**: If the LLM refuses (detects copyright/policy concerns), automatic fallback
3. **Local LLM loads**: System loads the quantized model into VRAM
4. **Script generation**: Local model generates the script (usually succeeds where cloud models refuse)
5. **Model unloads**: System immediately unloads local model to free VRAM for Stable Diffusion
6. **Continue pipeline**: Video generation continues normally

## Performance Notes

- **Loading time**: ~5-10 seconds to load model
- **Generation time**: ~30-60 seconds for a 45-second script
- **VRAM usage**: ~4-6GB during local generation (freed after)
- **Quality**: Slightly lower than GPT-4/Claude, but still good for short-form content

## Troubleshooting

**"llama-cpp-python not installed"**
```powershell
pip install llama-cpp-python
```

**"LOCAL_LLM_PATH not set or file not found"**
- Check your .env file has the correct path
- Use forward slashes: `Z:/Models/model.gguf` not `Z:\Models\model.gguf`
- Verify the file exists

**"Failed to load local LLM: CUDA out of memory"**
- Your GPU doesn't have enough VRAM
- Try a smaller quantization (Q3_K_M instead of Q4_K_M)
- Or use CPU inference (slower): Set `n_gpu_layers=0` in code

**Script quality is poor**
- Try a different model (Llama 3.1 > Mistral > Llama 2)
- Use a larger quantization if you have VRAM (Q5_K_M instead of Q4_K_M)
- Adjust temperature in `_call_local_llm()` method

## Example Workflow

```
1. User generates video about "Pokémon Violet graphics comparison"
2. OpenAI GPT-4 refuses: "I'm sorry, but I can't provide content about copyrighted game franchises"
3. System detects refusal: "⚠️  OPENAI LLM refused the request"
4. System loads local LLM: "🦖 Loading local LLM..."
5. Local model generates script successfully: "✅ Local LLM generated script successfully"
6. System unloads model: "🗑️ Unloading local LLM to free VRAM..."
7. Stable Diffusion loads for image generation
8. Video completes normally
```

## Recommended Setup

For best results:
- **Model**: Llama 3.1 8B Instruct Q4_K_M (~4.5GB)
- **VRAM**: 8GB+ GPU (RTX 3060 or better)
- **Storage**: 10GB free for model + cache

This ensures smooth fallback when cloud LLMs refuse, without compromising video quality.
