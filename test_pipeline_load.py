#!/usr/bin/env python3
import sys
sys.stdout = sys.__stdout__  # Force unbuffered output

print("Step 1", file=sys.stderr, flush=True)

import torch
print("Step 2: torch imported", file=sys.stderr, flush=True)

from diffusers import StableAudioPipeline
print("Step 3: diffusers imported", file=sys.stderr, flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Step 4: Using device {device}", file=sys.stderr, flush=True)

print("Step 5: About to load pipeline", file=sys.stderr, flush=True)
sys.stderr.flush()

try:
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float32
    )
    print("Step 6: Pipeline loaded", file=sys.stderr, flush=True)
    
    pipe = pipe.to(device)
    print(f"Step 7: Pipeline moved to {device}", file=sys.stderr, flush=True)
    
    print("SUCCESS", file=sys.stderr, flush=True)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
