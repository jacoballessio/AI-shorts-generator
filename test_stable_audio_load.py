#!/usr/bin/env python3
"""Test script to debug Stable Audio model loading"""

import sys
import torch
from diffusers import StableAudioPipeline

print("Starting test...", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

print("About to call from_pretrained...", flush=True)
sys.stdout.flush()

try:
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float32
    )
    print("Successfully loaded pipeline!", flush=True)
    sys.stdout.flush()
    
    print(f"Moving to {device}...", flush=True)
    sys.stdout.flush()
    
    pipe = pipe.to(device)
    print("Successfully moved to device!", flush=True)
    sys.stdout.flush()
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Test complete", flush=True)
