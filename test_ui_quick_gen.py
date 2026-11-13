#!/usr/bin/env python3
"""Quick test of the improved UI rendering"""

import json
from pathlib import Path

# Test with first 2 segments only to save time
config = {
    "input": {
        "sources": [],
        "duration": 48,
        "creator": "Daniel Hentschel",
        "topic": "People & Blogs",
        "title": "Texting be like… #texting #text #friendship #friend"
    },
    "output": {
        "segments": []
    },
    "metadata": {
        "video_id": "test_ui_quick",
        "upload_date": "20240209",
        "view_count": 136728
    }
}

# Load original data
with open('data/training_dataset/sENByrMHfyk_example.json') as f:
    original = json.load(f)

# Take first 2 segments only
config["output"]["segments"] = original["output"]["segments"][:2]

# Save as quick test file
with open('data/training_dataset/test_ui_quick.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Created test_ui_quick.json with 2 segments")
print("Run: python generate_video_from_example.py data/training_dataset/test_ui_quick.json")
