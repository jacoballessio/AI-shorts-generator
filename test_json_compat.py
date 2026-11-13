#!/usr/bin/env python3
"""Test JSON compatibility with video generation."""

from generate_video_from_example import VideoGenerator

vg = VideoGenerator()
data = vg.load_example('./new_example.xmlish.txt.raw.json')

print("✓ JSON loaded successfully!")
print(f"  Creator: {data['input']['creator']}")
print(f"  Title: {data['input']['title']}")
print(f"  Segments: {len(data['output']['segments'])}")
print(f"  Duration: {data['input']['duration']}s")

# Verify segment structure
for i, seg in enumerate(data['output']['segments']):
    print(f"\n  Segment {i+1}:")
    print(f"    Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
    print(f"    Duration: {seg['duration']:.2f}s")
    print(f"    Scene: {seg['scene_description'][:60]}...")
    print(f"    Transcript: {seg['transcript'][:60]}...")
