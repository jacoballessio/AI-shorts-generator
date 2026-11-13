"""Check analysis file for ui_elements."""
import json

with open('data/analysis/sENByrMHfyk_analysis.json') as f:
    analysis = json.load(f)

print(f"Total segments: {len(analysis['segments'])}")
print("\nFirst 10 frames:")
for segment in analysis['segments'][:2]:
    for i, frame in enumerate(segment['key_frames'][:5]):
        print(f"  Frame {frame['frame_number']}: ui={len(frame.get('ui_elements', []))}, desc_len={len(frame.get('description', ''))}")
        if i == 0:
            print(f"    Description: {frame.get('description', '')[:80]}")
