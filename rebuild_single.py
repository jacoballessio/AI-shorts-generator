"""Rebuild training example for a specific video."""

import json
from pathlib import Path
from src.training.dataset_builder import DatasetBuilder

video_id = "sENByrMHfyk"
base_dir = Path("data")

# Load metadata
metadata_path = base_dir / "raw_videos" / f"{video_id}.json"
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load labels from main analysis.json (has ui_elements)
analysis_file = base_dir / "analysis" / f"{video_id}_analysis.json"
with open(analysis_file, 'r', encoding='utf-8') as f:
    analysis = json.load(f)
    # Extract all key frames from all segments
    labels = []
    for segment in analysis.get('segments', []):
        for frame in segment.get('key_frames', []):
            labels.append({
                'timestamp': frame.get('timestamp', 0),
                'description': frame.get('description', ''),
                'frame_number': frame.get('frame_number', 0),
                'text_visible': frame.get('text_visible', ''),
                'ui_elements': frame.get('ui_elements', [])
            })

print(f"Loaded {len(labels)} frames")
print(f"Checking first frame:")
print(f"  Description: {labels[0]['description'][:100]}")
print(f"  UI elements: {len(labels[0].get('ui_elements', []))}")

# Load sources
sources_path = base_dir / "sources" / f"{video_id}_sources.json"
if sources_path.exists():
    with open(sources_path, 'r', encoding='utf-8') as f:
        sources = json.load(f)
else:
    sources = []

# Load transcript from analysis
analysis_path = base_dir / "analysis" / video_id / "transcript.json" 
if analysis_path.exists():
    with open(analysis_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
        metadata['transcript'] = transcript_data.get('text', '')
        metadata['transcript_segments'] = transcript_data.get('segments', [])

# Build training example
builder = DatasetBuilder(output_dir=str(base_dir / "training_dataset"))
training_example = builder.build_training_example(metadata, labels, sources)

# Save
example_path = base_dir / "training_dataset" / f"{video_id}_example.json"
with open(example_path, 'w', encoding='utf-8') as f:
    json.dump(training_example, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Training example saved: {example_path}")
print(f"   Segments: {len(training_example['output']['segments'])}")

# Check if UI elements are populated
for i, seg in enumerate(training_example['output']['segments'][:5]):
    scene_desc = seg.get('scene_description', '')
    ui_html = seg.get('ui_elements_html', '')
    
    # Check if scene_description contains JSON
    is_json = scene_desc.strip().startswith('{')
    
    print(f"   Segment {i}:")
    print(f"     Scene is JSON: {is_json}")
    print(f"     Scene length: {len(scene_desc)} chars")
    print(f"     UI HTML length: {len(ui_html)} chars")
    if is_json:
        print(f"     [WARNING] Scene description contains JSON!")
