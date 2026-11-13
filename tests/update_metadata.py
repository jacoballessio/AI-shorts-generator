import os, sys, json
# ensure top‑level package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_captioner import AudioCaptioner
from src.training.dataset_builder import DatasetBuilder

md_path = 'data/raw_videos/sENByrMHfyk.json'
metadata = json.load(open(md_path, 'r', encoding='utf-8'))

builder = DatasetBuilder()
labels = json.load(open('data/analysis/sENByrMHfyk/frames_analysis.json', 'r', encoding='utf-8'))
transcript_segments = metadata.get('transcript_segments', [])
segments = builder._create_segments(transcript_segments, labels, [])

cap = AudioCaptioner()
seg_audio_descs = []
for seg in segments:
    desc = cap.describe_video_audio_segment('data/raw_videos/sENByrMHfyk.mp4', seg['start_time'], seg['end_time'])
    print(f"seg {seg['start_time']}-{seg['end_time']} -> {desc}")
    seg_audio_descs.append(desc)

metadata['segment_audio_descriptions'] = seg_audio_descs
metadata['audio_description'] = cap.describe_video_audio('data/raw_videos/sENByrMHfyk.mp4')

with open(md_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print('metadata updated')
