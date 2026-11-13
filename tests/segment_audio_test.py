import os, sys, json

# ensure project root is on sys.path so imports work when running script from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_captioner import AudioCaptioner
from src.training.dataset_builder import DatasetBuilder

video_path = 'data/raw_videos/sENByrMHfyk.mp4'
if not os.path.exists(video_path):
    print('video file missing')
    exit(1)

builder = DatasetBuilder()
with open('data/analysis/sENByrMHfyk/frames_analysis.json','r',encoding='utf-8') as f:
    labels = json.load(f)
metadata = json.load(open('data/raw_videos/sENByrMHfyk.json'))
transcript_segments = metadata.get('transcript_segments', [])
segments = builder._create_segments(transcript_segments, labels, [])

# verify there are no positive gaps between consecutive segments
for i in range(len(segments) - 1):
    gap = segments[i+1]['start_time'] - segments[i]['end_time']
    if gap > 0.001:
        print(f"[ERROR] Gap detected between {segments[i]['start_time']} and {segments[i+1]['start_time']}: {gap}")
        raise AssertionError("Segments contain a gap")

cap = AudioCaptioner()
segs = []
for seg in segments:
    start = seg['start_time']
    end = seg['end_time']
    d = cap.describe_video_audio_segment(video_path, start, end)
    segs.append((start, end, d))
    print(f"seg {start}-{end} => {d}")

print('Summary:', segs)