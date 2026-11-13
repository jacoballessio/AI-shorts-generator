"""Utility: attach audio description to a video's metadata using the audio captioner model.

Usage:
    python add_audio_description.py path/to/video.mp4

This script demonstrates the automatic generation of audio descriptions without running
the full pipeline or loading heavy visual models.
"""
import sys, json
from pathlib import Path

from audio_captioner import AudioCaptioner


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_audio_description.py <video_file>")
        sys.exit(1)
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print("Video file not found", video_path)
        sys.exit(1)

    metadata_path = video_path.with_suffix('.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    else:
        metadata = {}

    cap = AudioCaptioner()
    print("Generating audio description...")
    audio_desc = cap.describe_video_audio(str(video_path))
    if audio_desc:
        metadata['audio_description'] = audio_desc
        print("Audio description:", audio_desc)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Saved metadata to {metadata_path}")
    else:
        print("Audio captioner returned empty description")

if __name__ == '__main__':
    main()
