"""
Main orchestrator for the complete data collection pipeline.
"""
import os
import json
import re
from pathlib import Path
from typing import List, Optional
import argparse

from src.data_collection.youtube_scraper import YouTubeScraper
# Import the full video_labeler package with Whisper support
from video_labeler import VideoLabeler, LabelerConfig
from src.data_collection.source_detective import SourceDetective
# SourceScraper imported lazily due to newspaper3k compatibility issues
from src.training.dataset_builder import DatasetBuilder


class PipelineOrchestrator:
    """Orchestrate the complete YouTuber training data collection pipeline."""
    
    def __init__(self, base_dir: str = "data", force_reprocess: bool = False, skip_sources: bool = False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.force_reprocess = force_reprocess
        self.skip_sources = skip_sources
        
        # Initialize components
        self.scraper = YouTubeScraper(
            output_dir=str(self.base_dir / "raw_videos")
        )
        
        # Initialize VideoLabeler with Whisper transcription
        # For better username detection, use OpenAI GPT-4o or Anthropic Claude
        # Change vlm_provider to "openai" or "anthropic" for better results
        config = LabelerConfig(
            frame_interval=30,
            use_whisper=True,
            whisper_model="base",
            use_speaker_diarization=True,
            vlm_provider="local",
            vlm_model="AudioVisual-Caption/ASID-Captioner-3B",
            extract_ui=False,
            asid_max_new_tokens=320,
        )
        self.labeler = VideoLabeler(config)
        
        # Disable implicit source detection (local LLM has compatibility issues)
        # Explicit sources (URLs, on-screen text) still work
        self.detective = SourceDetective(
            provider="none",
            newsapi_key=os.getenv("NEWS_API_KEY"),
        )
        self.source_scraper = None  # Lazy load SourceScraper when needed
        self.dataset_builder = DatasetBuilder(
            output_dir=str(self.base_dir / "training_dataset")
        )
        

    
    def _ensure_labeler(self):
        """VideoLabeler is now initialized in __init__."""
        pass  # Already initialized
    
    def _ensure_source_scraper(self):
        """Lazy load SourceScraper when needed."""
        if self.source_scraper is None:
            try:
                from src.data_collection.source_scraper import SourceScraper
                self.source_scraper = SourceScraper(
                    output_dir=str(self.base_dir / "sources")
                )
            except ImportError as e:
                print(f"[WARNING] SourceScraper unavailable (newspaper3k compatibility issue): {e}")
                print("    Source URLs will be saved but content won't be scraped.")
                self.source_scraper = None

    def _extract_video_id_from_url(self, video_url: str) -> Optional[str]:
        """Extract a YouTube video ID from common URL formats."""
        if not video_url:
            return None

        short_match = re.search(r"/shorts/([^/?&#]+)", video_url)
        if short_match:
            return short_match.group(1)

        watch_match = re.search(r"[?&]v=([^&#]+)", video_url)
        if watch_match:
            return watch_match.group(1)

        youtu_match = re.search(r"youtu\.be/([^/?&#]+)", video_url)
        if youtu_match:
            return youtu_match.group(1)

        return None

    def _is_video_already_processed(self, video_id: Optional[str]) -> bool:
        """Return True when the final per-video training example already exists."""
        if self.force_reprocess:
            return False
        if not video_id:
            return False
        example_path = self.base_dir / "training_dataset" / f"{video_id}_example.json"
        return example_path.exists()
        
    def process_video(self, video_url: str) -> str:
        """Process a single video through the entire pipeline."""

        requested_video_id = self._extract_video_id_from_url(video_url)
        if self._is_video_already_processed(requested_video_id):
            print(f"\n[SKIP] Already processed video {requested_video_id}; leaving existing JSON files unchanged")
            return requested_video_id
        
        print(f"\n{'='*70}")
        print(f"PROCESSING VIDEO: {video_url}")
        print(f"{'='*70}\n")
        
        # Step 1: Download video and metadata
        print("[DOWNLOAD] Step 1/5: Downloading video and metadata...")
        print("   (Testing multiple client strategies to bypass restrictions)")
        metadata = self.scraper.scrape_video(video_url, download_video=True)
        if not metadata:
            print("[ERROR] Failed to download video")
            return None
        
        video_id = metadata['video_id']
        print(f"\n[OK] Downloaded: {metadata['title']}")
        print(f"   Duration: {metadata.get('duration', 0)}s")
        print(f"   Channel: {metadata.get('channel', 'N/A')}")
        
        # Check if video file exists (would need manual download)
        video_path = self.base_dir / "raw_videos" / f"{video_id}.mp4"
        if not video_path.exists():
            # Try to find the video file (might have different extension)
            video_files = list((self.base_dir / "raw_videos").glob(f"{video_id}.*"))
            video_files = [f for f in video_files if f.suffix in ['.mp4', '.webm', '.mkv', '.json']]
            video_files = [f for f in video_files if f.suffix != '.json']  # Exclude metadata files
            if video_files:
                video_path = video_files[0]
            else:
                video_path = None
                print(f"[WARNING] Video file not available (YouTube blocked download)")
                print(f"   Continuing with metadata-only mode...")
        
        # Step 2: Analyze video (Whisper + VLM)
        transcript_text = ""
        transcript_segments = []
        labels = []
        
        if video_path and video_path.exists():
            print("\n[ANALYZE] Step 2/5: Analyzing video (Whisper + VLM)...")
            
            output_dir = self.base_dir / "analysis" / video_id
            frames_path = output_dir / "frames_analysis.json"

            # Skip re-analysis if already done
            if frames_path.exists():
                print(f"[SKIP] Already analyzed — loading cached frames_analysis.json")
                transcript_text = ""
                transcript_segments = []
                transcript_file = output_dir / "transcript.json"
                if transcript_file.exists():
                    import json as _json
                    with open(transcript_file) as f:
                        td = _json.load(f)
                    transcript_text = td.get("full_transcript", "")
                    transcript_segments = td.get("segments", []) if isinstance(td, dict) else []

                    # Backfill speaker labels when diarization is now enabled but
                    # cached transcript was generated before this feature existed.
                    if transcript_segments and self.labeler.config.use_speaker_diarization:
                        has_speaker_tags = any(
                            (isinstance(seg, dict) and (seg.get("speaker") or "[SPEAKER_" in seg.get("text", "")))
                            for seg in transcript_segments
                        )
                        if not has_speaker_tags:
                            try:
                                print("[INFO] Rebuilding cached transcript with speaker diarization...")
                                rebuilt_segments, rebuilt_text = self.labeler.transcript_extractor.extract(
                                    str(video_path),
                                    transcript_file,
                                )
                                transcript_segments = [
                                    s.to_dict() if hasattr(s, "to_dict") else s
                                    for s in rebuilt_segments
                                ]
                                transcript_text = rebuilt_text or transcript_text
                            except Exception as e:
                                print(f"[WARNING] Could not rebuild transcript diarization: {e}")
                analysis = None  # not needed; frames_path branch below handles labels
            else:
                # Use the full video labeler package
                analysis = self.labeler.analyze_video(str(video_path), output_dir=str(output_dir))
                
                # Extract transcript
                transcript_text = analysis.full_transcript
                transcript_segments = [
                    s.to_dict() if hasattr(s, 'to_dict') else s
                    for s in analysis.transcript_segments
                ]
                print(f"[OK] Transcribed: {len(transcript_text)} chars")
            
            # Also check for pre-existing labels from data/labels/ (may have better VLM descriptions)
            labels_path = self.base_dir / "labels" / f"{video_id}.json"
            
            if labels_path.exists():
                # Use pre-existing labels with better descriptions
                print(f"[OK] Found existing labels file: {labels_path}")
                with open(labels_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                    # Ensure each label has required fields
                    for label in labels:
                        if 'text_visible' not in label:
                            label['text_visible'] = ''
                        if 'ui_elements' not in label:
                            label['ui_elements'] = []
                print(f"[OK] Loaded {len(labels)} frame descriptions from labels file")
            elif frames_path.exists():
                with open(frames_path, 'r', encoding='utf-8') as f:
                    frames_data = json.load(f)
                    labels = [
                        {
                            'timestamp': frame['timestamp'],
                            # Propagate window bounds saved by segment_based_analysis so that
                            # dataset_builder can create properly-bounded clips instead of
                            # zero-duration point entries that would otherwise be skipped.
                            'start_time': frame.get('segment_start', frame['timestamp']),
                            'end_time': frame.get('segment_end', frame['timestamp']),
                            'description': frame['description'],
                            'frame_number': frame.get('frame_number', 0),
                            'text_visible': frame.get('text_visible', ''),
                            'ui_elements': frame.get('ui_elements', []),
                            'ui_html': frame.get('ui_html', ''),
                        }
                        for frame in frames_data
                    ]
                print(f"[OK] Loaded {len(labels)} frame descriptions")
            else:
                # Fallback: extract from segments if available
                labels = []
                for segment in analysis.segments:
                    for frame in segment.key_frames:
                        labels.append({
                            'timestamp': frame.timestamp,
                            'description': frame.description,
                            'frame_number': 0,
                            'text_visible': frame.text_visible or '',
                            'ui_elements': frame.ui_elements or []  # Include UI elements
                        })
                print(f"[OK] Extracted {len(labels)} frames from segments")
            
            # Save analysis only when this run produced a fresh analysis object.
            if analysis is not None:
                analysis_path = self.base_dir / "analysis" / f"{video_id}_analysis.json"
                analysis_path.parent.mkdir(parents=True, exist_ok=True)
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis.to_dict(), f, indent=2)
        else:
            print("\n[SKIP] Step 2/5: Skipping video analysis (no video file)")
            # Fall back to yt-dlp transcript if available
            transcript_text = metadata.get('transcript', '')
        
        # Optionally run ASID captioner to get audiovisual summary (runs if video file exists)
        # TEMPORARILY DISABLED: ASID causing hangs, skipping to allow audio captioning
        if False and video_path and video_path.exists():
            try:
                from asid_captioner import ASIDCaptioner
                print("\n[ASID] Step 2.1/5: Running audiovisual captioner...")
                asid = ASIDCaptioner()
                asid_desc = asid.describe_video(str(video_path))
                if asid_desc:
                    metadata['asid_description'] = asid_desc
                    print(f"[ASID] Description: {asid_desc[:80]}...")
                    # Save updated metadata with ASID description
                    metadata_path = self.base_dir / "raw_videos" / f"{video_id}.json"
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ASID] Failed: {e}")
                print("[ASID] Continuing without audiovisual description...")
        
        # Run audio captioner to describe music, sound effects, and audio content
        if video_path and video_path.exists():
            try:
                from audio_captioner import AudioCaptioner
                print("\n[AUDIO] Step 2.2/5: Running audio captioner for sound/music description...")
                audio_cap = AudioCaptioner()

                # full-video description (legacy)
                audio_desc = audio_cap.describe_video_audio(str(video_path))
                if audio_desc:
                    metadata['audio_description'] = audio_desc
                    print(f"[AUDIO] Video-level description: {audio_desc[:80]}...")

                # compute segments now so we can run audio captioner per-segment
                segments_for_audio = self.dataset_builder._create_segments(
                    transcript_segments,
                    labels,
                    []  # sources not needed for segmentation
                )
                seg_audio_descs = []
                for seg in segments_for_audio:
                    start = seg.get('start_time', 0)
                    end = seg.get('end_time', 0)
                    desc = audio_cap.describe_video_audio_segment(
                        str(video_path), start, end
                    )
                    seg_audio_descs.append(desc)
                    print(f"[AUDIO] Segment {start}-{end}s: {desc[:60]}...")
                if seg_audio_descs:
                    metadata['segment_audio_descriptions'] = seg_audio_descs

                # Save updated metadata with audio description
                if audio_desc or seg_audio_descs:
                    metadata_path = self.base_dir / "raw_videos" / f"{video_id}.json"
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
            except ImportError:
                pass
        
        # Step 3: Detect sources (use Whisper transcript if available)
        if self.skip_sources:
            print("\n[SOURCES] Step 3/5: Skipping source detection (--no-sources)")
            sources = []
            scraped_sources = []
        else:
            print("\n[SOURCES] Step 3/5: Detecting sources...")

            # Combine title + description for better source detection
            title_and_desc = f"{metadata.get('title', '')} {metadata.get('description', '')}".strip()

            sources = self.detective.detect_sources(
                transcript=transcript_text if transcript_text else metadata.get('transcript', ''),
                description=title_and_desc,
                labels=labels,
                upload_date=metadata.get('upload_date'),
                topic=metadata.get('topic', ''),
            )

            print(f"[OK] Found {len(sources)} potential sources")
            for i, source in enumerate(sources[:5]):  # Show first 5
                print(f"  {i+1}. [{source['type']}] {source.get('content', '')[:60]}... (conf: {source['confidence']:.2f})")

            # Step 4: Scrape source content
            print("\n[SCRAPE] Step 4/5: Scraping source content...")
            self._ensure_source_scraper()

            if self.source_scraper:
                scraped_sources = self.source_scraper.scrape_all_sources(sources)
                self.source_scraper.save_sources(scraped_sources, video_id)
                print(f"[OK] Scraped {len(scraped_sources)} sources")
            else:
                # Scrape article bodies for news/url sources using newspaper3k directly
                scraped_sources = list(sources)
                for s in scraped_sources:
                    if s.get('type') in ('news', 'url') and not s.get('text'):
                        url = s.get('url') or s.get('content', '')
                        if url.startswith('http'):
                            try:
                                from newspaper import Article as _Article
                                art = _Article(url)
                                art.download()
                                art.parse()
                                if art.text:
                                    s['text'] = art.text[:3000]
                                    s['success'] = True
                            except Exception:
                                pass
                sources_path = self.base_dir / "sources" / f"{video_id}_sources.json"
                sources_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sources_path, 'w', encoding='utf-8') as f:
                    json.dump(scraped_sources, f, indent=2, ensure_ascii=False)
                scraped_with_text = sum(1 for s in scraped_sources if s.get('text'))
                print(f"[OK] {scraped_with_text}/{len(scraped_sources)} sources scraped directly")
        
        # Add Whisper transcript to metadata
        if transcript_text:
            metadata['transcript'] = transcript_text
        if transcript_segments:
            metadata['transcript_segments'] = transcript_segments
        
        # Step 5: Build training example
        print("\n[BUILD] Step 5/5: Building training example...")
        training_example = self.dataset_builder.build_training_example(
            metadata, labels, scraped_sources
        )
        
        # Save individual example
        example_path = self.base_dir / "training_dataset" / f"{video_id}_example.json"
        example_path.parent.mkdir(exist_ok=True)
        
        with open(example_path, 'w', encoding='utf-8') as f:
            json.dump(training_example, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Training example saved: {example_path}")
        
        # Show helpful tip if video wasn't downloaded
        if not video_path or not video_path.exists():
            print(f"\n[TIP] TIP: To add video labeling later:")
            print(f"   1. Manually download the video (use browser extension/tool)")
            print(f"   2. Save as: {self.base_dir / 'raw_videos' / f'{video_id}.mp4'}")
            print(f"   3. Re-run this command to add visual labels")
        
        print(f"\n{'='*70}")
        print(f"[OK] COMPLETED: {video_id}")
        print(f"{'='*70}\n")
        
        return video_id
    
    def process_channel(self, channel_url: str, max_videos: int = 50) -> List[str]:
        """Process multiple videos from a channel."""
        
        print(f"\n{'='*70}")
        print(f"PROCESSING CHANNEL: {channel_url}")
        print(f"Target: {max_videos} videos")
        print(f"{'='*70}\n")
        
        # Download all videos first
        print("📥 Downloading videos from channel...")
        videos = self.scraper.scrape_channel(channel_url, max_videos=max_videos)
        
        print(f"\n✅ Downloaded {len(videos)} videos\n")
        
        # Process each video
        processed_ids = []
        
        for i, video_metadata in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] Processing video...")
            
            video_id = video_metadata['video_id']
            video_url = video_metadata.get('webpage_url') or (
                f"https://www.youtube.com/watch?v={video_id}" if video_id else None
            )

            if not video_id:
                print(f"[SKIP] No video_id in metadata, skipping entry")
                continue

            if not video_url:
                print(f"[SKIP] No URL for {video_id}, skipping")
                continue

            if self._is_video_already_processed(video_id):
                print(f"[SKIP] Already processed: {video_id} (keeping existing JSON files)")
                processed_ids.append(video_id)
                continue
            
            try:
                result = self.process_video(video_url)
                if result:
                    processed_ids.append(result)
            except Exception as e:
                print(f"❌ Error processing {video_id}: {e}")
                continue
        
        # Build final dataset
        print("\n🎯 Building final dataset...")
        dataset = self.dataset_builder.build_dataset()
        self.dataset_builder.save_dataset(dataset, "youtuber_dataset.json")
        
        print(f"\n{'='*70}")
        print(f"✅ CHANNEL PROCESSING COMPLETE")
        print(f"Processed: {len(processed_ids)}/{len(videos)} videos")
        print(f"{'='*70}\n")
        
        return processed_ids


def main():
    parser = argparse.ArgumentParser(description="YouTuber Training Data Collection Pipeline")
    parser.add_argument("--video", type=str, help="Single video URL to process")
    parser.add_argument("--channel", type=str, nargs="+", metavar="URL",
                        help="One or more channel URLs to process")
    parser.add_argument("--channels-file", type=str, metavar="FILE",
                        help="Path to a text file with one channel URL per line (# lines are ignored)")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos per channel")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Reprocess videos even when *_example.json already exists")
    parser.add_argument("--no-sources", action="store_true",
                        help="Skip source detection and scraping (steps 3 and 4)")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        base_dir=args.data_dir,
        force_reprocess=args.force_reprocess,
        skip_sources=args.no_sources,
    )

    if args.video:
        orchestrator.process_video(args.video)
        return

    # Collect (url, limit) pairs from --channel and/or --channels-file.
    # channels-file format per line:  <url> [limit]   (# = comment)
    channel_entries: List[tuple] = []  # (url: str, limit: int | None)

    for url in (args.channel or []):
        channel_entries.append((url, None))

    if args.channels_file:
        channels_path = Path(args.channels_file)
        if not channels_path.exists():
            print(f"[ERROR] Channels file not found: {channels_path}")
            return
        for raw in channels_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            url = parts[0]
            limit: Optional[int] = None
            if len(parts) >= 2:
                try:
                    limit = int(parts[1])
                except ValueError:
                    print(f"[WARNING] Ignoring non-integer limit {parts[1]!r} for {url}")
            channel_entries.append((url, limit))

    if not channel_entries:
        print("Usage:")
        print("  Single video:    python pipeline_orchestrator.py --video <URL>")
        print("  One channel:     python pipeline_orchestrator.py --channel <URL> --max-videos 50")
        print("  Many channels:   python pipeline_orchestrator.py --channel <URL1> <URL2> ...")
        print("  From file:       python pipeline_orchestrator.py --channels-file channels.txt")
        print("")
        print("channels.txt format (per line):  <url> [max_videos]")
        print("  https://www.youtube.com/@channel/shorts 100")
        print("  https://www.youtube.com/@other/shorts       # uses --max-videos default")
        return

    # Deduplicate by URL while preserving order; first occurrence wins for per-channel limit
    seen: set = set()
    unique_channels: List[tuple] = []
    for url, limit in channel_entries:
        if url not in seen:
            seen.add(url)
            unique_channels.append((url, limit))

    print(f"\nProcessing {len(unique_channels)} channel(s)...")
    all_processed: List[str] = []
    for idx, (channel_url, channel_limit) in enumerate(unique_channels, 1):
        effective_limit = channel_limit if channel_limit is not None else args.max_videos
        print(f"\n[Channel {idx}/{len(unique_channels)}] {channel_url}  (max {effective_limit} videos)")
        processed = orchestrator.process_channel(channel_url, max_videos=effective_limit)
        all_processed.extend(processed)

    print(f"\n{'='*70}")
    print(f"ALL CHANNELS DONE — {len(all_processed)} total video(s) processed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
