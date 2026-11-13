"""
Test video_labeler with example.mp4

This script demonstrates all features of the video_labeler package:
- Frame extraction and VLM analysis (using local SmolVLM)
- Audio transcription with Whisper
- Automatic segmentation
- Keyword extraction
- Quality scoring
- JSON serialization
"""

import os
import sys
from pathlib import Path

# Add video_labeler to path
sys.path.insert(0, str(Path(__file__).parent / "video_labeler"))

from video_labeler import VideoLabeler, LabelerConfig


def main():
    print("=" * 70)
    print("Video Labeler Test - example.mp4")
    print("=" * 70)
    print()
    
    # Check if example video exists
    video_path = "example.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Error: {video_path} not found!")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    print(f"✅ Found video: {video_path}")
    print()
    
    # Configure for detailed analysis
    print("Configuring video labeler...")
    config = LabelerConfig(
        # Use local SmolVLM (free!)
        vlm_provider="local",
        vlm_model="HuggingFaceTB/SmolVLM-Instruct",
        
        # Frame sampling
        frame_interval=120,  # Analyze every 4 seconds (at 30fps) - wider gaps for faster testing
        max_frames=None,     # Analyze entire video
        
        # Transcription
        use_whisper=True,
        whisper_model="base",  # fast and accurate enough
        
        # Analysis features
        analyze_actions=True,
        analyze_objects=True,
        analyze_scenes=True,
        generate_summary=True,
        generate_keywords=True,
        
        # Segmentation
        auto_segment=True,
        min_segment_duration=2.0,
        
        # Performance
        batch_size=5,  # Process 5 frames in parallel
        include_frame_images=False  # Don't save frame images to save space
    )
    
    print(f"✅ Configuration:")
    print(f"   VLM: {config.vlm_provider} ({config.vlm_model})")
    print(f"   Whisper: {config.whisper_model if config.use_whisper else 'disabled'}")
    print(f"   Frame interval: every {config.frame_interval} frames")
    print()
    
    # Create labeler
    print("Initializing video labeler...")
    print("(This may take a minute to load SmolVLM on first run)")
    print()
    labeler = VideoLabeler(config)
    print("✅ Labeler ready!")
    print()
    
    # Analyze video
    print("-" * 70)
    print("Analyzing video...")
    print("-" * 70)
    print()
    
    try:
        analysis = labeler.analyze_video(
            video_path,
            output_dir="./analysis_cache"  # Cache results here
        )
        
        print()
        print("=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print()
        
        # Display results
        print("📊 VIDEO INFO")
        print("-" * 70)
        print(f"Path: {analysis.video_path}")
        print(f"Duration: {analysis.duration:.1f} seconds")
        print(f"FPS: {analysis.fps}")
        print(f"Frames analyzed: {analysis.total_frames_analyzed}")
        print(f"Analysis time: {analysis.analysis_duration:.1f}s")
        print()
        
        # Summary
        if analysis.summary:
            print("📝 SUMMARY")
            print("-" * 70)
            print(analysis.summary)
            print()
        
        # Global keywords
        if analysis.global_keywords:
            print("🔑 KEYWORDS")
            print("-" * 70)
            print(", ".join(analysis.global_keywords[:20]))
            print()
        
        # Transcript
        if analysis.full_transcript:
            print("🎤 FULL TRANSCRIPT")
            print("-" * 70)
            print(analysis.full_transcript[:500])
            if len(analysis.full_transcript) > 500:
                print(f"... ({len(analysis.full_transcript) - 500} more characters)")
            print()
        
        # Segments
        print("✂️  SEGMENTS")
        print("-" * 70)
        print(f"Total segments: {len(analysis.segments)}")
        print()
        
        for i, segment in enumerate(analysis.segments, 1):
            print(f"Segment {i}: {segment.start_time:.1f}s - {segment.end_time:.1f}s ({segment.duration:.1f}s)")
            print(f"   Description: {segment.description}")
            
            if segment.transcript:
                transcript_preview = segment.transcript[:100]
                if len(segment.transcript) > 100:
                    transcript_preview += "..."
                print(f"   Transcript: {transcript_preview}")
            
            if segment.keywords:
                print(f"   Keywords: {', '.join(segment.keywords[:5])}")
            
            if segment.dominant_objects:
                print(f"   Objects: {', '.join(segment.dominant_objects[:3])}")
            
            if segment.actions_sequence:
                print(f"   Actions: {', '.join(segment.actions_sequence[:3])}")
            
            print(f"   Quality score: {segment.quality_score:.2f}")
            print(f"   Clip-worthy: {'✅ Yes' if segment.clip_worthy else '❌ No'}")
            print()
        
        # Find clip-worthy segments
        print("🎬 CLIP-WORTHY SEGMENTS")
        print("-" * 70)
        clips = analysis.get_clip_worthy_segments(min_score=0.7)
        print(f"Found {len(clips)} clip-worthy segments (score >= 0.7):")
        print()
        
        for i, clip in enumerate(clips[:5], 1):  # Show top 5
            print(f"{i}. {clip.start_time:.1f}s - {clip.end_time:.1f}s (score: {clip.quality_score:.2f})")
            print(f"   {clip.description}")
            print()
        
        # Save to JSON
        print("💾 SAVING ANALYSIS")
        print("-" * 70)
        json_path = "example_analysis.json"
        analysis.to_json(json_path)
        print(f"✅ Saved to: {json_path}")
        print(f"   File size: {os.path.getsize(json_path) / 1024:.1f} KB")
        print()
        
        # Test loading from JSON
        print("📂 TESTING JSON LOAD")
        print("-" * 70)
        loaded_analysis = analysis.__class__.from_json(json_path)
        print(f"✅ Loaded {len(loaded_analysis.segments)} segments from JSON")
        print()
        
        # Search example
        if analysis.segments:
            print("🔍 SEARCH EXAMPLE")
            print("-" * 70)
            
            # Try searching for common keywords
            search_terms = ["person", "talking", "face", "speaking", "video"]
            for term in search_terms:
                results = analysis.find_segments_by_keyword(term)
                if results:
                    print(f"Search '{term}': {len(results)} segments found")
                    print(f"   First match: {results[0].start_time:.1f}s - {results[0].description[:60]}...")
                    break
            print()
        
        # Success!
        print("=" * 70)
        print("✅ TEST COMPLETE!")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  • Analyzed {analysis.duration:.1f}s of video")
        print(f"  • Found {len(analysis.segments)} segments")
        print(f"  • Extracted {len(analysis.full_transcript.split()) if analysis.full_transcript else 0} words of transcript")
        print(f"  • Generated {len(analysis.global_keywords)} keywords")
        print(f"  • Identified {len(clips)} clip-worthy moments")
        print()
        print(f"Results saved to: {json_path}")
        print()
        print("Next steps:")
        print("  • Review the segments and quality scores")
        print("  • Use analysis.get_clip_worthy_segments() to extract best clips")
        print("  • Integrate with AI shorts generator for intelligent clip matching")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR DURING ANALYSIS")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
