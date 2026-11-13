"""
Example usage of the Video Labeler package

All examples use LOCAL SmolVLM by default (100% free, no API costs!)
"""

from video_labeler import VideoLabeler, LabelerConfig

# Example 1: Basic usage with local SmolVLM (FREE!)
def basic_example():
    """Simple analysis with local VLM - completely free!"""
    
    print("="*70)
    print("EXAMPLE 1: Basic Usage with Local SmolVLM (FREE!)")
    print("="*70)
    
    # Create labeler with defaults (uses local SmolVLM)
    # No API keys needed!
    labeler = VideoLabeler()
    
    # Analyze a video (first run downloads SmolVLM ~2GB)
    analysis = labeler.analyze_video("path/to/video.mp4")
    
    # Print results
    print(f"\nVideo Summary: {analysis.summary}")
    print(f"Total Segments: {len(analysis.segments)}")
    print(f"Duration: {analysis.duration:.1f}s")
    
    # Show first few segments
    for seg in analysis.segments[:3]:
        print(f"\nSegment {seg.segment_id} ({seg.start_time:.1f}s - {seg.end_time:.1f}s):")
        print(f"  Description: {seg.description}")
        print(f"  Transcript: {seg.transcript[:100]}...")
        print(f"  Keywords: {', '.join(seg.keywords[:5])}")


# Example 2: Optimized local configuration
def custom_config_example():
    """Analysis with optimized local settings for best performance."""
    
    print("="*70)
    print("EXAMPLE 2: Optimized Local Configuration")
    print("="*70)
    
    # Configure for high-detail local analysis
    config = LabelerConfig(
        frame_interval=15,  # Analyze every 15 frames (0.5s at 30fps)
        vlm_provider="local",  # Local SmolVLM (default)
        vlm_model="HuggingFaceTB/SmolVLM-Instruct",
        whisper_model="medium",  # Better transcription (still local!)
        generate_summary=True,
        generate_keywords=True,
        include_frame_images=True,  # Save analyzed frames
        batch_size=10  # Process 10 frames at a time (adjust for your GPU)
    )
    
    labeler = VideoLabeler(config)
    analysis = labeler.analyze_video(
        "path/to/video.mp4",
        output_dir="./analysis_output"
    )
    
    # Save analysis to JSON
    analysis.to_json("video_analysis.json")
    
    print(f"\nAnalysis saved to video_analysis.json")
    print(f"Analyzed {analysis.total_frames_analyzed} frames in {analysis.analysis_duration:.1f}s")


# Example 3: Using cloud APIs (optional - costs money)
def cloud_api_example():
    """Using cloud APIs instead of local VLM (requires API keys)."""
    
    print("="*70)
    print("EXAMPLE 3: Cloud APIs (OpenAI/Anthropic)")
    print("="*70)
    print("⚠️  Note: This costs money! ~$0.50-$1.00 per video")
    print()
    
    # Option 1: OpenAI GPT-4V
    config_openai = LabelerConfig(
        vlm_provider="openai",
        vlm_model="gpt-4o",
        api_key="your-openai-key",  # or set OPENAI_API_KEY env var
        frame_interval=30
    )
    
    # Option 2: Anthropic Claude Vision  
    config_anthropic = LabelerConfig(
        vlm_provider="anthropic",
        vlm_model="claude-3-5-sonnet-20241022",
        api_key="your-anthropic-key",  # or set ANTHROPIC_API_KEY env var
        frame_interval=30
    )
    
    # Use whichever you prefer
    labeler = VideoLabeler(config_openai)
    analysis = labeler.analyze_video("path/to/video.mp4")
    
    print(f"Analyzed with {config_openai.vlm_model}")
    print(f"Summary: {analysis.summary}")


# Example 4: Find specific content
def search_example():
    """Search for specific content in a video."""
    
    print("="*70)
    print("EXAMPLE 4: Content Search")
    print("="*70)
    
    labeler = VideoLabeler()
    analysis = labeler.analyze_video("cooking_video.mp4")
    
    # Find segments about cooking
    cooking_segments = analysis.find_segments_by_keyword("cooking")
    print(f"\nFound {len(cooking_segments)} cooking segments:")
    for seg in cooking_segments:
        print(f"  {seg.start_time:.1f}s - {seg.end_time:.1f}s: {seg.description}")
    
    # Get clip-worthy segments
    clips = analysis.get_clip_worthy_segments(min_score=0.7)
    print(f"\nFound {len(clips)} clip-worthy segments:")
    for seg in clips:
        print(f"  {seg.start_time:.1f}s - {seg.end_time:.1f}s (score: {seg.quality_score:.2f})")
        print(f"    {seg.description}")


# Example 5: Extract specific clips
def clip_extraction_example():
    """Extract specific clips based on analysis."""
    
    print("="*70)
    print("EXAMPLE 5: Clip Extraction")
    print("="*70)
    
    from moviepy import VideoFileClip
    
    labeler = VideoLabeler()
    analysis = labeler.analyze_video("full_video.mp4")
    
    # Get best segments
    best_segments = analysis.get_clip_worthy_segments(min_score=0.8)
    
    # Extract clips
    video = VideoFileClip("full_video.mp4")
    
    for i, segment in enumerate(best_segments[:5]):  # Top 5 clips
        clip = video.subclipped(segment.start_time, segment.end_time)
        output_path = f"clip_{i+1}_{segment.segment_id}.mp4"
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"\nExtracted clip {i+1}:")
        print(f"  Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
        print(f"  Description: {segment.description}")
        print(f"  Saved to: {output_path}")
    
    video.close()


# Example 6: Load and reuse analysis
def load_analysis_example():
    """Load previously saved analysis."""
    
    print("="*70)
    print("EXAMPLE 6: Load Saved Analysis")
    print("="*70)
    
    # Load from JSON
    analysis = VideoAnalysis.from_json("video_analysis.json")
    
    # Use the loaded analysis
    print(f"Loaded analysis for: {analysis.video_path}")
    print(f"Segments: {len(analysis.segments)}")
    print(f"Keywords: {', '.join(analysis.global_keywords[:10])}")
    
    # Find segments by time range
    segments_in_range = analysis.get_segments_by_timerange(10.0, 30.0)
    print(f"\nSegments between 10s-30s: {len(segments_in_range)}")


# Example 7: Fast mode (fewer frames, no Whisper)
def fast_mode_example():
    """Quick analysis for preview."""
    
    print("="*70)
    print("EXAMPLE 6: Fast Mode")
    print("="*70)
    
    config = LabelerConfig(
        frame_interval=60,  # Every 60 frames (2s at 30fps)
        use_whisper=False,  # Skip transcription
        generate_summary=False,  # Skip summary
        max_frames=30,  # Analyze max 30 frames
        batch_size=10
    )
    
    labeler = VideoLabeler(config)
    analysis = labeler.analyze_video("long_video.mp4")
    
    print(f"\nQuick analysis completed in {analysis.analysis_duration:.1f}s")
    print(f"Segments detected: {len(analysis.segments)}")


if __name__ == "__main__":
    # Run examples
    print("Video Labeler Examples\n")
    
    # Uncomment to run specific examples:
    # basic_example()
    # custom_config_example()
    # search_example()
    # clip_extraction_example()
    # load_analysis_example()
    # fast_mode_example()
