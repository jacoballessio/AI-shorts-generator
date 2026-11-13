"""
Quick test of video_labeler with example.mp4

This is a FAST version that analyzes just a few frames for quick testing.
For full analysis, use test_example_video.py
"""

import os
import sys
from pathlib import Path

# Add video_labeler to path
sys.path.insert(0, str(Path(__file__).parent / "video_labeler"))

from video_labeler import VideoLabeler, LabelerConfig


def main():
    print("=" * 70)
    print("Quick Video Labeler Test - example.mp4")
    print("=" * 70)
    print()
    
    # Check if example video exists
    video_path = "example.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Error: {video_path} not found!")
        return
    
    print(f"✅ Found video: {video_path}")
    print()
    
    # Configure for FAST testing
    print("Configuring for quick test...")
    config = LabelerConfig(
        # Use local SmolVLM
        vlm_provider="local",
        vlm_model="HuggingFaceTB/SmolVLM-Instruct",
        
        # FAST settings
        frame_interval=180,  # Every 6 seconds - wider gaps for even faster testing
        max_frames=10,       # Only analyze 10 frames
        
        # Skip transcription for speed
        use_whisper=False,
        
        # Minimal analysis
        analyze_actions=False,
        analyze_objects=True,
        analyze_scenes=True,
        generate_summary=True,
        generate_keywords=True,
        auto_segment=False,  # Skip segmentation
        
        batch_size=5
    )
    
    print(f"✅ Quick test configuration:")
    print(f"   Max frames: {config.max_frames}")
    print(f"   Whisper: disabled (for speed)")
    print(f"   Segmentation: disabled (for speed)")
    print()
    
    # Create labeler
    print("Initializing labeler...")
    labeler = VideoLabeler(config)
    print("✅ Ready!")
    print()
    
    # Analyze video
    print("Analyzing video (this will be quick)...")
    print()
    
    try:
        analysis = labeler.analyze_video(video_path)
        
        print()
        print("=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print()
        
        # Display results
        print(f"📊 Video: {analysis.video_path}")
        print(f"   Duration: {analysis.duration:.1f}s")
        print(f"   Frames analyzed: {analysis.total_frames_analyzed}")
        print(f"   Analysis time: {analysis.analysis_duration:.1f}s")
        print()
        
        if analysis.summary:
            print("📝 Summary:")
            print(f"   {analysis.summary}")
            print()
        
        if analysis.global_keywords:
            print(f"🔑 Keywords: {', '.join(analysis.global_keywords[:10])}")
            print()
        
        # Show frame analyses
        print("🖼️  Frame Analyses:")
        print()
        for i, seg in enumerate(analysis.segments[:5], 1):
            if seg.key_frames:
                frame = seg.key_frames[0]
                print(f"{i}. Frame at {frame.timestamp:.1f}s:")
                print(f"   {frame.description}")
                if frame.objects:
                    print(f"   Objects: {', '.join(frame.objects[:3])}")
                print()
        
        # Save results
        json_path = "quick_test_analysis.json"
        analysis.to_json(json_path)
        print(f"💾 Saved to: {json_path}")
        print()
        
        print("✅ Quick test passed!")
        print()
        print("For full analysis with transcription and segmentation,")
        print("run: python test_example_video.py")
        print()
        
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
