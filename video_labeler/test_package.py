"""
Quick test to verify video_labeler package structure
"""

import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing video_labeler package imports...")
    
    try:
        from video_labeler import VideoLabeler, LabelerConfig, VideoAnalysis, VideoSegment
        print("✅ Main imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from video_labeler.config import LabelerConfig
        print("✅ Config import successful")
    except ImportError as e:
        print(f"❌ Config import error: {e}")
        return False
    
    try:
        from video_labeler.models import (
            VideoAnalysis, VideoSegment, FrameAnalysis, TranscriptSegment
        )
        print("✅ Models import successful")
    except ImportError as e:
        print(f"❌ Models import error: {e}")
        return False
    
    try:
        from video_labeler.frame_extractor import FrameExtractor
        from video_labeler.transcript_extractor import TranscriptExtractor
        from video_labeler.vlm_analyzer import VLMAnalyzer
        from video_labeler.segmentation import SegmentDetector
        print("✅ All component imports successful")
    except ImportError as e:
        print(f"❌ Component import error: {e}")
        return False
    
    print("\n✅ All imports successful! Package structure is correct.")
    return True


def test_config_creation():
    """Test creating a config object."""
    print("\nTesting config creation...")
    
    try:
        from video_labeler import LabelerConfig
        
        # Test default config
        config = LabelerConfig()
        print(f"✅ Default config created")
        print(f"   Frame interval: {config.frame_interval}")
        print(f"   VLM provider: {config.vlm_provider}")
        print(f"   VLM model: {config.vlm_model}")
        
        # Test custom config
        custom_config = LabelerConfig(
            frame_interval=60,
            vlm_provider="anthropic",
            vlm_model="claude-3-5-sonnet-20241022",
            use_whisper=False
        )
        print(f"✅ Custom config created")
        print(f"   Frame interval: {custom_config.frame_interval}")
        print(f"   VLM provider: {custom_config.vlm_provider}")
        
        return True
    except Exception as e:
        print(f"❌ Config creation error: {e}")
        return False


def test_model_serialization():
    """Test model to_dict and JSON serialization."""
    print("\nTesting model serialization...")
    
    try:
        from video_labeler.models import VideoSegment, FrameAnalysis
        
        # Create a frame analysis
        frame = FrameAnalysis(
            frame_number=100,
            timestamp=3.33,
            description="Person walking in park",
            objects=["person", "tree", "path"],
            actions=["walking"],
            scene_type="outdoor",
            mood="peaceful"
        )
        
        frame_dict = frame.to_dict()
        print(f"✅ FrameAnalysis serialization successful")
        print(f"   Keys: {list(frame_dict.keys())}")
        
        # Create a segment
        segment = VideoSegment(
            segment_id=0,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            description="Opening scene",
            transcript="Hello and welcome",
            keywords=["intro", "welcome"]
        )
        
        segment_dict = segment.to_dict()
        print(f"✅ VideoSegment serialization successful")
        print(f"   Keys: {list(segment_dict.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Serialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_package_info():
    """Show package information."""
    print("\n" + "="*70)
    print("VIDEO LABELER PACKAGE INFO")
    print("="*70)
    
    try:
        from video_labeler import __version__
        print(f"Version: {__version__}")
    except:
        print("Version: Not set")
    
    print("\nKey Components:")
    print("  - VideoLabeler: Main analysis class")
    print("  - LabelerConfig: Configuration")
    print("  - VideoAnalysis: Complete analysis results")
    print("  - VideoSegment: Individual segment data")
    print("  - FrameAnalysis: Frame-level analysis")
    
    print("\nSupported VLM Providers:")
    print("  - openai (GPT-4 Vision)")
    print("  - anthropic (Claude Vision)")
    print("  - local (Coming soon)")
    
    print("\nFeatures:")
    print("  ✓ Frame extraction and analysis")
    print("  ✓ Whisper transcription")
    print("  ✓ Automatic segmentation")
    print("  ✓ Keyword generation")
    print("  ✓ Quality scoring")
    print("  ✓ JSON serialization")
    print("  ✓ Content search")


if __name__ == "__main__":
    print("="*70)
    print("VIDEO LABELER PACKAGE TEST")
    print("="*70)
    print()
    
    # Run tests
    imports_ok = test_imports()
    config_ok = test_config_creation()
    models_ok = test_model_serialization()
    
    # Show info
    if imports_ok and config_ok and models_ok:
        show_package_info()
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - Package is ready to use!")
        print("="*70)
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in environment")
        print("2. Install openai-whisper: pip install openai-whisper")
        print("3. See examples.py for usage examples")
        print("4. Read README.md for full documentation")
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED - Check errors above")
        print("="*70)
