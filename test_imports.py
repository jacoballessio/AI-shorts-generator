"""
Test imports to verify all modules load correctly
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing imports...")

try:
    from research_agent import ResearchAgent
    print("✓ ResearchAgent imported successfully")
except Exception as e:
    print(f"✗ ResearchAgent import failed: {e}")

try:
    from content_creator import ContentCreator
    print("✓ ContentCreator imported successfully")
except Exception as e:
    print(f"✗ ContentCreator import failed: {e}")

try:
    from media_generator import MediaGenerator
    print("✓ MediaGenerator imported successfully")
except Exception as e:
    print(f"✗ MediaGenerator import failed: {e}")

try:
    from video_composer import VideoComposer
    print("✓ VideoComposer imported successfully")
except Exception as e:
    print(f"✗ VideoComposer import failed: {e}")

try:
    from main import ShortVideoOrchestrator
    print("✓ ShortVideoOrchestrator imported successfully")
except Exception as e:
    print(f"✗ ShortVideoOrchestrator import failed: {e}")

print("\n✅ All core modules loaded successfully!")
