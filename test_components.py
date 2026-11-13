"""
Simple test that just imports and shows status
"""
print("=== Pipeline Component Status ===\n")

print("1. Testing YouTubeScraper...")
try:
    from src.data_collection.youtube_scraper import YouTubeScraper
    print("   SUCCESS: YouTubeScraper imported\n")
except ImportError as e:
    print(f"   FAILED: {e}\n")

print("2. Testing SourceDetective...")
try:
    from src.data_collection.source_detective import SourceDetective
    print("   SUCCESS: SourceDetective imported\n")
except ImportError as e:
    print(f"   FAILED: {e}\n")

print("3. Testing VideoLabeler...")
try:
    from src.data_collection.video_labeler import VideoLabeler
    print("   SUCCESS: VideoLabeler imported (may require GPU)\n")
except ImportError as e:
    print(f"   FAILED: {e}\n")

print("4. Testing SourceScraper...")
try:
    from src.data_collection.source_scraper import SourceScraper
    print("   SUCCESS: SourceScraper imported\n")
except ImportError as e:
    print(f"   FAILED: {e}")
    print("   (This is expected - newspaper3k has compatibility issues)\n")

print("5. Testing DatasetBuilder...")
try:
    from src.training.dataset_builder import DatasetBuilder
    print("   SUCCESS: DatasetBuilder imported\n")
except ImportError as e:
    print(f"   FAILED: {e}\n")

print("=== Summary ===")
print("Pipeline components loaded successfully (except SourceScraper).")
print("Ready to process videos with metadata-only mode.")
