"""
Test individual components of the pipeline.
Quick sanity checks before running the full pipeline.
"""
import sys


def test_youtube_scraper():
    """Test YouTube video downloading."""
    print("\n🧪 Testing YouTube Scraper...")
    
    from src.data_collection.youtube_scraper import YouTubeScraper
    
    scraper = YouTubeScraper(output_dir="data/test/raw_videos")
    
    # Test with a short video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Classic test video
    
    print(f"Downloading: {video_url}")
    metadata = scraper.scrape_video(video_url)
    
    if metadata:
        print(f"✅ SUCCESS!")
        print(f"   Title: {metadata['title']}")
        print(f"   Duration: {metadata['duration']}s")
        print(f"   Transcript length: {len(metadata.get('transcript', ''))} chars")
        return True
    else:
        print("❌ FAILED")
        return False


def test_video_labeler():
    """Test video frame labeling."""
    print("\n🧪 Testing Video Labeler...")
    
    from src.data_collection.video_labeler import VideoLabeler
    from pathlib import Path
    
    # Find a downloaded video
    video_dir = Path("data/test/raw_videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
    
    if not video_files:
        print("❌ No video files found. Run test_youtube_scraper first.")
        return False
    
    video_path = video_files[0]
    print(f"Labeling: {video_path.name}")
    
    labeler = VideoLabeler()
    
    # Sample just a few frames for testing
    labels = labeler.label_video(str(video_path), sample_rate=60)  # Every 2 seconds
    
    if labels:
        print(f"✅ SUCCESS!")
        print(f"   Labeled {len(labels)} frames")
        print(f"   First label: {labels[0]['description'][:80]}...")
        
        # Save
        labeler.save_labels(labels, "data/test/labels/test.json")
        return True
    else:
        print("❌ FAILED")
        return False


def test_source_detective():
    """Test source detection."""
    print("\n🧪 Testing Source Detective...")
    
    from src.data_collection.source_detective import SourceDetective
    
    detective = SourceDetective(provider="anthropic")
    
    # Test data
    transcript = """
    According to a study published in Nature, researchers at MIT discovered 
    a new quantum computing breakthrough. Recent research shows that this 
    phenomenon occurs at specific wavelengths.
    """
    
    description = """
    Links:
    - Paper: https://www.nature.com/articles/d41586-024-00001-1
    - More info: https://mit.edu/research
    """
    
    labels = [
        {'timestamp': 45.2, 'description': 'Screenshot of research paper from Nature journal'},
        {'timestamp': 120.5, 'description': 'Graph showing experimental data'},
    ]
    
    sources = detective.detect_sources(transcript, description, labels)
    
    if sources:
        print(f"✅ SUCCESS!")
        print(f"   Found {len(sources)} sources")
        for i, source in enumerate(sources[:3]):
            print(f"   {i+1}. [{source['type']}] {source.get('content', '')[:60]}... (conf: {source['confidence']:.2f})")
        return True
    else:
        print("❌ FAILED")
        return False


def test_source_scraper():
    """Test source content scraping."""
    print("\n🧪 Testing Source Scraper...")
    
    from src.data_collection.source_scraper import SourceScraper
    
    scraper = SourceScraper(output_dir="data/test/sources")
    
    # Test with a public article
    test_sources = [
        {
            'type': 'url',
            'content': 'https://www.wikipedia.org/wiki/Artificial_intelligence',
            'confidence': 0.95
        }
    ]
    
    scraped = scraper.scrape_all_sources(test_sources)
    
    if scraped and scraped[0].get('success'):
        print(f"✅ SUCCESS!")
        print(f"   Scraped {len(scraped)} sources")
        print(f"   Text length: {len(scraped[0].get('text', ''))} chars")
        return True
    else:
        print("❌ FAILED")
        return False


def test_dataset_builder():
    """Test dataset building."""
    print("\n🧪 Testing Dataset Builder...")
    
    from src.training.dataset_builder import DatasetBuilder
    
    builder = DatasetBuilder(output_dir="data/test/training_dataset")
    
    # Create mock data
    metadata = {
        'video_id': 'test123',
        'title': 'Test Video',
        'duration': 300,
        'channel': 'Test Channel',
        'transcript': 'This is a test transcript with some content.',
        'categories': ['Science'],
    }
    
    labels = [
        {'timestamp': 0, 'description': 'Opening scene', 'scene_change': False},
        {'timestamp': 10, 'description': 'New scene', 'scene_change': True},
    ]
    
    sources = [
        {
            'success': True,
            'title': 'Test Article',
            'text': 'Article content here...',
            'url': 'https://example.com',
            'source_type': 'article'
        }
    ]
    
    example = builder.build_training_example(metadata, labels, sources)
    
    if example and 'input' in example and 'output' in example:
        print(f"✅ SUCCESS!")
        print(f"   Sources: {len(example['input']['sources'])}")
        print(f"   Segments: {len(example['output']['segments'])}")
        return True
    else:
        print("❌ FAILED")
        return False


def run_all_tests():
    """Run all component tests."""
    
    print("="*70)
    print("🧪 PIPELINE COMPONENT TESTS")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['YouTube Scraper'] = test_youtube_scraper()
    results['Video Labeler'] = test_video_labeler()
    results['Source Detective'] = test_source_detective()
    results['Source Scraper'] = test_source_scraper()
    results['Dataset Builder'] = test_dataset_builder()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {component}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline components")
    parser.add_argument(
        "--component",
        choices=['scraper', 'labeler', 'detective', 'source-scraper', 'dataset', 'all'],
        default='all',
        help="Which component to test"
    )
    
    args = parser.parse_args()
    
    if args.component == 'all':
        run_all_tests()
    elif args.component == 'scraper':
        test_youtube_scraper()
    elif args.component == 'labeler':
        test_video_labeler()
    elif args.component == 'detective':
        test_source_detective()
    elif args.component == 'source-scraper':
        test_source_scraper()
    elif args.component == 'dataset':
        test_dataset_builder()
