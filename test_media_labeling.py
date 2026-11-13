"""
Test script for media labeling integration.
Demonstrates how labeled media from articles is used in script generation.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from research_agent import TopicCandidate
from content_creator import ContentCreator
from media_labeler import MediaLabeler


async def test_media_labeling():
    """Test media labeling and script generation."""
    
    print("=" * 70)
    print("🧪 Testing Media Labeling Integration")
    print("=" * 70)
    
    # Create a sample topic with media URLs (e.g., from a Reddit post or article)
    sample_topic = TopicCandidate(
        title="DeepSeek's Impact on AI Development",
        description="Chinese AI startup DeepSeek releases open-source models that match GPT-4 performance at fraction of the cost",
        source="reddit",
        engagement_score=0.9,
        trending_score=0.8,
        url="https://example.com/article",
        metadata={
            # Example media URLs from article
            "media_urls": {
                "images": [
                    "https://picsum.photos/800/600",  # Placeholder images for testing
                    "https://picsum.photos/800/600?random=2"
                ],
                "videos": [
                    # Add your own video URL here for testing, or use example.mp4
                ],
                "youtube": []
            }
        }
    )
    
    print(f"\n📰 Sample Topic:")
    print(f"   Title: {sample_topic.title}")
    print(f"   Media URLs: {len(sample_topic.metadata['media_urls']['images'])} images, {len(sample_topic.metadata['media_urls']['videos'])} videos")
    
    # Step 1: Label media from article
    print(f"\n🏷️  STEP 1: Labeling media from article...")
    media_labeler = MediaLabeler(cache_dir="./cache/test_media_labels")
    
    labeled_media = await media_labeler.label_media_from_article(
        media_urls=sample_topic.metadata['media_urls'],
        article_id="test_article"
    )
    
    if labeled_media:
        print(f"\n✅ Labeled {len(labeled_media)} media assets:")
        for i, media in enumerate(labeled_media):
            media_dict = media.to_dict()
            print(f"\n   Media {i+1} ({media_dict['media_type']}):")
            print(f"   Description: {media_dict['description'][:100]}...")
            print(f"   Keywords: {', '.join(media_dict.get('keywords', [])[:5])}")
            if media_dict.get('duration'):
                print(f"   Duration: {media_dict['duration']:.1f}s")
                print(f"   Clip-worthy segments: {len(media_dict.get('clip_worthy_segments', []))}")
    
    # Step 2: Generate script with labeled media
    print(f"\n✍️  STEP 2: Generating script with labeled media...")
    content_creator = ContentCreator(provider="openai")  # or "anthropic"
    
    script = content_creator.create_script(
        topic=sample_topic.to_dict(),
        duration=45,
        style="engaging",
        tone="informative",
        labeled_media=labeled_media
    )
    
    print(f"\n📝 Generated Script:")
    print(f"   Full script: {script.full_script[:200]}...")
    print(f"   Duration estimate: {script.estimated_duration}s")
    print(f"   Number of segments: {len(script.segments)}")
    
    # Show media assignments
    print(f"\n📸 Media Assignments:")
    for i, segment in enumerate(script.segments):
        print(f"\n   Segment {i+1} ({segment.type}):")
        print(f"   Text: {segment.text[:60]}...")
        print(f"   Media Assignment: {segment.media_assignment}")
        if segment.assigned_media_path:
            print(f"   ✅ Assigned: {Path(segment.assigned_media_path).name}")
            if segment.clip_start is not None:
                print(f"   Clip: {segment.clip_start:.1f}s - {segment.clip_end:.1f}s")
        else:
            print(f"   🎨 Will generate: {segment.media_type}")
            print(f"   Search: {segment.search_query}")
    
    print(f"\n✅ Test complete!")
    print(f"\nNext steps:")
    print(f"1. The video composer will use assigned_media_path for segments with media")
    print(f"2. For segments without assigned media, it will generate/fetch as before")
    print(f"3. TTS timing will determine when each media appears based on segment duration")


if __name__ == "__main__":
    # Note: You'll need API keys configured in .env
    # OPENAI_API_KEY or ANTHROPIC_API_KEY
    
    asyncio.run(test_media_labeling())
