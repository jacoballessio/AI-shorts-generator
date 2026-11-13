"""
Example: Basic usage of AI Shorts Generator
This script demonstrates how to use the system programmatically.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from research_agent import ResearchAgent
from content_creator import ContentCreator
from media_generator import MediaGenerator
from video_composer import VideoComposer, VideoConfig
from config.settings import Config


async def example_autonomous():
    """
    Example: Fully autonomous video generation.
    """
    print("=" * 70)
    print("Example 1: Autonomous Video Generation")
    print("=" * 70)
    
    # Print current configuration
    Config.print_config()
    
    # Step 1: Research trending topics
    print("\n🔍 Researching trending topics...")
    research_agent = ResearchAgent()
    topics = await research_agent.research_topics(num_topics=5)
    
    if not topics:
        print("❌ No topics found!")
        return
    
    best_topic = topics[0]
    print(f"\n✨ Best Topic: {best_topic.title}")
    print(f"   Viral Potential: {best_topic.viral_potential:.2f}")
    print(f"   Source: {best_topic.source}")
    
    # Step 2: Create script
    print("\n✍️ Creating video script...")
    content_creator = ContentCreator(provider="anthropic")
    script = content_creator.create_script(
        topic=best_topic.to_dict(),
        duration=30,
        style="engaging",
        tone="excited"
    )
    
    print(f"\n📝 Script Preview:")
    print(f"   Hook: {script.hook[:80]}...")
    print(f"   Estimated Duration: {script.estimated_duration}s")
    
    # Step 3: Generate media
    print("\n🎬 Generating media assets...")
    media_generator = MediaGenerator()
    assets = await media_generator.generate_all_assets(
        script=script.to_dict(),
        video_id="example_001"
    )
    
    print(f"\n📦 Assets:")
    print(f"   Audio: {assets.audio_duration:.2f}s")
    print(f"   Images: {len(assets.images)}")
    print(f"   Captions: {len(assets.captions)} words")
    
    # Step 4: Compose video
    print("\n🎥 Composing video...")
    video_composer = VideoComposer(VideoConfig())
    
    assets_dict = {
        'audio_path': assets.audio_path,
        'audio_duration': assets.audio_duration,
        'images': assets.images,
        'captions': assets.captions
    }
    
    video_path = video_composer.compose_video(
        assets=assets_dict,
        output_filename="example_autonomous.mp4"
    )
    
    print(f"\n✅ VIDEO COMPLETE!")
    print(f"📹 Location: {video_path}")


async def example_manual():
    """
    Example: Manual topic video generation.
    """
    print("\n" + "=" * 70)
    print("Example 2: Manual Topic Video Generation")
    print("=" * 70)
    
    # Create a manual topic
    topic = {
        "title": "The Future of AI in 2026",
        "description": "Exploring the latest breakthroughs and predictions for artificial intelligence.",
        "source": "manual_example"
    }
    
    print(f"\n📝 Topic: {topic['title']}")
    
    # Create script
    print("\n✍️ Creating script...")
    content_creator = ContentCreator(provider="anthropic")
    script = content_creator.create_script(
        topic=topic,
        duration=45,
        style="educational",
        tone="professional"
    )
    
    print(f"\n📝 Script Generated:")
    print(f"   Word count: {script.metadata['word_count']}")
    print(f"   Duration: ~{script.estimated_duration}s")
    print(f"\n   Hook: {script.hook}")
    print(f"\n   Full script:\n   {script.full_script[:200]}...")
    
    # Save script for review
    content_creator.save_script(script, "example_script.json")
    print(f"\n💾 Script saved to: example_script.json")


async def example_research_only():
    """
    Example: Just research trending topics without generating video.
    """
    print("\n" + "=" * 70)
    print("Example 3: Research Only")
    print("=" * 70)
    
    research_agent = ResearchAgent()
    topics = await research_agent.research_topics(num_topics=10)
    
    print("\n🎯 Top 10 Trending Topics:\n")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. [{topic.viral_potential:.2f}] {topic.title}")
        print(f"   Source: {topic.source}")
        print(f"   {topic.description[:100]}...")
        print()
    
    # Save research results
    research_agent.save_research(topics, "research_results.json")
    print(f"💾 Research saved to: research_results.json")


async def main():
    """
    Run all examples.
    """
    print("\n" + "🎬" * 35)
    print("AI SHORTS GENERATOR - EXAMPLES")
    print("🎬" * 35)
    
    # Example 1: Research only (fast)
    await example_research_only()
    
    # Example 2: Manual topic (creates script only)
    await example_manual()
    
    # Example 3: Full autonomous generation (takes 2-5 minutes)
    # Uncomment to run full video generation:
    # await example_autonomous()
    
    print("\n" + "=" * 70)
    print("✅ Examples Complete!")
    print("=" * 70)
    print("\nTo generate a full video, uncomment the 'example_autonomous()' call")
    print("or run: python src/main.py --mode auto")
    print()


if __name__ == "__main__":
    asyncio.run(main())
