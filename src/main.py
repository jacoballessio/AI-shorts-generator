"""
Main Orchestrator
Coordinates the entire AI shorts generation pipeline.
"""

import os
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, List
import sys
from dotenv import load_dotenv

load_dotenv()

from .research_agent import ResearchAgent, TopicCandidate
from .content_creator import ContentCreator, VideoScript
from .media_generator import MediaGenerator, MediaAssets
from .video_composer import VideoComposer, VideoConfig

# Lazy import for media labeler (loaded when needed)
MediaLabeler = None


class ShortVideoOrchestrator:
    """
    Main orchestrator that coordinates the entire video generation pipeline.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize all components
        print("🚀 Initializing AI Shorts Generator...")
        
        # Pass style to research agent config
        research_config = self.config.get('research', {}).copy()
        research_config['style'] = self.config.get('style', 'engaging')
        
        self.research_agent = ResearchAgent(
            config=research_config
        )
        
        self.content_creator = ContentCreator(
            provider=self.config.get('llm_provider', 'anthropic'),
            model=self.config.get('llm_model')
        )
        
        self.media_generator = MediaGenerator(
            config=self.config.get('media', {})
        )
        
        self.video_composer = VideoComposer(
            config=VideoConfig(**self.config.get('video', {}))
        )
        
        # Media labeler will be lazy-loaded when needed
        self.media_labeler = None
        self._media_labeling_enabled = self.config.get('enable_media_labeling', True)
        
        # Setup output directories
        self.output_dir = Path(self.config.get('output_dir', './output'))
        self.cache_dir = Path(self.config.get('cache_dir', './cache'))
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        print("✅ All components initialized\n")
    
    def _ensure_media_labeler(self):
        """Lazy load media labeler when needed."""
        if self.media_labeler is None and self._media_labeling_enabled:
            try:
                global MediaLabeler
                if MediaLabeler is None:
                    from .media_labeler import MediaLabeler as ML
                    MediaLabeler = ML
                
                self.media_labeler = MediaLabeler(
                    cache_dir=str(self.cache_dir / 'media_labels')
                )
                print("✅ Media labeler initialized")
            except ImportError as e:
                print(f"⚠️  Warning: Could not load media labeler: {e}")
                print("   Continuing without media labeling...")
                self._media_labeling_enabled = False
        
        return self.media_labeler
        
        return self.media_labeler
    
    async def generate_video_autonomous(self, num_candidate_topics: int = 10) -> str:
        """
        Fully autonomous video generation pipeline.
        The AI researches, selects topic, creates content, and produces video.
        
        Args:
            num_candidate_topics: Number of topics to research before selecting best
            
        Returns:
            Path to generated video file
        """
        print("=" * 70)
        print("🤖 AUTONOMOUS VIDEO GENERATION MODE")
        print("=" * 70)
        
        # Step 1: Research trending topics (style-aware)
        print(f"\n📊 STEP 1: Researching trending topics optimized for '{self.config.get('style', 'engaging')}' style...")
        # Pass style to research agent
        self.research_agent.style = self.config.get('style', 'engaging')
        topics = await self.research_agent.research_topics(num_topics=num_candidate_topics)
        
        if not topics:
            raise RuntimeError("No topics found during research!")
        
        # Topics are already randomized with weighted selection by research agent
        # Pick the first one from the diverse, high-quality pool
        best_topic = topics[0]
        print(f"\n✨ Selected Topic (Viral Potential: {best_topic.viral_potential:.2f}):")
        print(f"   Title: {best_topic.title}")
        print(f"   Source: {best_topic.source}")
        
        # Save research results
        research_file = self.cache_dir / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.research_agent.save_research(topics, str(research_file))
        
        # Step 1.5: Label media from article (if any media URLs found)
        labeled_media = None
        topic_metadata = best_topic.metadata or {}
        media_urls = topic_metadata.get('media_urls', {})
        
        if any(media_urls.get(k, []) for k in ['images', 'videos']):
            print("\n🏷️  STEP 1.5: Labeling media from article...")
            try:
                # Initialize media labeler if needed
                media_labeler = self._ensure_media_labeler()
                
                if media_labeler:
                    article_id = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    labeled_media = await media_labeler.label_media_from_article(
                        media_urls=media_urls,
                        article_id=article_id
                    )
                    
                    if labeled_media:
                        print(f"\n📋 Labeled Media Summary:")
                        for i, media in enumerate(labeled_media):
                            media_dict = media.to_dict()
                            print(f"   {i+1}. {media_dict['media_type']}: {media_dict['description'][:60]}...")
                            if media_dict.get('duration'):
                                print(f"      Duration: {media_dict['duration']:.1f}s")
                            print(f"      Keywords: {', '.join(media_dict.get('keywords', [])[:3])}")
                else:
                    print("  ⚠️ Media labeling disabled or unavailable")
            except Exception as e:
                print(f"  ⚠️ Media labeling failed: {e}")
                print("  Continuing without labeled media...")
                labeled_media = None
        
        # Step 2: Create content
        print("\n✍️ STEP 2: Creating video script...")
        script = self.content_creator.create_script(
            topic=best_topic.to_dict(),
            duration=self.config.get('duration', 45),
            style=self.config.get('style', 'engaging'),
            tone=self.config.get('tone', 'informative'),
            labeled_media=labeled_media
        )
        
        # Save script
        script_file = self.cache_dir / f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.content_creator.save_script(script, str(script_file))
        
        print(f"\n📝 Script Preview:")
        print(f"   Hook: {script.hook[:80]}...")
        print(f"   Duration: ~{script.estimated_duration}s")
        
        # Step 3: Generate media assets
        print("\n🎬 STEP 3: Generating media assets...")
        video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        assets = await self.media_generator.generate_all_assets(
            script=script.to_dict(),
            video_id=video_id
        )
        
        print(f"\n📦 Assets Generated:")
        print(f"   Audio: {assets.audio_duration:.2f}s")
        print(f"   Images: {len(assets.images)}")
        print(f"   Video clips: {len(assets.video_clips)}")
        print(f"   Captions: {len(assets.captions)} words")
        
        # Step 4: Compose final video
        print("\n🎥 STEP 4: Composing final video...")
        
        # Convert MediaAssets to dict for video composer
        assets_dict = {
            'audio_path': assets.audio_path,
            'audio_duration': assets.audio_duration,
            'images': assets.images,
            'video_clips': assets.video_clips,
            'captions': assets.captions
        }
        
        # Debug: Verify video_clips are being passed
        if assets.video_clips:
            print(f"   DEBUG: Passing {len(assets.video_clips)} video clips to composer")
            print(f"   DEBUG: First clip: {assets.video_clips[0]}")
        
        output_filename = f"{video_id}.mp4"
        video_path = self.video_composer.compose_video(
            assets=assets_dict,
            output_filename=output_filename
        )
        
        # Step 5: Save metadata
        metadata = {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'topic': best_topic.to_dict(),
            'script': script.to_dict(),
            'assets': {
                'audio_path': assets.audio_path,
                'num_images': len(assets.images),
                'num_video_clips': len(assets.video_clips),
                'num_captions': len(assets.captions)
            },
            'video_path': video_path,
            'config': self.config
        }
        
        metadata_file = self.output_dir / f"{video_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("✅ VIDEO GENERATION COMPLETE!")
        print("=" * 70)
        print(f"\n📹 Video: {video_path}")
        print(f"📄 Metadata: {metadata_file}")
        print(f"\n🎯 Topic: {best_topic.title}")
        print(f"📊 Viral Potential: {best_topic.viral_potential:.2f}")
        print(f"⏱️  Duration: {assets.audio_duration:.2f}s")
        
        return video_path
    
    async def generate_video_manual(self, topic_title: str, topic_description: str = "") -> str:
        """
        Generate video from manually specified topic.
        
        Args:
            topic_title: Title/subject of the video
            topic_description: Additional context about the topic
            
        Returns:
            Path to generated video file
        """
        print("=" * 70)
        print("👤 MANUAL TOPIC MODE")
        print("=" * 70)
        
        # Create manual topic
        manual_topic = TopicCandidate(
            title=topic_title,
            description=topic_description,
            source="manual_input",
            engagement_score=0.8,
            trending_score=0.8,
            url=""
        )
        
        print(f"\n📝 Topic: {topic_title}")
        
        # Step 2: Create content
        print("\n✍️ Creating video script...")
        script = self.content_creator.create_script(
            topic=manual_topic.to_dict(),
            duration=self.config.get('duration', 45),
            style=self.config.get('style', 'engaging'),
            tone=self.config.get('tone', 'informative')
        )
        
        # Save script
        script_file = self.cache_dir / f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.content_creator.save_script(script, str(script_file))
        
        # Step 3: Generate media
        video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        assets = await self.media_generator.generate_all_assets(
            script=script.to_dict(),
            video_id=video_id
        )
        
        # Step 4: Compose video
        assets_dict = {
            'audio_path': assets.audio_path,
            'audio_duration': assets.audio_duration,
            'images': assets.images,
            'video_clips': assets.video_clips,
            'captions': assets.captions
        }
        
        output_filename = f"{video_id}.mp4"
        video_path = self.video_composer.compose_video(
            assets=assets_dict,
            output_filename=output_filename
        )
        
        print("\n✅ VIDEO GENERATION COMPLETE!")
        print(f"📹 Video: {video_path}\n")
        
        return video_path
    
    def generate_batch(self, num_videos: int = 5) -> List[str]:
        """
        Generate multiple videos in batch.
        
        Args:
            num_videos: Number of videos to generate
            
        Returns:
            List of paths to generated videos
        """
        print(f"\n🔄 BATCH MODE: Generating {num_videos} videos...")
        
        video_paths = []
        
        for i in range(num_videos):
            print(f"\n{'=' * 70}")
            print(f"Video {i+1}/{num_videos}")
            print(f"{'=' * 70}")
            
            try:
                video_path = asyncio.run(self.generate_video_autonomous())
                video_paths.append(video_path)
            except Exception as e:
                print(f"❌ Error generating video {i+1}: {e}")
                continue
        
        print(f"\n✅ Batch complete: {len(video_paths)}/{num_videos} videos generated")
        return video_paths


def load_config(config_file: Optional[str] = None) -> Dict:
    """Load configuration from file or use defaults."""
    default_config = {
        'llm_provider': 'anthropic',
        # llm_model is auto-selected based on provider (see ContentCreator.__init__)
        'duration': 45,
        'style': 'engaging',
        'tone': 'informative',
        'research': {
            'sources': ['reddit', 'google_trends', 'news'],
            'viral_threshold': 0.6
        },
        'media': {
            'tts_provider': os.getenv('TTS_PROVIDER', 'edge'),  # 'edge' (free) or 'elevenlabs' (paid, better quality)
            'voice_name': os.getenv('EDGE_TTS_VOICE', 'en-US-AriaNeural'),  # Edge TTS voice
            'elevenlabs_voice_id': os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM'),  # Rachel - free tier voice
            'elevenlabs_model': os.getenv('ELEVENLABS_MODEL', 'eleven_turbo_v2_5'),  # Fast with good quality
            'use_stable_diffusion': os.getenv('USE_STABLE_DIFFUSION', 'false').lower() == 'true',
            'output_dir': './cache/media'
        },
        'video': {
            'width': 1080,
            'height': 1920,
            'fps': 30,
            'caption_style': 'animated',
            'add_background_music': False
        },
        'output_dir': './output',
        'cache_dir': './cache'
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        # Merge configs
        default_config.update(user_config)
    
    return default_config


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='AI Shorts Generator - Autonomous short-form video creation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully autonomous mode (AI picks topic)
  python main.py --mode auto
  
  # Manual topic mode
  python main.py --mode manual --topic "AI breakthrough in 2026"
  
  # Batch generation
  python main.py --mode batch --count 5
  
  # Custom settings
  python main.py --mode auto --duration 60 --style dramatic
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['auto', 'manual', 'batch'],
        default='auto',
        help='Generation mode'
    )
    
    parser.add_argument(
        '--topic',
        type=str,
        help='Manual topic (required for manual mode)'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Topic description (optional for manual mode)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=45,
        help='Target video duration in seconds'
    )
    
    parser.add_argument(
        '--style',
        choices=['engaging', 'dramatic', 'educational', 'humorous'],
        default='engaging',
        help='Video style'
    )
    
    parser.add_argument(
        '--tone',
        choices=['informative', 'casual', 'professional', 'excited'],
        default='informative',
        help='Narrative tone'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=5,
        help='Number of videos for batch mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config JSON file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with CLI arguments
    if args.duration:
        config['duration'] = args.duration
    if args.style:
        config['style'] = args.style
    if args.tone:
        config['tone'] = args.tone
    
    # Initialize orchestrator
    orchestrator = ShortVideoOrchestrator(config)
    
    # Run based on mode
    try:
        if args.mode == 'auto':
            video_path = asyncio.run(orchestrator.generate_video_autonomous())
            print(f"\n🎉 Success! Video saved to: {video_path}")
            
        elif args.mode == 'manual':
            if not args.topic:
                print("❌ Error: --topic required for manual mode")
                sys.exit(1)
            
            video_path = asyncio.run(orchestrator.generate_video_manual(
                topic_title=args.topic,
                topic_description=args.description
            ))
            print(f"\n🎉 Success! Video saved to: {video_path}")
            
        elif args.mode == 'batch':
            video_paths = orchestrator.generate_batch(num_videos=args.count)
            print(f"\n🎉 Success! Generated {len(video_paths)} videos")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
