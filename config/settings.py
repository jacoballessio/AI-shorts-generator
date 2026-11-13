"""
Configuration Management
Centralized settings and configuration for AI Shorts Generator
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class with all settings."""
    
    # =============================================================================
    # Project Paths
    # =============================================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    OUTPUT_DIR = PROJECT_ROOT / "output"
    CACHE_DIR = PROJECT_ROOT / "cache"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # =============================================================================
    # LLM Settings
    # =============================================================================
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic or openai
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # =============================================================================
    # Research Settings
    # =============================================================================
    RESEARCH_SOURCES = ["reddit", "google_trends", "news"]  # Available sources
    RESEARCH_ENABLED_SOURCES = os.getenv("RESEARCH_SOURCES", "reddit,google_trends,news").split(",")
    VIRAL_THRESHOLD = float(os.getenv("VIRAL_THRESHOLD", "0.4"))  # Lower threshold = more topics
    
    # Reddit API
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = "AI-Shorts-Generator/1.0"
    
    # News API
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Twitter/X API (future)
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    
    # =============================================================================
    # Content Creation Settings
    # =============================================================================
    DEFAULT_DURATION = int(os.getenv("DEFAULT_DURATION", "45"))
    DEFAULT_STYLE = os.getenv("DEFAULT_STYLE", "engaging")
    DEFAULT_TONE = os.getenv("DEFAULT_TONE", "informative")
    
    AVAILABLE_STYLES = ["engaging", "dramatic", "educational", "humorous"]
    AVAILABLE_TONES = ["informative", "casual", "professional", "excited"]
    
    # =============================================================================
    # TTS Settings
    # =============================================================================
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge")  # edge or elevenlabs
    
    # Edge TTS (free)
    EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")
    AVAILABLE_EDGE_VOICES = [
        "en-US-AriaNeural",      # Female, friendly
        "en-US-GuyNeural",       # Male, professional
        "en-US-JennyNeural",     # Female, warm
        "en-GB-SoniaNeural",     # British female
        "en-AU-NatashaNeural",   # Australian female
    ]
    
    # ElevenLabs (premium)
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE", "Rachel")
    
    # =============================================================================
    # Image Settings
    # =============================================================================
    USE_STABLE_DIFFUSION = os.getenv("USE_STABLE_DIFFUSION", "true").lower() == "true"
    SD_MODEL_PATH = os.getenv("SD_MODEL_PATH", "runwayml/stable-diffusion-v1-5")
    
    # Image sources
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
    
    # Image generation settings
    SD_INFERENCE_STEPS = 30
    SD_GUIDANCE_SCALE = 7.5
    IMAGE_WIDTH = 512   # 9:16 aspect ratio
    IMAGE_HEIGHT = 768
    
    # =============================================================================
    # Video Settings
    # =============================================================================
    VIDEO_WIDTH = 1080
    VIDEO_HEIGHT = 1920  # 9:16 vertical for shorts
    VIDEO_FPS = 30
    VIDEO_CODEC = "libx264"
    VIDEO_BITRATE = "8000k"
    AUDIO_CODEC = "aac"
    
    # Caption settings
    CAPTION_STYLE = os.getenv("CAPTION_STYLE", "animated")  # animated, static, karaoke
    CAPTION_FONT_SIZE = 80
    CAPTION_FONT_COLOR = "white"
    CAPTION_STROKE_COLOR = "black"
    CAPTION_STROKE_WIDTH = 3
    
    # Transition settings
    TRANSITION_TYPE = "crossfade"  # crossfade, zoom, slide
    TRANSITION_DURATION = 0.5
    
    # Background music
    ADD_BACKGROUND_MUSIC = os.getenv("ADD_BACKGROUND_MUSIC", "false").lower() == "true"
    MUSIC_VOLUME = 0.2
    
    # =============================================================================
    # Performance Settings
    # =============================================================================
    USE_GPU = os.getenv("USE_GPU", "auto")  # auto, true, false
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    # =============================================================================
    # Web UI Settings
    # =============================================================================
    UI_SERVER_NAME = os.getenv("UI_SERVER_NAME", "0.0.0.0")
    UI_SERVER_PORT = int(os.getenv("UI_SERVER_PORT", "7860"))
    UI_SHARE = os.getenv("UI_SHARE", "false").lower() == "true"
    
    # =============================================================================
    # Logging Settings
    # =============================================================================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = CACHE_DIR / "generator.log"
    
    @classmethod
    def to_dict(cls) -> Dict:
        """Export configuration as dictionary."""
        return {
            "llm_provider": cls.LLM_PROVIDER,
            "llm_model": cls.LLM_MODEL,
            "duration": cls.DEFAULT_DURATION,
            "style": cls.DEFAULT_STYLE,
            "tone": cls.DEFAULT_TONE,
            "research": {
                "sources": cls.RESEARCH_ENABLED_SOURCES,
                "viral_threshold": cls.VIRAL_THRESHOLD
            },
            "media": {
                "tts_provider": cls.TTS_PROVIDER,
                "voice_name": cls.EDGE_TTS_VOICE,
                "use_stable_diffusion": cls.USE_STABLE_DIFFUSION,
                "sd_model_path": cls.SD_MODEL_PATH,
                "output_dir": str(cls.CACHE_DIR / "media")
            },
            "video": {
                "width": cls.VIDEO_WIDTH,
                "height": cls.VIDEO_HEIGHT,
                "fps": cls.VIDEO_FPS,
                "caption_style": cls.CAPTION_STYLE,
                "add_background_music": cls.ADD_BACKGROUND_MUSIC
            },
            "output_dir": str(cls.OUTPUT_DIR),
            "cache_dir": str(cls.CACHE_DIR)
        }
    
    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages
        """
        messages = []
        
        # Check required API keys
        if not cls.ANTHROPIC_API_KEY and not cls.OPENAI_API_KEY:
            messages.append("❌ ERROR: No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        
        # Check optional but recommended keys
        if not cls.REDDIT_CLIENT_ID and "reddit" in cls.RESEARCH_ENABLED_SOURCES:
            messages.append("⚠️  WARNING: Reddit API not configured. Reddit research will be disabled.")
        
        if not cls.NEWS_API_KEY and "news" in cls.RESEARCH_ENABLED_SOURCES:
            messages.append("⚠️  WARNING: News API not configured. News research will be disabled.")
        
        if not cls.PEXELS_API_KEY and not cls.USE_STABLE_DIFFUSION:
            messages.append("⚠️  WARNING: No image sources configured. Enable Stable Diffusion or add Pexels API key.")
        
        # Create directories
        for directory in [cls.OUTPUT_DIR, cls.CACHE_DIR, cls.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        return messages
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "=" * 70)
        print("📋 AI Shorts Generator Configuration")
        print("=" * 70)
        
        print(f"\n🤖 LLM:")
        print(f"   Provider: {cls.LLM_PROVIDER}")
        print(f"   Model: {cls.LLM_MODEL}")
        print(f"   API Key: {'✓' if cls.ANTHROPIC_API_KEY or cls.OPENAI_API_KEY else '✗'}")
        
        print(f"\n🔍 Research:")
        print(f"   Sources: {', '.join(cls.RESEARCH_ENABLED_SOURCES)}")
        print(f"   Viral Threshold: {cls.VIRAL_THRESHOLD}")
        
        print(f"\n🎙️  TTS:")
        print(f"   Provider: {cls.TTS_PROVIDER}")
        print(f"   Voice: {cls.EDGE_TTS_VOICE if cls.TTS_PROVIDER == 'edge' else cls.ELEVENLABS_VOICE}")
        
        print(f"\n🖼️  Images:")
        print(f"   Stable Diffusion: {'Enabled' if cls.USE_STABLE_DIFFUSION else 'Disabled'}")
        print(f"   Pexels API: {'✓' if cls.PEXELS_API_KEY else '✗'}")
        
        print(f"\n🎥 Video:")
        print(f"   Resolution: {cls.VIDEO_WIDTH}x{cls.VIDEO_HEIGHT}")
        print(f"   FPS: {cls.VIDEO_FPS}")
        print(f"   Caption Style: {cls.CAPTION_STYLE}")
        
        print(f"\n📁 Directories:")
        print(f"   Output: {cls.OUTPUT_DIR}")
        print(f"   Cache: {cls.CACHE_DIR}")
        
        print("\n" + "=" * 70 + "\n")
        
        # Print validation messages
        validation = cls.validate()
        if validation:
            print("Validation Messages:")
            for msg in validation:
                print(f"  {msg}")
            print()


# Auto-validate on import
if __name__ != "__main__":
    Config.validate()
