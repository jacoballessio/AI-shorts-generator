"""                                                                                                                   
Content Creator Module
Generates engaging scripts for short-form videos using AI language models.
"""

import os
import json
import gc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Local LLM support (optional)
try:
    from llama_cpp import Llama
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    Llama = None


@dataclass
class ScriptSegment:
    """Represents a single segment of the script with its media description."""
    text: str
    type: str  # "hook", "content", "cta"
    media_description: str  # Description for the image/video to accompany this segment
    media_type: str  # "image" or "video" - preferred type
    media_assignment: Optional[str] = None  # "GENERATE", "MEDIA 1", "MEDIA 2", etc.
    assigned_media_path: Optional[str] = None  # Resolved path to actual media file
    clip_start: Optional[float] = None  # For video clips
    clip_end: Optional[float] = None  # For video clips
    sd_prompt: Optional[str] = None  # Stable Diffusion prompt if generating image
    search_query: Optional[str] = None  # Search query for stock images/videos
    duration_estimate: Optional[float] = None  # Estimated duration in seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.type,
            "media_description": self.media_description,
            "media_type": self.media_type,
            "media_assignment": self.media_assignment,
            "assigned_media_path": self.assigned_media_path,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "sd_prompt": self.sd_prompt,
            "search_query": self.search_query,
            "duration_estimate": self.duration_estimate
        }


@dataclass
class VideoScript:
    """Represents a complete video script with metadata."""
    segments: List[ScriptSegment]  # Ordered list of script segments with media
    full_script: str
    estimated_duration: int  # seconds
    metadata: Dict
    
    # Legacy fields for backward compatibility
    @property
    def hook(self) -> str:
        """Get hook text (first segment)."""
        return self.segments[0].text if self.segments else ""
    
    @property
    def main_content(self) -> List[str]:
        """Get main content segments (middle segments)."""
        return [s.text for s in self.segments[1:-1]] if len(self.segments) > 2 else []
    
    @property
    def call_to_action(self) -> str:
        """Get CTA text (last segment)."""
        return self.segments[-1].text if self.segments else ""
    
    @property
    def image_prompts(self) -> List[str]:
        """Legacy: Get all SD prompts."""
        return [s.sd_prompt for s in self.segments if s.sd_prompt]
    
    @property
    def search_queries(self) -> List[str]:
        """Legacy: Get all search queries."""
        return [s.search_query for s in self.segments if s.search_query]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "segments": [s.to_dict() for s in self.segments],
            "full_script": self.full_script,
            "estimated_duration": self.estimated_duration,
            "hook": self.hook,  # Legacy
            "main_content": self.main_content,  # Legacy
            "call_to_action": self.call_to_action,  # Legacy
            "image_prompts": self.image_prompts,  # Legacy
            "search_queries": self.search_queries,  # Legacy
            "metadata": self.metadata
        }


class ContentCreator:
    """
    AI-powered content creator that generates engaging scripts for short videos.
    """
    
    def __init__(self, provider: str = "anthropic", model: str = None):
        """
        Initialize the content creator.
        
        Args:
            provider: "anthropic", "openai", or "local"
            model: Specific model to use (defaults to best available)
        """
        self.provider = provider
        self.local_llm = None  # Lazy loaded when needed
        self.local_model_path = os.getenv('LOCAL_LLM_PATH', '')
        
        print(f"💬 ContentCreator initialized with provider: {provider}")
        
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-5-sonnet-20241022"
            print(f"   Using model: {self.model}")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  WARNING: OPENAI_API_KEY not found in environment!")
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4-turbo-preview"
            print(f"   Using model: {self.model}")
        elif provider == "local":
            if not LOCAL_LLM_AVAILABLE:
                raise ValueError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            if not self.local_model_path or not os.path.exists(self.local_model_path):
                raise ValueError(f"LOCAL_LLM_PATH not set or file not found: {self.local_model_path}")
            self.model = model or "local-8b"
            print(f"   Using local model: {self.local_model_path}")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def create_script(self, topic: Dict, duration: int = 45, 
                     style: str = "engaging", tone: str = "informative",
                     labeled_media: List = None) -> VideoScript:
        """
        Generate a complete video script from a topic.
        
        Args:
            topic: Topic dictionary with title, description, source, etc.
            duration: Target duration in seconds
            style: Video style (engaging, dramatic, educational, humorous)
            tone: Narrative tone (informative, casual, professional, excited)
            labeled_media: List of LabeledMedia objects from the article (optional)
            
        Returns:
            VideoScript object with complete script and metadata
        """
        print(f"✍️ Generating {duration}s {style} script about: {topic.get('title', 'Unknown')}")
        if labeled_media:
            print(f"   Using {len(labeled_media)} labeled media assets from article")
        
        # Create the prompt
        prompt = self._create_prompt(topic, duration, style, tone, labeled_media)
        
        # Get script from LLM with retry on refusal
        response = self._call_llm_with_retry(prompt, topic, duration, style, tone)
        
        # Parse response into structured script
        script = self._parse_script_response(response, topic, duration, labeled_media)
        
        print(f"✅ Script generated: {len(script.full_script)} characters, ~{script.estimated_duration}s")
        return script
    
    def _generate_thesis(self, topic: Dict, style: str, tone: str) -> str:
        """Generate a unique angle/thesis for the video based on research."""
        topic_metadata = topic.get('metadata', {})
        article_content = topic_metadata.get('article_content', '')
        key_points = topic_metadata.get('key_points', [])
        top_comments = topic_metadata.get('top_comments', [])
        
        # Build research summary
        research_summary = f"""TOPIC: {topic.get('title', '')}
DESCRIPTION: {topic.get('description', '')}
SOURCE: {topic.get('source', '')}
"""
        
        if article_content:
            research_summary += f"\nARTICLE EXCERPT:\n{article_content[:1000]}"
        
        if key_points:
            research_summary += "\n\nKEY POINTS:\n"
            for point in key_points[:5]:
                research_summary += f"- {point}\n"
        
        if top_comments:
            research_summary += "\n\nCOMMUNITY INSIGHTS:\n"
            for comment in top_comments[:5]:
                research_summary += f"- {comment.get('text', '')[:200]}\n"
        
        # Generate thesis
        thesis_prompt = f"""You are a content strategist analyzing research to find the MOST compelling angle for a short-form video.

{research_summary}

STYLE: {style}
TONE: {tone}

Your task: Analyze this research and identify ONE specific, unique angle that will make this video stand out.

IMPORTANT: Focus on the CONTENT and STORY, not the fact that this is from a post or article.
Don't mention: "this post", "the article says", "according to sources", social media metrics, etc.
DO mention: the actual people, events, facts, and insights from the research.

Good angles:
- Educational: "The real reason why X happened that nobody talks about"
- Humorous: "The absurd situation that everyone missed in this story"
- Dramatic: "The hidden consequence of X that changes everything"
- Inspirational: "What we can learn from X's unexpected success"
- Controversial: "Why everyone is wrong about X"

Bad angles:
- Generic summaries ("Let me tell you about X")
- Obvious takes ("X is interesting because...")
- Vague statements ("This will change everything")

Based on the research above, what is the ONE specific angle/thesis that will make this video valuable?

Respond with ONLY the thesis/angle in 1-2 sentences. Be specific and make it compelling.

Thesis:"""
        
        response = self._call_llm(thesis_prompt)
        return response.strip()
    
    def _create_prompt(self, topic: Dict, duration: int, style: str, tone: str, labeled_media: List = None, thesis: str = "") -> str:
        """Create the prompt for the LLM."""
        
        # Calculate word count (average speaking rate: 150 words/minute)
        target_words = int((duration / 60) * 150)
        
        # Build media context if available
        media_context = ""
        if labeled_media:
            media_context = "\n\nAVAILABLE MEDIA FROM ARTICLE:\n"
            for i, media in enumerate(labeled_media):
                media_dict = media.to_dict() if hasattr(media, 'to_dict') else media
                media_context += f"\nMEDIA {i+1} ({media_dict.get('media_type', 'unknown')}):\n"
                media_context += f"  Description: {media_dict.get('description', 'No description')[:200]}\n"
                media_context += f"  Keywords: {', '.join(media_dict.get('keywords', [])[:5])}\n"
                
                if media_dict.get('media_type') == 'video':
                    media_context += f"  Duration: {media_dict.get('duration', 0):.1f}s\n"
                    if media_dict.get('clip_worthy_segments'):
                        media_context += f"  Clip-worthy segments: {len(media_dict.get('clip_worthy_segments', []))}\n"
        
        # Style-specific instructions
        style_instructions = {
            'engaging': 'Make it captivating with pattern interrupts, surprising facts, and curiosity gaps. Conversational but punchy.',
            'dramatic': 'Build tension and suspense. Use contrast, stakes, and emotional weight. Cinematic storytelling.',
            'educational': 'Teach something valuable. Clear explanations, practical examples, actionable takeaways. Make complex simple.',
            'humorous': 'Make them laugh. Witty observations, unexpected twists, self-aware humor. Natural comedy, not forced jokes. Think jacksfilms - quick wit, sarcasm, absurdist takes.',
            'inspirational': 'Move and motivate. Real stories, genuine emotion, aspirational but authentic. No empty platitudes.',
            'controversial': 'Challenge assumptions. Take a strong stance. Present contrarian views with solid reasoning.',
            'storytelling': 'Narrative arc with characters, conflict, and resolution. Make them feel invested in the outcome.'
        }
        
        style_guide = style_instructions.get(style.lower(), style_instructions['engaging'])
        
        # Extract rich research context
        topic_metadata = topic.get('metadata', {})
        article_content = topic_metadata.get('article_content', '')
        key_points = topic_metadata.get('key_points', [])
        
        # Build research context section
        research_context = ""
        if article_content:
            research_context += f"\n\nDETAILED ARTICLE CONTENT:\n{article_content[:1500]}"  # First 1500 chars
        
        if key_points:
            research_context += "\n\nKEY POINTS FROM ARTICLE:\n"
            for i, point in enumerate(key_points[:5], 1):
                research_context += f"{i}. {point}\n"
        
        # Add top comments for community perspective (but don't reference the post itself)
        top_comments = topic_metadata.get('top_comments', [])
        if top_comments:
            research_context += "\n\nCOMMUNITY INSIGHTS & PERSPECTIVES:\n"
            for comment in top_comments[:3]:
                research_context += f"- {comment.get('text', '')[:200]}\n"
        
        prompt = f"""You are a top-tier short-form content creator. Write a {duration}-second video script that people ACTUALLY want to watch.

TOPIC: {topic.get('title', '')}
DESCRIPTION: {topic.get('description', '')}{research_context}{media_context}

🎯 YOUR UNIQUE ANGLE/THESIS:
{thesis}

This is your creative direction. The ENTIRE script must support and deliver on this thesis.
Don't just summarize the topic - BUILD YOUR SCRIPT AROUND THIS SPECIFIC ANGLE.

⚠️ CRITICAL: Focus on the STORY/CONTENT, not the SOURCE ⚠️
DO NOT mention:
- "This post/article/source says..."
- Social media metrics (likes, upvotes, comments, shares)
- "According to...", "The author states...", "People are discussing..."
- References to Reddit, news sites, or where you found this

INSTEAD, present the information directly:
- BAD: "A Reddit post revealed that Netflix canceled the deal..."
- GOOD: "Netflix just canceled their Warner Bros deal..."
- BAD: "This viral story with 10k upvotes shows..."
- GOOD: "Here's what actually happened..."

Your viewer doesn't care WHERE you found this. They care about the STORY.

VOICE & DELIVERY (CRITICAL):
Write like a REAL PERSON talking, not a content robot. Think jacksfilms/YouTube commentary style:
- Quick, punchy observations ("And here's where it gets weird...")
- Self-aware asides ("Yeah, you heard that right")
- Casual but sharp ("Dude literally...", "So apparently...")
- Slight sarcasm when appropriate ("Because that worked out great")
- Direct address to viewer ("You know what's wild?", "Here's the thing...")
- Natural speech patterns with personality
- Sound like someone EXCITED to tell this story, not reading a script
- Use Gen Alpha/Gen Z slang naturally where it fits:
  * "lowkey/highkey" ("This is lowkey insane")
  * "no cap" ("Netflix really said no cap and switched deals")
  * "fr/fr fr" ("This happened fr fr")
  * "ngl" ("Ngl, this is wild")
  * "cinema/kino" for epic moments ("That's actual cinema")
  * "cooked" ("Dude is cooked", "They're so cooked")
  * "ate/eating" ("She ate with that response")
  * "it's giving..." ("It's giving main character energy")
  * "rent free" ("This lives in my head rent free")
  * BUT: Don't force it - use naturally, not every sentence

DO NOT sound like:
- A documentary narrator
- A news anchor
- An AI content farm
- A corporate presentation
- A Wikipedia article being read aloud
- Someone who learned slang from a textbook

YOUR MISSION:
Every word should work toward delivering on your thesis. Ask yourself:
- "Does this sentence advance my specific angle?"
- "Am I teaching something specific, revealing something hidden, or making someone laugh?"
- "Would a viewer who ONLY cares about my thesis find value in this line?"

If the answer is no, cut it. Your goal is NOT to inform people about the topic generally.
Your goal is to deliver EXACTLY what your thesis promises - nothing more, nothing less.

STYLE MANDATE - {style.upper()}:
{style_guide}

NON-NEGOTIABLE RULES:
✅ DO:
- **DELIVER ON YOUR THESIS** - Every line should advance your specific angle
- **SOUND LIKE A HUMAN** - Use natural speech, asides, self-aware comments
- Start with a PUNCH that sets up your thesis (question, shocking stat, bold claim, or "what if" scenario)
- Use SPECIFIC details from the research above (exact numbers, names, quotes, facts)
- Mine the article content for surprising angles and lesser-known details
- Use concrete examples and real events from the research
- Build toward a payoff that validates your thesis
- Write how people ACTUALLY talk - contractions, fragments, casual asides
- Add personality: slight sarcasm, excitement, disbelief when appropriate
- Direct address to viewer ("You know what's crazy?", "Get this...")
- Every sentence must earn its place - cut ruthlessly
- Create curiosity loops ("But here's what they missed..." "And that's when things got weird...")
- End by delivering the insight your thesis promised
- Transform dry facts into compelling narratives using the research details

❌ DON'T:
- NO aimless rambling - every word must advance your thesis
- NO generic topic summaries - focus ONLY on your specific angle
- NO meta-references to posts, articles, sources, or social media
- NO social engagement metrics ("10k upvotes", "viral post", "everyone's talking about")
- NO attribution phrases ("according to", "the article states", "sources say")
- Just tell the story/facts directly - viewers don't care about your research process
- NO corporate speak or buzzwords ("journey", "game-changer", "innovative solution")
- NO obvious filler ("In today's video...", "So without further ado...", "Be sure to...")
- NO generic statements anyone could say ("This is really interesting", "You won't believe...")
- NO explaining that you're explaining something ("I'm going to tell you about...")
- NO asking for likes/subscribes in a boring way
- NO surface-level summaries - dig into the research and find the compelling details
- NO making up facts - ONLY use information from the research above
- NO treating this like a general explainer - it's a SPECIFIC argument/story/lesson
- ZERO AI-sounding fluff - every word must add value

TECHNICAL SPECS:
- {duration} seconds (~{target_words} words MAX - shorter is better)
- Tone: {tone}
- Must be factually accurate based on the research provided
- Use the detailed article content above to find unique angles
- Hook must grab within 2 seconds
- Optimize for watch-through rate (not just clickbait)

CRITICAL: Structure as SEGMENTS where each segment has:
1. The script text for that segment
2. A detailed visual description for the image/video that should accompany it
3. Whether it should be an image or video clip
4. Media assignment: If available media matches this segment, reference it by number (e.g., "MEDIA 1"), otherwise "GENERATE"
5. A Stable Diffusion prompt (ALWAYS include this - used for image generation fallback)
6. A search query for finding stock videos from Pexels (ALWAYS include this)

⚠️ MANDATORY: The FIRST segment (hook) MUST have "media_type": "video" - NO EXCEPTIONS ⚠️

STRUCTURE YOUR RESPONSE AS JSON:
{{
  "segments": [
    {{
      "text": "Opening hook that grabs attention",
      "type": "hook",
      "media_description": "Detailed description of what visual should show during this narration",
      "media_type": "video",  ← MUST BE "video" FOR FIRST SEGMENT
      "media_assignment": "GENERATE",
      "sd_prompt": "cinematic opening shot, dramatic lighting, photorealistic, high detail, 4k",
      "search_query": "Stock video search query for Pexels",
      "duration_estimate": 3
    }},
    {{
      "text": "First main content point",
      "type": "content",
      "media_description": "What should be shown visually",
      "media_type": "video",
      "media_assignment": "MEDIA 1",
      "clip_start": 10.5,
      "clip_end": 20.0,
      "sd_prompt": "fallback image description, photorealistic, cinematic",
      "duration_estimate": 10
    }},
    {{
      "text": "Second main point",
      "type": "content",
      "media_description": "Visual description",
      "media_type": "video",
      "media_assignment": "GENERATE",
      "sd_prompt": "detailed Stable Diffusion prompt, photorealistic, dramatic",
      "search_query": "Stock video search query",
      "duration_estimate": 10
    }},
    {{
      "text": "Call to action and closing",
      "type": "cta",
      "media_description": "Closing visual description",
      "media_type": "video",
      "media_assignment": "GENERATE",
      "sd_prompt": "compelling closing shot, cinematic, inspiring, photorealistic",
      "search_query": "Search query for stock video",
      "duration_estimate": 5
    }}
  ],
  "background_music_mood": "energetic|calm|dramatic|upbeat"
}}

MEDIA & PACING GUIDELINES (READ CAREFULLY):

🚨 ABSOLUTE REQUIREMENT: First segment MUST be "media_type": "video" 🚨
   - If you set it to "image" for the first segment, the script will be rejected
   - Opening hook = video clip = maximum engagement
   - This is the #1 most important rule

Additional media guidelines:
- STRONGLY PREFER "video" clips for all segments (aim for 80%+ video usage)
- Use images only for diagrams, text overlays, or specific visual references that require stillness
- PREFER available media (MEDIA 1, MEDIA 2) when keywords align
- For video clips, specify clip_start/clip_end for the most impactful moments
- ALWAYS include BOTH "sd_prompt" AND "search_query" for every segment (even video segments)
  - sd_prompt is used if AI image generation is needed
  - search_query is used to fetch stock videos/images from Pexels
  - Having both ensures we always have visual content
- Make sd_prompt detailed: style, lighting, mood, composition, photorealistic
- Make search_query concise and specific for stock footage
- Each segment's script must flow naturally when SPOKEN OUT LOUD
- Media descriptions should be SPECIFIC and visually concrete
- Duration estimates should sum to ~{duration} seconds
- Create 4-6 segments depending on content depth

REMEMBER: You're writing for SHORT ATTENTION SPANS. Every second counts. Make it count.

NATURAL vs ROBOTIC DELIVERY:

✅ NATURAL (sounds like a real person): 
"Okay so the guy who designed the Titanic? Yeah, he went down with it. And here's the messed up part - Thomas Andrews KNEW every single flaw in that ship. The watertight compartments that didn't go high enough, the rivets that couldn't handle ice. Dude literally built the thing. So when it started sinking, he didn't even try to save himself. Just walked around directing people to lifeboats, fully knowing he'd designed his own coffin. That's lowkey the most tragic thing ever. And the psychology behind why he stayed? That's the wild part..."

❌ ROBOTIC (sounds like an AI/documentary):
"In today's video, we're going to explore the fascinating and interesting story of the Titanic's designer, Thomas Andrews. This is a compelling tale that demonstrates how things can go wrong. He was involved in the ship's construction and ultimately perished when the vessel sank. Let me explain what happened and why it's important to know about this historical event..."

See the difference?
- First one: Casual ("Okay so", "Yeah", "Dude literally"), uses slang ("lowkey"), sounds excited
- Second one: Formal ("we're going to explore", "compelling tale"), sounds like reading a textbook

ANOTHER EXAMPLE:

✅ NATURAL:
"Netflix just ditched Warner Bros for Paramount. And you know why? Money. Shocking, I know. But here's what makes this actually interesting - they're literally paying MORE for worse content. Paramount's library is... let's just say it's not exactly HBO-level stuff. Ngl, this deal makes zero sense. They're so cooked if this backfires. So why'd they switch? Well..."

❌ ROBOTIC:
"Netflix has made the decision to terminate their partnership with Warner Bros. Discovery in favor of a new agreement with Paramount. This represents a significant shift in the streaming landscape. The reasons behind this decision are multifaceted and warrant examination..."

Be NATURAL. NOT ROBOTIC."""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return the response."""
        
        if self.provider == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        elif self.provider == "local":
            return self._call_local_llm(prompt)
    
    def _load_local_llm(self):
        """Load local LLM model (VRAM-intensive)."""
        if self.local_llm is None and LOCAL_LLM_AVAILABLE:
            print("  🦖 Loading local LLM (this may take a moment)...")
            try:
                # Load quantized model with GPU offloading
                self.local_llm = Llama(
                    model_path=self.local_model_path,
                    n_ctx=4096,  # Context window
                    n_gpu_layers=-1,  # Offload all layers to GPU if possible
                    n_batch=512,
                    verbose=False
                )
                print("  ✅ Local LLM loaded")
            except Exception as e:
                print(f"  ❌ Failed to load local LLM: {e}")
                raise
        return self.local_llm
    
    def _unload_local_llm(self):
        """Unload local LLM to free VRAM."""
        if self.local_llm is not None:
            print("  🗑️ Unloading local LLM to free VRAM...")
            del self.local_llm
            self.local_llm = None
            # Force garbage collection
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("  ✅ VRAM freed")
    
    def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM with automatic loading/unloading."""
        llm = self._load_local_llm()
        
        try:
            # Create a chat-style prompt for instruct models
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            response = llm(
                formatted_prompt,
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        finally:
            # Always unload after use to free VRAM for SD
            self._unload_local_llm()
    
    def _call_llm_with_retry(self, prompt: str, topic: Dict, duration: int, style: str, tone: str) -> str:
        """Call LLM with retry mechanism for refusals."""
        
        # First attempt with main provider
        response = self._call_llm(prompt)
        
        # Check if LLM refused (common patterns)
        refusal_indicators = [
            "i'm sorry",
            "i cannot",
            "i can't",
            "i apologize",
            "i'm unable",
            "against my",
            "policy",
            "inappropriate",
            "i don't feel comfortable",
            "i can't assist",
            "i won't"
        ]
        
        response_start = response.lower()[:300]  # Check first 300 chars
        is_refusal = any(indicator in response_start for indicator in refusal_indicators)
        
        # Also check if response is suspiciously short (likely a refusal)
        if not is_refusal and len(response.strip()) < 100:
            is_refusal = True
        
        if is_refusal:
            print(f"  ⚠️  {self.provider.upper()} LLM refused the request")
            print("  Reason detected:", response[:200] if len(response) < 200 else response[:200] + "...")
            
            # Try with local LLM if available
            if LOCAL_LLM_AVAILABLE and self.local_model_path and os.path.exists(self.local_model_path):
                print("  🔄 Retrying with local uncensored LLM...")
                try:
                    # Create simplified prompt for local model
                    local_prompt = self._create_local_llm_prompt(topic, duration, style, tone)
                    response = self._call_local_llm(local_prompt)
                    
                    # Check if local model also refused
                    if any(indicator in response.lower()[:300] for indicator in refusal_indicators):
                        print("  ❌ Local LLM also refused")
                    else:
                        print("  ✅ Local LLM generated script successfully")
                        return response
                        
                except Exception as e:
                    print(f"  ❌ Local LLM failed: {e}")
            else:
                if not LOCAL_LLM_AVAILABLE:
                    print("  💡 Install llama-cpp-python for local LLM fallback: pip install llama-cpp-python")
                elif not self.local_model_path:
                    print("  💡 Set LOCAL_LLM_PATH in .env to use local LLM fallback")
            
            print("  🛑 No fallback available, using generic script")
        
        return response
    
    def _call_llm_with_retry(self, prompt: str, topic: Dict, duration: int, style: str, tone: str) -> str:
        """Call LLM with retry mechanism for refusals."""
        
        # First attempt
        response = self._call_llm(prompt)
        
        # Check if LLM refused (common patterns)
        refusal_indicators = [
            "i'm sorry",
            "i cannot",
            "i can't",
            "i apologize",
            "i'm unable",
            "against my",
            "policy",
            "inappropriate"
        ]
        
        if any(indicator in response.lower()[:200] for indicator in refusal_indicators):
            print("  \u26a0\ufe0f  LLM refused initial request, retrying with clarified prompt...")
            
            # Create a clarified prompt emphasizing this is educational/commentary
            retry_prompt = self._create_retry_prompt(topic, duration, style, tone)
            response = self._call_llm(retry_prompt)
            
            # Check again
            if any(indicator in response.lower()[:200] for indicator in refusal_indicators):
                print("  \u274c LLM refused again. Topic may involve sensitive content.")
                # Return refusal so parser can handle it properly
        
        return response
    
    def _create_retry_prompt(self, topic: Dict, duration: int, style: str, tone: str) -> str:
        """Create a simplified retry prompt that emphasizes fair use."""
        target_words = int((duration / 60) * 150)
        
        topic_metadata = topic.get('metadata', {})
        article_content = topic_metadata.get('article_content', '')
        key_points = topic_metadata.get('key_points', [])
        
        prompt = f"""Create an educational video script analyzing this topic:

TOPIC: {topic.get('title', '')}
DESCRIPTION: {topic.get('description', '')}

"""        
        if article_content:
            prompt += f"CONTEXT:\n{article_content[:800]}\n\n"
        
        if key_points:
            prompt += "KEY POINTS:\n"
            for point in key_points[:3]:
                prompt += f"- {point}\n"
            prompt += "\n"
        
        prompt += f"""This is for educational/commentary purposes (fair use). Create a {duration}-second video script in a {style} style with {tone} tone.

IMPORTANT:
- Focus on ANALYSIS and COMMENTARY, not reproduction of copyrighted material
- Discuss concepts, facts, and public information
- Add educational value through your unique perspective
- Make it engaging for viewers (~{target_words} words)
- First segment MUST be a video clip for the hook

Return JSON format:
{{
  "segments": [
    {{
      "text": "Hook text here",
      "type": "hook",
      "media_description": "Visual description",
      "media_type": "video",
      "media_assignment": "GENERATE",
      "sd_prompt": "image generation prompt",
      "search_query": "stock video query",
      "duration_estimate": 3
    }},
    ...
  ],
  "background_music_mood": "upbeat"
}}
"""
        return prompt
    
    def _create_local_llm_prompt(self, topic: Dict, duration: int, style: str, tone: str) -> str:
        """Create a simplified prompt for local 8B models."""
        target_words = int((duration / 60) * 150)
        
        topic_metadata = topic.get('metadata', {})
        article_content = topic_metadata.get('article_content', '')
        key_points = topic_metadata.get('key_points', [])
        
        prompt = f"""Create a {duration}-second video script about this topic.

TOPIC: {topic.get('title', '')}
DESCRIPTION: {topic.get('description', '')}
"""
        
        if article_content:
            prompt += f"\\nCONTEXT:\\n{article_content[:600]}\\n"
        
        if key_points:
            prompt += "\\nKEY POINTS:\\n"
            for i, point in enumerate(key_points[:3], 1):
                prompt += f"{i}. {point}\\n"
        
        prompt += f"""\\nGuidelines:
- Write ~{target_words} words
- Style: {style}, Tone: {tone}
- Start with an attention hook
- Explain using the context
- End with call to action
- Sound natural
- First segment MUST be video

JSON format:
{{
  "segments": [
    {{ "text": "Hook", "type": "hook", "media_description": "Visual", "media_type": "video", "media_assignment": "GENERATE", "sd_prompt": "cinematic", "search_query": "video", "duration_estimate": 3 }},
    {{ "text": "Content", "type": "content", "media_description": "Visual", "media_type": "video", "media_assignment": "GENERATE", "sd_prompt": "image", "search_query": "video", "duration_estimate": 15 }},
    {{ "text": "CTA", "type": "cta", "media_description": "Visual", "media_type": "image", "media_assignment": "GENERATE", "sd_prompt": "cta", "search_query": "social", "duration_estimate": 4 }}
  ],
  "background_music_mood": "upbeat"
}}"""
        
        return prompt
    
    def _parse_script_response(self, response: str, topic: Dict, duration: int, labeled_media: List = None) -> VideoScript:
        """Parse the LLM response into a VideoScript object."""
        
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            # Parse segments
            segments = []
            full_script_parts = []
            
            for seg_data in data.get('segments', []):
                # Parse media assignment and resolve to actual media path
                media_assignment = seg_data.get('media_assignment', 'GENERATE')
                assigned_media_path = None
                
                # Resolve media assignment if it references labeled media
                if labeled_media and media_assignment and media_assignment.startswith('MEDIA'):
                    try:
                        media_idx = int(media_assignment.split()[1]) - 1  # "MEDIA 1" -> index 0
                        if 0 <= media_idx < len(labeled_media):
                            media = labeled_media[media_idx]
                            media_dict = media.to_dict() if hasattr(media, 'to_dict') else media
                            assigned_media_path = media_dict.get('local_path')
                    except (IndexError, ValueError):
                        pass  # Could not parse media reference
                
                segment = ScriptSegment(
                    text=seg_data['text'],
                    type=seg_data.get('type', 'content'),
                    media_description=seg_data.get('media_description', ''),
                    media_type=seg_data.get('media_type', 'image'),
                    media_assignment=media_assignment,
                    assigned_media_path=assigned_media_path,
                    clip_start=seg_data.get('clip_start'),
                    clip_end=seg_data.get('clip_end'),
                    sd_prompt=seg_data.get('sd_prompt'),
                    search_query=seg_data.get('search_query'),
                    duration_estimate=seg_data.get('duration_estimate')
                )
                segments.append(segment)
                full_script_parts.append(segment.text)
            
            # CRITICAL: Enforce that first segment MUST be video
            if segments and segments[0].media_type != 'video':
                print(f"⚠️ First segment was '{segments[0].media_type}', correcting to 'video' (required for engagement)")
                segments[0].media_type = 'video'
                # If it had an SD prompt but no search query, create one
                if segments[0].sd_prompt and not segments[0].search_query:
                    segments[0].search_query = segments[0].media_description[:100]
            
            # Build full script
            full_script = " ".join(full_script_parts)
            
            # Estimate actual duration (150 words/minute average)
            word_count = len(full_script.split())
            estimated_duration = int((word_count / 150) * 60)
            
            return VideoScript(
                segments=segments,
                full_script=full_script,
                estimated_duration=estimated_duration,
                metadata={
                    "topic": topic.get('title', ''),
                    "source": topic.get('source', ''),
                    "target_duration": duration,
                    "word_count": word_count,
                    "background_music_mood": data.get('background_music_mood', 'upbeat'),
                    "topic_full": topic,  # Store full topic for media extraction
                    "num_segments": len(segments),
                    "labeled_media_count": len(labeled_media) if labeled_media else 0
                }
            )
        
        except Exception as e:
            print(f"⚠️ Error parsing script response: {e}")
            print(f"Raw response: {response[:500]}...")
            print(f"⚠️ Using fallback script - quality may be reduced")
            
            # Fallback: create basic script
            return self._create_fallback_script(topic, duration)
    
    def _create_fallback_script(self, topic: Dict, duration: int) -> VideoScript:
        """Create a basic fallback script if parsing fails."""
        
        title = topic.get('title', 'Unknown Topic')
        description = topic.get('description', '')
        
        # Extract some context from metadata if available
        topic_metadata = topic.get('metadata', {})
        key_points = topic_metadata.get('key_points', [])
        article_content = topic_metadata.get('article_content', '')
        
        # Try to build a meaningful script from available data
        hook_text = f"Let's talk about {title}."
        
        content_text = description[:300] if description else "This is an interesting development."
        
        # Add key points if available
        if key_points:
            content_text = " ".join(key_points[:2])[:300]
        elif article_content:
            # Use first few sentences of article
            sentences = article_content.split('. ')[:3]
            content_text = '. '.join(sentences)[:300]
        
        cta_text = "That's what's happening. Follow for more updates."
        
        # Create basic segments
        segments = [
            ScriptSegment(
                text=hook_text,
                type="hook",
                media_description=f"Visual related to {title}",
                media_type="video",  # ALWAYS video for hook
                search_query=title[:100],
                sd_prompt=f"{title}, cinematic, photorealistic, dramatic lighting",
                duration_estimate=3
            ),
            ScriptSegment(
                text=content_text,
                type="content",
                media_description="Relevant visual showing the topic",
                media_type="video",  # Prefer video
                search_query=f"{title}",
                sd_prompt=f"{title}, news, informative, photorealistic",
                duration_estimate=12
            ),
            ScriptSegment(
                text=cta_text,
                type="cta",
                media_description="Call to action visual",
                media_type="image",
                sd_prompt="social media notification, modern ui, vibrant colors",
                search_query="social media like subscribe",
                duration_estimate=4
            )
        ]
        
        full_script = " ".join([s.text for s in segments])
        word_count = len(full_script.split())
        
        return VideoScript(
            segments=segments,
            full_script=full_script,
            estimated_duration=int((word_count / 150) * 60),
            metadata={
                "topic": title,
                "source": topic.get('source', ''),
                "target_duration": duration,
                "word_count": word_count,
                "fallback": True,
                "num_segments": len(segments)
            }
        )
    
    def refine_script(self, script: VideoScript, feedback: str) -> VideoScript:
        """
        Refine an existing script based on feedback.
        
        Args:
            script: Original VideoScript
            feedback: Human feedback or refinement instructions
            
        Returns:
            Refined VideoScript
        """
        # Format current segments for the prompt
        segments_text = "\n".join([
            f"  {i+1}. [{s.type}] {s.text}\n     Media: {s.media_description}"
            for i, s in enumerate(script.segments)
        ])
        
        prompt = f"""Refine the following video script based on this feedback:

FEEDBACK: {feedback}

ORIGINAL SCRIPT SEGMENTS:
{segments_text}

Provide the refined version in the same segmented JSON format with media descriptions for each segment."""

        response = self._call_llm(prompt)
        
        # Reuse topic metadata from original script
        topic = {
            "title": script.metadata.get("topic", ""),
            "source": script.metadata.get("source", "")
        }
        
        return self._parse_script_response(response, topic, script.metadata.get("target_duration", 45))
    
    def save_script(self, script: VideoScript, filepath: str):
        """Save script to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(script.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"💾 Script saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Example topic
    topic = {
        "title": "AI Breakthrough: New Model Achieves Human-Level Reasoning",
        "description": "Researchers announce a new AI model that can solve complex problems with human-like reasoning capabilities.",
        "source": "tech_news",
        "viral_potential": 0.85
    }
    
    # Create content
    creator = ContentCreator(provider="anthropic")
    script = creator.create_script(topic, duration=45, style="engaging", tone="excited")
    
    # Display result
    print("\n📝 Generated Script:\n")
    print(f"Total Segments: {len(script.segments)}\n")
    for i, segment in enumerate(script.segments, 1):
        print(f"{i}. [{segment.type.upper()}] {segment.text}")
        print(f"   Media: {segment.media_description}")
        print(f"   Type: {segment.media_type} | Duration: ~{segment.duration_estimate}s")
        print()
    print(f"Total Duration: ~{script.estimated_duration}s")
    print(f"Word count: {script.metadata['word_count']}")
    
    # Save
    creator.save_script(script, "example_script.json")
