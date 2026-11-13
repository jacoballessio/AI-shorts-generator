"""
VLM (Vision-Language Model) analyzer for frame understanding
"""

import os
import re
import base64
import warnings
from io import BytesIO
from typing import Optional, Literal, List, Dict, Tuple
import numpy as np
from PIL import Image

from .models import FrameAnalysis


# qwen_omni_utils may emit this upstream librosa deprecation warning repeatedly.
warnings.filterwarnings(
    "ignore",
    message=r".*librosa\.core\.audio\.__audioread_load.*",
    category=FutureWarning,
)


class VLMAnalyzer:
    """Analyzes video frames using Vision-Language Models."""
    
    def __init__(
        self,
        provider: Literal["openai", "anthropic", "local"] = "openai",
        model: str = "gpt-4o",
        smolvlm_4bit: bool = False,
        asid_max_new_tokens: int = 512,
    ):
        """
        Initialize VLM analyzer.
        
        Args:
            provider: VLM provider (openai, anthropic, or local)
            model: Model name (for local: "HuggingFaceTB/SmolVLM-Instruct")
        """
        self.provider = provider
        self.model = model
        self.client = None
        self.processor = None
        self.local_model = None
        self.is_asid = False
        self.smolvlm_4bit = smolvlm_4bit
        self.asid_max_new_tokens = asid_max_new_tokens
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "local":
            self._init_local_vlm(model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_local_vlm(self, model_name: str):
        """Initialize local VLM (SmolVLM or ASID-Captioner)."""
        # Default to ASID if no model specified
        if not model_name or model_name == "local":
            model_name = "AudioVisual-Caption/ASID-Captioner-3B"

        # Route to ASID-Captioner (Qwen2.5-Omni based)
        if 'ASID' in model_name or 'AudioVisual-Caption' in model_name:
            self._init_asid_model(model_name)
            return

        try:
            import torch
            from transformers import AutoProcessor, Idefics3ForConditionalGeneration
            
            print(f"   Loading local VLM: {model_name}")
            print("   This may take a few minutes on first run...")
            
            if not model_name or model_name == "local":
                model_name = "HuggingFaceTB/SmolVLM-Instruct"
            
            # Debug: Check PyTorch and CUDA info
            print(f"   PyTorch version: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device}")
            
            if device == "cpu" and "+cpu" in torch.__version__:
                print(f"   [WARNING] You have CPU-only PyTorch installed!")
                print(f"   [WARNING] Install CUDA version for GPU support:")
                print(f"   [WARNING] pip uninstall torch torchvision")
                print(f"   [WARNING] pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            
            # Load model and processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            model_kwargs = {
                "trust_remote_code": True,
                "_attn_implementation": "eager",
            }
            if device == "cuda" and self.smolvlm_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model_kwargs["device_map"] = "auto"
                    print("   Using 4-bit quantization for SmolVLM")
                except Exception as e:
                    print(f"   [WARNING] Could not enable 4-bit quantization ({e}); falling back to fp16")
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
                model_kwargs["device_map"] = "auto" if device == "cuda" else None

            self.local_model = Idefics3ForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs,
            )
            
            if device == "cpu":
                self.local_model = self.local_model.to(device)
            
            self.model = model_name
            print(f"   [OK] Local VLM loaded successfully")
            
        except ImportError as e:
            print(f"[ERROR] Error loading local VLM: {e}")
            print("   Install dependencies: pip install torch transformers accelerate")
            raise
        except Exception as e:
            print(f"[ERROR] Error initializing local VLM: {e}")
            raise

    def _init_asid_model(self, model_name: str):
        """Initialize ASID-Captioner-3B (Qwen2.5-Omni thinker-only audiovisual model)."""
        try:
            # Force stable backend to avoid noisy torchcodec load failures.
            os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
            import torch
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
            import transformers.tokenization_utils_base as _tok_base

            print(f"   Loading ASID captioner: {model_name}")
            print(f"   PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")

            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.local_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    device_map=device_map,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                )
            except Exception:
                # Fall back if flash_attention_2 not available
                self.local_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )

            # Newer transformers versions may try to fetch
            # /additional_chat_templates and fail hard when it doesn't exist.
            # Treat missing template directories as "no extra templates".
            orig_list_repo_templates = _tok_base.list_repo_templates

            def _safe_list_repo_templates(*args, **kwargs):
                try:
                    return orig_list_repo_templates(*args, **kwargs)
                except Exception as e:
                    msg = str(e)
                    if "additional_chat_templates" in msg or "RemoteEntryNotFoundError" in msg:
                        return []
                    raise

            try:
                _tok_base.list_repo_templates = _safe_list_repo_templates
                self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
            finally:
                _tok_base.list_repo_templates = orig_list_repo_templates

            self.model = model_name
            self.is_asid = True
            print(f"   [OK] ASID captioner loaded")

        except ImportError as e:
            print(f"[ERROR] {e}")
            print("   pip install transformers accelerate qwen-omni-utils")
            raise
        except Exception as e:
            print(f"[ERROR] Error initializing ASID model: {e}")
            raise

    def _asid_infer(self, tmp_path: str, prompt: str) -> str:
        """Run a single ASID inference call on a temp clip file. Returns assistant reply."""
        import torch
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError("Install qwen-omni-utils: pip install qwen-omni-utils")

        VIDEO_MAX_PIXELS = 401408
        USE_AUDIO = True

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                )}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": tmp_path, "max_pixels": VIDEO_MAX_PIXELS},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device).to(self.local_model.dtype)

        with torch.no_grad():
            text_ids = self.local_model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO,
                do_sample=False,
                max_new_tokens=self.asid_max_new_tokens,
                repetition_penalty=1.1,
                use_cache=True,
            )

        decoded = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.split("\nassistant\n")[-1].strip()

    def analyze_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        transcript_context: str = "",
        extract_ui: bool = True,
    ) -> tuple:
        """
        Analyze a video segment with ASID-Captioner-3B using two separate calls:
        one for scene description, one for visible UI/text overlays.
        Returns (scene_description, ui_html, ui_elements).
        """
        import os
        import tempfile
        from moviepy import VideoFileClip

        tmp_path = None
        try:
            clip = VideoFileClip(video_path)
            sub = clip.subclipped(max(0, start_time), min(end_time, clip.duration))
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name
            sub.write_videofile(tmp_path, audio_codec='aac', logger=None)
            clip.close()

            # ── Call 1: scene description ─────────────────────────────────
            scene_prompt = (
                "Describe the PHYSICAL SCENE you see in this video clip.\n\n"
                "INCLUDE:\n"
                "- People: their appearance, clothing, facial expressions, actions\n"
                "- Environment: location, setting, background\n"
                "- Objects and what is happening\n"
                "- Camera framing (close-up, wide shot, etc.)\n\n"
                "DO NOT INCLUDE:\n"
                "- Any text, words, or numbers you see written/printed on the screen\n"
                "- Title cards, captions, watermarks, overlaid graphics\n"
                "- Audio, music, or spoken words\n\n"
                "Write one or two plain sentences describing only the physical visual world."
            )
            scene_desc = self._asid_infer(tmp_path, scene_prompt)

            if extract_ui:
                # ── Call 2: visible text / UI overlays ────────────────────
                ui_prompt = (
                    "Look at this video clip and find all TEXT and GRAPHICS that are displayed "
                    "as overlays on top of the footage.\n\n"
                    "INCLUDE:\n"
                    "- Title cards (e.g. 'Tour of Blue Claus\\'s Workshop')\n"
                    "- Captions and subtitles\n"
                    "- Usernames, handles, hashtags\n"
                    "- Timestamps and view counts\n"
                    "- Any other words or numbers visually printed on screen\n\n"
                    "DO NOT INCLUDE:\n"
                    "- Descriptions of people, objects, or the physical scene\n"
                    "- Audio or spoken dialogue that is NOT shown as text on screen\n"
                    "- Anything you are inferring — only text you can literally read\n\n"
                    "For each overlay, output one <div> with inline CSS:\n"
                    "<div style=\"position:absolute; top:10%; left:5%; color:white; font-size:40px; font-weight:bold\">Exact text here</div>\n\n"
                    "If there are NO visible text overlays, output exactly: (empty)"
                )
                ui_raw = self._asid_infer(tmp_path, ui_prompt)
                # Treat the explicit empty sentinel as blank
                if ui_raw.strip().lower() in ("(empty)", "empty", "none", ""):
                    ui_html = ""
                else:
                    ui_html = self._parse_html_response(ui_raw)
            else:
                ui_html = ""

            return scene_desc, ui_html, []

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        transcript_context: str = "",
        analyze_actions: bool = True,
        analyze_objects: bool = True,
        previous_frame_description: str = ""
    ) -> FrameAnalysis:
        """
        Analyze a single frame with VLM.
        
        Args:
            frame: Frame image as numpy array
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
            transcript_context: Nearby transcript for context
            analyze_actions: Whether to detect actions
            analyze_objects: Whether to detect objects
            previous_frame_description: Description of previous frame for continuity
            
        Returns:
            FrameAnalysis object
        """
        # Convert frame to base64 for API
        image_b64 = self._frame_to_base64(frame)
        
        # Create prompt
        prompt = self._create_analysis_prompt(
            transcript_context,
            analyze_actions,
            analyze_objects,
            previous_frame_description
        )
        
        # Call VLM
        response = self._call_vlm(image_b64, prompt)
        
        # Parse response into FrameAnalysis
        analysis = self._parse_vlm_response(response, frame_number, timestamp)
        
        return analysis
    
    def analyze_frame_combined(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        transcript_context: str = ""
    ):
        """
        Single combined VLM call that returns both scene description and UI HTML.
        This is the fast path used by segment_based_analysis — one call per segment
        instead of two calls per frame.

        Returns:
            Tuple[str, str, list]: (scene_description, ui_html, ui_elements)
        """
        image_b64 = self._frame_to_base64(frame)
        prompt = self._create_combined_prompt(transcript_context)
        response = self._call_vlm(image_b64, prompt)
        return self._parse_combined_response(response, transcript_context)

    def _create_combined_prompt(self, transcript_context: str = "") -> str:
        """Prompt that returns scene_description + ui_elements_html in one JSON call."""
        prompt = "Look at the image. Return a JSON object with exactly three keys.\n\n"

        prompt += '"scene_description": one sentence describing the PHYSICAL scene you see.\n'
        prompt += "- Who is in the frame, what they look like, what they are doing\n"
        prompt += "- Location / setting\n"
        prompt += "- Camera angle (close-up, wide shot, etc.)\n"
        prompt += "- Do NOT copy text overlays into this field\n\n"

        prompt += '"ui_elements_html": HTML string of text/graphics overlaid ON TOP of the footage.\n'
        prompt += "Use one <div> per element with inline CSS:\n"
        prompt += "  position:absolute; top:<Y%>; left:<X%>  (estimate from top-left corner)\n"
        prompt += "  color:<css-color>  (e.g. white, black, #FFD700)\n"
        prompt += "  font-size:<Npx>  (small ~18px, medium ~32px, large title ~56px)\n"
        prompt += "  font-weight:bold or normal\n"
        prompt += "  font-family:sans-serif | serif | 'Impact' | cursive\n"
        prompt += "  text-align:left | center | right\n"
        prompt += "  text-shadow:2px 2px 4px black  (if text has visible shadow/outline)\n"
        prompt += "  background:rgba(0,0,0,0.5)  (if text has a background box)\n"
        prompt += "Only include text you can actually READ in this specific image.\n"
        prompt += "Do NOT include audio or sounds.\n"
        prompt += 'Return empty string \"\" if there are no overlays.\n\n'

        prompt += '"ui_elements": array, one object per overlay:\n'
        prompt += '  {"content":"<exact text>","top_pct":<0-100>,"left_pct":<0-100>,'
        prompt += '"color":"<css>","font_size_px":<N>,"bold":<true|false>,'
        prompt += '"font_family":"<family>","text_align":"<align>","has_shadow":<true|false>,"has_bg":<true|false>}\n'
        prompt += "Return [] if nothing visible.\n\n"

        if transcript_context:
            prompt += f'Audio transcript (do NOT copy into any field): "{transcript_context}"\n\n'

        prompt += "Return only a raw JSON object with exactly these keys: scene_description, ui_elements_html, ui_elements. No markdown, no code fences, no explanation."
        return prompt

    def _parse_combined_response(self, response: str, transcript_context: str = ""):
        """Parse the combined JSON response into (scene_description, ui_html, ui_elements)."""
        import json as _json
        import re as _re

        try:
            # Strip code fences
            text = _re.sub(r'```(?:json)?\s*', '', response or '').strip()

            # Extract first {...} block
            m = _re.search(r'\{.*\}', text, _re.DOTALL)
            if m:
                text = m.group(0)

            data = _json.loads(text)
            scene_desc = self._clean_vlm_description(data.get('scene_description', ''))

            # Reject scene_desc if the model merely echoed the transcript
            if transcript_context and scene_desc:
                def _norm(s): return _re.sub(r'[^a-z0-9]', '', s.lower())
                if _norm(scene_desc) == _norm(transcript_context):
                    scene_desc = 'Unknown scene'

            ui_html = self._parse_html_response(data.get('ui_elements_html', ''))
            ui_elements = data.get('ui_elements', [])
            if not isinstance(ui_elements, list):
                ui_elements = []
            # Normalise ui_elements: accept both old {content, position, style} and new schema
            normalised = []
            for e in ui_elements:
                if not isinstance(e, dict):
                    continue
                content = e.get('content', '')
                if not content:
                    continue
                # Build a canonical element with all visual fields
                elem = {
                    'content': content,
                    'top_pct': e.get('top_pct', e.get('top', 50)),
                    'left_pct': e.get('left_pct', e.get('left', 50)),
                    'color': e.get('color', 'white'),
                    'font_size_px': e.get('font_size_px', e.get('font_size', 32)),
                    'bold': e.get('bold', False),
                    'font_family': e.get('font_family', 'sans-serif'),
                    'text_align': e.get('text_align', 'center'),
                    'has_shadow': e.get('has_shadow', False),
                    'has_bg': e.get('has_bg', False),
                }
                if not self._is_invalid_ui_element({'content': content, 'type': 'text'}, scene_desc):
                    normalised.append(elem)
            return scene_desc, ui_html, normalised
        except Exception:
            return response.strip()[:200] if response else 'Unknown scene', '', []

    def analyze_segment(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        timestamps: List[float],
        segment_start: float,
        segment_end: float,
        transcript_context: str = "",
        previous_segment_description: str = ""
    ) -> dict:
        """
        Analyze a video segment using multiple representative frames.
        
        Args:
            frames: List of representative frame images from the segment
            frame_numbers: Corresponding frame numbers
            timestamps: Corresponding timestamps
            segment_start: Segment start time in seconds
            segment_end: Segment end time in seconds
            transcript_context: Transcript content for this segment
            previous_segment_description: Description of previous segment for continuity
            
        Returns:
            Dictionary with segment analysis results
        """
        if not frames:
            return {
                'scene_description': 'No visual data available',
                'ui_elements': [],
                'actions': [],
                'objects': [],
                'scene_type': 'unknown'
            }
        
        # Convert frames to base64
        image_b64_list = [self._frame_to_base64(frame) for frame in frames]
        
        # Create segment analysis prompt
        prompt = self._create_segment_analysis_prompt(
            len(frames),
            transcript_context,
            previous_segment_description
        )
        
        # Call VLM with multiple images
        response = self._call_vlm_multi_image(image_b64_list, prompt)
        
        # Parse response
        analysis = self._parse_segment_vlm_response(response)
        
        return analysis
    
    def generate_ui_html_from_segment(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        timestamps: List[float],
        segment_start: float,
        segment_end: float,
        transcript_context: str = "",
        previous_ui_html: str = ""
    ) -> str:
        """
        Generate HTML for UI elements in a segment using representative frames.
        
        Args:
            frames: List of representative frame images from the segment
            frame_numbers: Corresponding frame numbers
            timestamps: Corresponding timestamps
            segment_start: Segment start time in seconds
            segment_end: Segment end time in seconds
            transcript_context: Transcript content for this segment
            previous_ui_html: HTML from previous segment for continuity
            
        Returns:
            HTML string that recreates UI elements across the segment
        """
        if not frames:
            return ""
        
        # Use the middle frame for UI analysis (most representative)
        mid_idx = len(frames) // 2
        representative_frame = frames[mid_idx]
        
        # Generate UI HTML using existing method
        html = self.generate_ui_html_from_frame(
            representative_frame,
            frame_numbers[mid_idx],
            timestamps[mid_idx],
            transcript_context,
            previous_ui_html
        )
        
        return html
    
    def generate_ui_html_from_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        transcript_context: str = "",
        previous_ui_html: str = ""
    ) -> str:
        """
        Generate HTML that matches the visual style of UI elements in the frame.
        
        Args:
            frame: Frame image as numpy array
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
            transcript_context: Nearby transcript for context
            previous_ui_html: HTML from previous frame for continuity
            
        Returns:
            HTML string that recreates the visual appearance of UI elements
        """
        # Convert frame to base64 for API
        image_b64 = self._frame_to_base64(frame)
        
        # Create prompt for HTML generation
        prompt = self._create_ui_html_prompt(transcript_context, previous_ui_html)
        
        # Call VLM
        response = self._call_vlm(image_b64, prompt)
        
        # Parse and clean HTML response
        html = self._parse_html_response(response)
        
        return html
    
    def _create_ui_html_prompt(self, transcript_context: str = "", previous_ui_html: str = "") -> str:
        """Create the VLM prompt for generating styled HTML with better styling."""
        
        prompt = "TASK: Generate clean, professional HTML that recreates the visual UI elements visible in this video frame.\n\n"
        
        prompt += "VISUAL GUIDELINES:\n"
        prompt += "- Focus on realistic messaging app UI, social media overlays, or video captions\n"
        prompt += "- Include: usernames, timestamps, message bubbles, icons, hashtags, view counts\n"
        prompt += "- STRICTLY EXCLUDE: scene descriptions, audio descriptions, sound effects, speech content\n"
        prompt += "- Keep text SHORT and CLEAN (max 20 chars per element unless it's a caption)\n"
        prompt += "- Use realistic font sizes and spacing\n\n"
        
        if previous_ui_html:
            prompt += f"CONTINUITY: Previous frame had:\n{previous_ui_html[:150]}\n"
            prompt += "Reuse similar elements if the UI hasn't changed. Modernize if it has.\n\n"
        
        prompt += "HTML STRUCTURE:\n"
        prompt += "- Text overlay: <div class=\"pos-[POSITION] style-[STYLE]\">[TEXT]</div>\n"
        prompt += "- Icons/badges: <span class=\"icon pos-[POSITION]\" aria-label=\"[ICON_TYPE]\"></span>\n"
        prompt += "- Use these position classes: pos-top-left, pos-top-right, pos-top, pos-bottom-left, pos-bottom-right, pos-bottom\n"
        prompt += "- Use these style classes: style-white, style-bold, style-yellow, style-blue, style-green\n"
        prompt += "- Add extra classes for visual styling: data-semibold, data-shadow, data-outline\n\n"
        
        prompt += "STYLING HINTS:\n"
        prompt += "- Messaging apps: light backgrounds, dark text, subtle shadows\n"
        prompt += "- Social media: bold text, bright colors, overlays at bottom\n"
        prompt += "- Videos: white/light text on dark backgrounds for legibility\n\n"
        
        if transcript_context:
            prompt += f"DIALOGUE CONTEXT: \"{transcript_context}\"\n"
            prompt += "Use this to make text overlays more contextual.\n\n"
        
        prompt += "OUTPUT: Return ONLY the HTML. No explanation, no code blocks, no markdown.\n"
        prompt += "Keep it minimal but visually appealing.\n\n"
        
        prompt += "EXAMPLES:\n"
        prompt += "<div class=\"pos-top-left style-white\">Sarah</div>\n"
        prompt += "<div class=\"pos-bottom-right style-bold\">Hey!</div>\n"
        prompt += "<div class=\"pos-top style-white\" data-semibold>12:47 PM</div>\n"
        prompt += "<span class=\"icon pos-top-right\" aria-label=\"heart\"></span>\n"
        
        return prompt
    
    def _parse_html_response(self, response: str) -> str:
        """Parse and clean HTML from VLM response."""
        if not response:
            return ""
        
        # Remove code blocks if present
        response = re.sub(r'```html\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Remove any JSON wrapping
        response = re.sub(r'^\s*{\s*"html"\s*:\s*"', '', response)
        response = re.sub(r'"\s*}\s*$', '', response)
        
        # Remove [Sound: ...] markers
        response = re.sub(r'\[Sound:.*?\]', '', response)

        # Reject plain <div>text</div> with no positioning CSS — that's a scene description, not a UI overlay
        if re.match(r'^\s*<div>[^<]*</div>\s*$', response, re.IGNORECASE):
            return ""
        if re.match(r'^\s*<html>[^<]*</html>\s*$', response, re.IGNORECASE):
            return ""
        
        # Only drop a whole <div> line if it contains audio-description content
        # that is NOT inside a style="..." attribute (to avoid nuking CSS color names).
        # Strategy: strip style attributes before checking for audio keywords.
        audio_keywords = ['speech', 'voice', 'talking', 'speaking', 'shout', 'yell',
                          'scream', 'whisper', '[sound', '\\[sound']
        lines = response.split('\n')
        filtered_lines = []
        for line in lines:
            # Remove style="..." portion before checking for audio keywords
            line_no_style = re.sub(r'style="[^"]*"', '', line)
            line_lower = line_no_style.lower()
            if any(keyword in line_lower for keyword in audio_keywords):
                continue
            filtered_lines.append(line)
        response = '\n'.join(filtered_lines)
        
        # Remove lines with excessive repetition (corrupted data)
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if re.search(r'([a-z])\1{5,}', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy frame to base64 string."""
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # Resize if too large (save API costs)
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode()
    
    def _create_analysis_prompt(
        self,
        transcript_context: str,
        analyze_actions: bool,
        analyze_objects: bool,
        previous_frame_description: str = ""
    ) -> str:
        """Create the VLM analysis prompt."""
        
        prompt = "Describe this video frame using JSON format.\n\n"
        
        # Add previous frame context for continuity
        if previous_frame_description:
            prompt += f"Previous frame: {previous_frame_description[:100]}\n"
            prompt += "If same as previous, just say: 'Same shot'\n\n"
        
        prompt += "Provide scene_description (ONLY the main video content, NOT any text overlays):\n"
        prompt += "- Describe what you actually see: camera angle, people, objects, setting, actions\n"
        prompt += "- Be specific: describe camera angle, people, objects, setting, actions\n"
        prompt += "- DO NOT include any text that appears on screen as overlays/captions\n"
        prompt += "- DO NOT describe UI elements, watermarks, or text graphics\n\n"
        
        prompt += "Provide ui_elements (ONLY text/graphics overlaid on the video):\n"
        prompt += "- ONLY include: visible text captions, usernames, timestamps, hashtags, icons\n"
        prompt += "- Must be TEXT OR ICONS that are overlaid on the video content\n"
        prompt += "- DO NOT include scene descriptions or what people/objects are doing\n"
        prompt += "- If no text overlays visible, return empty array []\n\n"
        
        if transcript_context:
            prompt += f"Spoken audio context: \"{transcript_context}\"\n"
            prompt += "Use this to understand the scene, but don't include it in descriptions.\n\n"
        
        prompt += "IMPORTANT DISTINCTIONS:\n"
        prompt += "- Scene = what's actually filmed (people, objects, actions, setting)\n"
        prompt += "- UI = text/graphics overlaid on top (captions, usernames, timestamps)\n"
        prompt += "- Audio descriptions go elsewhere, not in scene or UI\n\n"
        
        prompt += "Response format (JSON only, no other text):\n{\n"
        prompt += "  \"scene_description\": \"...\",\n"
        prompt += "  \"ui_elements\": [\n"
        prompt += "    {\"type\": \"text\", \"content\": \"...\", \"position\": \"...\", \"style\": \"...\"}\n"
        prompt += "  ]\n}"
        
        return prompt
    
    def _call_vlm(self, image_b64: str, prompt: str) -> str:
        """Call the VLM API."""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            import anthropic
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text
        
        elif self.provider == "local":
            return self._call_local_vlm(image_b64, prompt)
    
    def _call_local_vlm(self, image_b64: str, prompt: str) -> str:
        """Call local VLM (SmolVLM)."""
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Decode base64 to PIL Image
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(img_bytes))
        
        # Format prompt for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        )
        
        # Save input_ids length before moving to device
        input_ids = inputs["input_ids"]
        
        # Move to same device as model
        device = next(self.local_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.local_model.generate(
                **inputs,
                max_new_tokens=1000,  # Increased for complete JSON responses
                do_sample=False
            )
        
        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        # Decode response
        generated_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Extract response (already trimmed of prompt)
        response = generated_texts[0].strip()
        
        # SmolVLM might include "Assistant:" prefix
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        
        return response
    
    def _clean_vlm_description(self, description: str) -> str:
        """Clean VLM description by removing prompt leakage."""
        import re
        
        # Strategy 1: Split on common leak markers and keep only the first part
        leak_markers = [
            'PART 1',
            'PART 2',
            'If identical to previous',
            'If similar with changes',
            'if same as previous',  # NEW
            'Describe the VISUAL elements',
            'Describe what you actually see',  # NEW
            'Provide scene_description',  # NEW
            'main video content',  # NEW
            'NOT UI overlays',  # NEW
            'camera angle, people, objects',  # NEW
            'What\'s happening in the main video',  # Placeholder from old prompt
            'people, actions, setting',  # Placeholder from old prompt
            'Audio:',
            'Visual actions',
            '\n- Camera type',
            'Objects visible:',
            'Scene type:',
            'Text visible:',
            'UI elements',
            'UI/OVERLAY ELEMENTS',
            'Response format',  # NEW
            'JSON only',  # NEW
        ]
        
        # Find the earliest occurrence of any leak marker
        earliest_pos = len(description)
        for marker in leak_markers:
            pos = description.lower().find(marker.lower())
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        # Truncate at the leak marker
        if earliest_pos < len(description):
            description = description[:earliest_pos]
        
        # Strategy 2: Remove structured field outputs that appear at the start
        # Remove lines like "Scene type: Video\n" or "Objects: X, Y\n"
        description = re.sub(r'^Scene type:\s*[^\n]+\n', '', description, flags=re.IGNORECASE | re.MULTILINE)
        description = re.sub(r'^Objects:\s*[^\n]+\n', '', description, flags=re.IGNORECASE | re.MULTILINE)
        description = re.sub(r'^Actions:\s*[^\n]+\n', '', description, flags=re.IGNORECASE | re.MULTILINE)
        description = re.sub(r'^Text visible:\s*[^\n]+\n', '', description, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up leftover formatting
        description = description.strip()
        # Remove multiple consecutive newlines
        description = re.sub(r'\n\s*\n+', '\n', description)

        # Detect garbled / corrupted output (too many non-ASCII characters)
        if description:
            non_ascii = sum(1 for c in description if ord(c) > 127)
            if non_ascii / len(description) > 0.3:
                description = 'Unknown scene'

        # If description is empty after cleaning, mark as same shot
        if not description:
            description = "Same shot"
        
        return description
    
    def _is_invalid_ui_element(self, elem: dict, scene_desc: str) -> bool:
        """Check if UI element is prompt leakage or scene description."""
        content = elem.get('content', '').strip()
        if not content:
            return True
        
        content_lower = content.lower()
        
        # 1. Prompt leakage phrases
        leak_phrases = [
            'describe what',
            'provide scene',
            'camera angle',
            'main video content',
            'not ui overlays',
            'if same as previous',
            'same shot',
            'exact text visible',
            'response format',
            'json only',
            'ui elements',
            'scene_description',
            'actual text from video',  # Placeholder from examples
            'exact caption or label',  # Placeholder from examples
        ]
        if any(phrase in content_lower for phrase in leak_phrases):
            return True
        
        # 2. Content that starts with a person/subject noun → scene description, not overlay
        scene_subject_starters = (
            'woman ', 'man ', 'person ', 'people ', 'a woman', 'a man', 'a person',
            'bald ', 'young man', 'young woman', 'an old', 'an elderly',
        )
        if any(content_lower.startswith(s) for s in scene_subject_starters):
            return True

        # 3. Scene description keywords (not UI overlays)
        scene_keywords = [
            'camera',
            'blurry',
            'close up',
            'selfie',
            'looking at',
            'holding',
            'wearing',
            'with blonde',
            'with brown',
            'in a suit',
            'in a jacket',
            'in a room',
            'on keyboard',
            'face of',
            'finger on',
            'person\'s',
            'man with',
            'woman with',
            # Scene action verbs that appear in hallucinated descriptions
            'typing on',
            'sitting at',
            'sitting in',
            'standing in',
            'walking in',
            'working at',
            'drinking from',
            # Location phrases that never appear as UI text
            'coffee shop',
            'in the car',
            'at the desk',
            'at a desk',
            'in background',
        ]
        if any(keyword in content_lower for keyword in scene_keywords):
            return True
        
        # 3. Very similar to scene_description (likely duplicate)
        if scene_desc and content_lower in scene_desc.lower():
            return True
        
        # 4. Too long for typical UI text overlay (probably description)
        if elem.get('type') == 'text' and len(content) > 60:
            return True
        
        return False
    
    def _parse_vlm_response(
        self,
        response: str,
        frame_number: int,
        timestamp: float
    ) -> FrameAnalysis:
        """Parse VLM response into FrameAnalysis."""
        
        import json
        
        try:
            # Try to extract JSON from response
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
            
            # Fix common JSON syntax errors before parsing
            # 1. Extra closing brackets: "] ]" -> "]"
            json_str = re.sub(r'\]\s*\]\s*\}', ']\n}', json_str)
            # 2. Missing closing bracket before brace: "} ]" -> "}]" 
            json_str = re.sub(r'\}\s*\]\s*$', '}', json_str)
            # 3. Broken key-value pair with newline in middle: "\"t\n            \"type\"" -> "\"type\""
            json_str = re.sub(r'"\w\n\s+"', '"', json_str)
            
            data = json.loads(json_str)
            
            # Parse objects and actions (handle both list and comma-separated string)
            objects = data.get('objects', [])
            if isinstance(objects, str):
                objects = [o.strip() for o in objects.split(',') if o.strip()]
            # Deduplicate objects
            objects = list(dict.fromkeys(objects))  # Preserves order
            
            actions = data.get('actions', [])
            if isinstance(actions, str):
                actions = [a.strip() for a in actions.split(',') if a.strip()]
            # Deduplicate actions
            actions = list(dict.fromkeys(actions))
            
            # Filter out narrative actions (actions that mention names or abstract concepts)
            visual_actions = []
            narrative_keywords = ['was', 'were', 'directed', 'flying', 'advised', 'told', 'said']
            for action in actions:
                action_lower = action.lower()
                # Keep if it's a visual action (short, no narrative verbs)
                if not any(keyword in action_lower for keyword in narrative_keywords):
                    visual_actions.append(action)
            actions = visual_actions
            
            # Parse UI elements (should be a list of dicts)
            ui_elements = data.get('ui_elements', [])
            if not isinstance(ui_elements, list):
                ui_elements = []
            
            # Get scene description for comparison
            scene_description = data.get('scene_description', data.get('description', ''))
            
            # Deduplicate UI elements and filter invalid ones
            seen_ui = set()
            unique_ui = []
            for elem in ui_elements:
                if isinstance(elem, dict):
                    # Skip invalid UI elements
                    if self._is_invalid_ui_element(elem, scene_description):
                        continue
                    
                    content = elem.get('content', '')
                    key = (elem.get('type', ''), content, elem.get('position', ''))
                    if key not in seen_ui:
                        seen_ui.add(key)
                        unique_ui.append(elem)
            ui_elements = unique_ui
            
            # Clean up scene description - remove prompt leakage
            scene_description = self._clean_vlm_description(scene_description)
            
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp=timestamp,
                description=scene_description,  # Now contains scene_description
                objects=objects,
                actions=actions,
                scene_type=data.get('scene_type'),
                mood=data.get('mood'),
                text_visible=data.get('text_visible'),
                ui_elements=ui_elements  # Now structured with type, content, position, style
            )
        
        except Exception as e:
            # Fallback: try to extract useful info from partial JSON or raw text
            description = response
            ui_elements = []
            
            # Try to extract scene_description and ui_elements from malformed JSON
            if response.strip().startswith('{'):
                # Try regex extraction for scene_description
                scene_match = re.search(r'"scene_description"\s*:\s*"([^"]*)"', response)
                if scene_match:
                    description = scene_match.group(1)
                else:
                    # Fallback to description field
                    desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', response)
                    if desc_match:
                        description = desc_match.group(1)
                    else:
                        # JSON parsing failed and no description found - use truncated text
                        description = response[:500]
                
                # Try to extract ui_elements array using regex
                ui_match = re.search(r'"ui_elements"\s*:\s*\[(.*?)\]\s*(?:\]|$)', response, re.DOTALL)
                if ui_match:
                    # Found ui_elements array - try to parse each element
                    ui_array_str = '[' + ui_match.group(1) + ']'
                    try:
                        # Clean up common issues
                        ui_array_str = re.sub(r'"\w\n\s+"', '"', ui_array_str)  # Fix broken keys
                        ui_array_str = re.sub(r',\s*\]', ']', ui_array_str)  # Trailing commas
                        ui_parsed = json.loads(ui_array_str)
                        if isinstance(ui_parsed, list):
                            # Deduplicate and filter invalid elements
                            seen_ui = set()
                            for elem in ui_parsed:
                                if isinstance(elem, dict):
                                    # Skip invalid UI elements
                                    if self._is_invalid_ui_element(elem, description):
                                        continue
                                    key = (elem.get('type', ''), elem.get('content', ''), elem.get('position', ''))
                                    if key not in seen_ui:
                                        seen_ui.add(key)
                                        ui_elements.append(elem)
                    except:
                        # UI elements extraction failed - keep empty
                        pass
            
            # Apply cleaning to fallback description as well
            description = self._clean_vlm_description(description)
            
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp=timestamp,
                description=description,
                objects=[],
                actions=[],
                scene_type=None,
                mood=None,
                text_visible=None,
                ui_elements=ui_elements  # Now tries to extract even on failure
            )
    
    def generate_summary(self, prompt: str) -> str:
        """Generate a text summary using the LLM (text-only, no image)."""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for text-only
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            import anthropic
            
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Cheaper model
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        
        elif self.provider == "local":
            # For text-only, we can just use the VLM without an image
            # or fall back to a simpler summary
            import torch
            from PIL import Image
            import numpy as np
            
            # Create a blank/dummy image for text-only queries
            dummy_image = Image.fromarray(
                np.zeros((100, 100, 3), dtype=np.uint8)
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=prompt_text,
                images=[dummy_image],
                return_tensors="pt"
            )
            
            device = next(self.local_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.local_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            response = generated_texts[0]
            if prompt_text in response:
                response = response.replace(prompt_text, "").strip()
            if response.startswith("Assistant:"):
                response = response[10:].strip()
            
            return response
        
        return "Summary generation not available."    
    def _create_segment_analysis_prompt(
        self,
        num_frames: int,
        transcript_context: str = "",
        previous_segment_description: str = ""
    ) -> str:
        """Create the VLM prompt for segment analysis."""
        
        prompt = f"Analyze this video segment using {num_frames} representative frames.\n\n"
        
        if previous_segment_description:
            prompt += f"Previous segment: {previous_segment_description[:100]}\n"
            prompt += "If same as previous, just say: 'Same scene'\n\n"
        
        prompt += "Provide scene_description (ONLY the main video content across the segment):\n"
        prompt += "- Describe the overall scene, actions, and setting that persist through the segment\n"
        prompt += "- Be specific: 'Conversation between two people in office', 'Person walking through park'\n"
        prompt += "- DO NOT include any text overlays, UI elements, or captions\n"
        prompt += "- Focus on the core visual narrative of the segment\n\n"
        
        prompt += "Provide ui_elements (text/graphics overlaid on video that appear in segment):\n"
        prompt += "- ONLY include: usernames, timestamps, hashtags, captions, icons\n"
        prompt += "- Must be TEXT OR ICONS overlaid on the video content\n"
        prompt += "- If no consistent UI elements, return empty array []\n\n"
        
        prompt += "Provide actions (key activities happening in the segment):\n"
        prompt += "- List main actions: 'speaking', 'walking', 'typing', 'gesturing'\n"
        prompt += "- Focus on movement and behavior patterns\n\n"
        
        prompt += "Provide objects (important objects visible in segment):\n"
        prompt += "- Key items: 'phone', 'computer', 'car', 'coffee cup'\n"
        prompt += "- Only consistently visible objects\n\n"
        
        prompt += "Provide scene_type (overall category):\n"
        prompt += "- Options: 'conversation', 'action', 'explanation', 'demonstration', 'narrative'\n\n"
        
        if transcript_context:
            prompt += f"Audio context: \"{transcript_context}\"\n"
            prompt += "Use this to understand the scene, but don't include in descriptions.\n\n"
        
        prompt += "Response format (JSON only):\n{\n"
        prompt += "  \"scene_description\": \"Two people discussing project in modern office\",\n"
        prompt += "  \"ui_elements\": [\n"
        prompt += "    {\"type\": \"text\", \"content\": \"Team Meeting\", \"position\": \"top\"}\n"
        prompt += "  ],\n"
        prompt += "  \"actions\": [\"speaking\", \"gesturing\"],\n"
        prompt += "  \"objects\": [\"laptop\", \"coffee\"],\n"
        prompt += "  \"scene_type\": \"conversation\"\n}"
        
        return prompt
    
    def _call_vlm_multi_image(self, image_b64_list: List[str], prompt: str) -> str:
        """Call VLM with multiple images for segment analysis."""
        
        if self.provider == "openai":
            # OpenAI supports multiple images in one message
            content = [{"type": "text", "text": prompt}]
            for img_b64 in image_b64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            # Anthropic also supports multiple images
            content = [{"type": "text", "text": prompt}]
            for img_b64 in image_b64_list:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }
                })
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": content}]
            )
            return message.content[0].text
        
        elif self.provider == "local":
            # For local models, use the first image as representative
            # (SmolVLM doesn't easily support multiple images)
            return self._call_vlm(image_b64_list[0], prompt)
        
        return "Multi-image analysis not supported."
    
    def _parse_segment_vlm_response(self, response: str) -> dict:
        """Parse segment VLM response into analysis dictionary."""
        
        import json
        
        try:
            # Extract JSON from response
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
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract fields with defaults
            result = {
                'scene_description': data.get('scene_description', ''),
                'ui_elements': data.get('ui_elements', []),
                'actions': data.get('actions', []),
                'objects': data.get('objects', []),
                'scene_type': data.get('scene_type', 'unknown')
            }
            
            # Clean scene description
            result['scene_description'] = self._clean_vlm_description(result['scene_description'])
            
            # Filter invalid UI elements
            result['ui_elements'] = [
                elem for elem in result['ui_elements']
                if not self._is_invalid_ui_element(elem, result['scene_description'])
            ]
            
            return result
            
        except Exception as e:
            # Fallback parsing
            return {
                'scene_description': response[:200] if response else 'Analysis failed',
                'ui_elements': [],
                'actions': [],
                'objects': [],
                'scene_type': 'unknown'
            }