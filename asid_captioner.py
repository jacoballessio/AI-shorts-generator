"""
ASID Captioner wrapper
Provides an interface to the ASID-Captioner-3B audiovisual captioning model from Hugging Face.
The model can generate fine-grained descriptions of video content including audio/music/sfx.

Usage:
    from asid_captioner import ASIDCaptioner
    cap = ASIDCaptioner()
    desc = cap.describe_video("path/to/video.mp4")

If the model or dependencies are unavailable, the methods will gracefully return an empty string.
"""

import os

# try to import core requirements; qwen_omni_utils is optional
try:
    import torch
    ASID_DEPENDENCIES_OK = True
except ImportError:
    ASID_DEPENDENCIES_OK = False

# qwen_omni_utils may not be installed on pip; provide a minimal fallback
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    def process_mm_info(conversation, use_audio_in_video: bool = False):
        """Simplified media extractor used when the real helper isn't present.

        The conversation is a list of dicts with possible media entries, e.g.
        {"type": "video", "video": path} or {"type": "audio", "audio": path}.
        We return three lists (audio, image, video) which the processor expects.
        """
        audios = []
        images = []
        videos = []
        for item in conversation:
            if isinstance(item, dict):
                if item.get('type') == 'audio' and 'audio' in item:
                    audios.append(item['audio'])
                elif item.get('type') == 'image' and 'image' in item:
                    images.append(item['image'])
                elif item.get('type') == 'video' and 'video' in item:
                    videos.append(item['video'])
        # if nothing was found but the conversation has a top-level video prompt,
        # try to use it as fallback (some wrappers pass the video separately anyway)
        if not videos:
            for item in conversation:
                if isinstance(item, dict) and item.get('type') == 'video':
                    videos.append(item.get('video', ''))
        return audios, images, videos


class ASIDCaptioner:
    def __init__(self, model_id: str = "AudioVisual-Caption/ASID-Captioner-3B"):
        """Load the ASID model and processor."""
        self.model = None
        self.processor = None
        if not ASID_DEPENDENCIES_OK:
            print("[INFO] ASID dependencies not installed; skipping ASID captioner")
            return
        
        # Try importing the transformers classes here instead of at module level
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        except ImportError as e:
            print(f"[INFO] Failed to import Qwen classes from transformers: {e}")
            return
            
        try:
            # loading may be slow and GPU-intensive
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                enable_talker=False,  # Disable talker in config
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f"[INFO] Failed to load ASID model: {e}")
            self.model = None
            self.processor = None

    def describe_video(self, video_path: str, prompt: str = None) -> str:
        """Run the ASID model on a video file and return the generated caption.

        Args:
            video_path: Path to the video file.
            prompt: Optional instruction prompt; if None a generic comprehensive prompt is used.
        """
        if self.model is None or self.processor is None:
            return ""  # nothing available

        if prompt is None:
            prompt = (
                "Provide a comprehensive description of all the content in the video, "
                "leaving out no details, and naturally covering the scene, characters, "
                "objects, actions, narrative elements, speech, camera, and emotions in a "
                "single coherent account."
            )

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a video captioning assistant."}]},
            {"role": "user", "content": [{"type": "video", "video": video_path},
                                            {"type": "text", "text": prompt}]},
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # we no longer rely on process_mm_info since the helper may be missing
        audios = []
        images = []
        videos = [video_path]
        try:
            # build kwargs dynamically so we don't pass an empty audio list,
            # which triggers a Whisper feature extraction bug.
            proc_kwargs = {
                "text": text,
                "videos": videos,
                "return_tensors": "pt",
                "padding": True,
                "use_audio_in_video": False,  # Disable audio processing to avoid the StopIteration bug
            }
            if audios:
                proc_kwargs["audio"] = audios
            if images:
                proc_kwargs["images"] = images
            print(f"[ASID] Processing video: {video_path}")
            print(f"[ASID] Videos list: {videos}")
            inputs = self.processor(**proc_kwargs)
            print(f"[ASID] Processor inputs created successfully")
        except Exception as e:
            print(f"[ASID] media processing failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device).to(self.model.dtype)

        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=False,  # Match processor setting to avoid talker issues
                do_sample=False,
                thinker_max_new_tokens=4096,
                repetition_penalty=1.1,
                use_cache=True,
            )

        decoded = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"[ASID] Raw decoded text length: {len(decoded)}")
        print(f"[ASID] Raw decoded text preview: {repr(decoded[:200])}")
        answer = decoded.split("\nassistant\n")[-1].strip()
        print(f"[ASID] Final answer length: {len(answer)}")
        print(f"[ASID] Final answer preview: {repr(answer[:200])}")
        return answer

    def describe_audio(self, audio_path: str, prompt: str = None) -> str:
        """Convenience wrapper that treats an audio file as a ""video"" for ASID.
        The model will still process audio and produce a description.
        """
        return self.describe_video(audio_path, prompt)
