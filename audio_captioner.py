"""
Audio Captioning using Whisper-based model
Provides descriptions of audio content including music, sound effects, and environmental noises.

Model: MU-NLPC/whisper-tiny-audio-captioning
Based on OpenAI Whisper, fine-tuned for audio captioning on AudioCaps, AudioSet, and Clotho datasets.

Usage:
    from audio_captioner import AudioCaptioner
    captioner = AudioCaptioner()
    description = captioner.describe_audio("path/to/audio.mp3")
"""

import warnings
import os
warnings.filterwarnings("ignore")


def _patch_hf_head_allow_redirects_compat():
    """Make huggingface_hub HEAD calls compatible with httpx follow_redirects arg."""
    try:
        from huggingface_hub import get_session
        session = get_session()
        head_fn = getattr(session, "head", None)
        if head_fn is None:
            return

        # requests.Session supports allow_redirects already; only patch incompatible clients.
        try:
            head_fn("https://huggingface.co", allow_redirects=True, timeout=1)
            return
        except TypeError:
            pass
        except Exception:
            # Any other runtime error means signature accepted; no patch needed.
            return

        original_head = head_fn

        def _head_with_redirect_compat(*args, **kwargs):
            if "allow_redirects" in kwargs and "follow_redirects" not in kwargs:
                kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
            return original_head(*args, **kwargs)

        session.head = _head_with_redirect_compat
    except Exception:
        # Non-fatal: keep existing behavior if patching fails.
        return

try:
    import torch
    import librosa
    AUDIO_CAPTION_DEPENDENCIES_OK = True
except ImportError:
    AUDIO_CAPTION_DEPENDENCIES_OK = False


class AudioCaptioner:
    """Audio captioning using Whisper-tiny model fine-tuned for audio captioning."""
    
    def __init__(self, model_id: str = "MU-NLPC/whisper-tiny-audio-captioning", style: str = "clotho"):
        """Initialize the audio captioner.
        
        Args:
            model_id: Hugging Face model identifier
            style: Caption style - "clotho" (default), "audiocaps", or "audioset"
        """
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.style = style
        self.style_prefixes = {
            "clotho": "clotho > caption: ",
            "audiocaps": "audiocaps > caption: ",
            "audioset": "audioset > keywords: ",
        }
        
        if not AUDIO_CAPTION_DEPENDENCIES_OK:
            print("[INFO] Audio captioning dependencies not installed; skipping audio captioner")
            return
        
        try:
            # Import transformers here to avoid Python 3.13 import issues
            from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

            # Fix hub client compatibility in environments where get_session() returns httpx.
            _patch_hf_head_allow_redirects_compat()
            
            print(f"[AUDIO] Loading audio captioner model: {model_id}")
            
            # Load as WhisperForConditionalGeneration (base Whisper encoder-decoder)
            # The custom generate logic will be handled in describe_audio method
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
            
            self.tokenizer = WhisperTokenizer.from_pretrained(model_id, language="en", task="transcribe")
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.model.eval()
            print(f"[AUDIO] Model loaded successfully on {device}")
            print(f"[AUDIO] Model class: {type(self.model).__name__}")
        except Exception as e:
            print(f"[AUDIO] Failed to load audio captioner: {e}")
            if os.getenv("AUDIO_CAPTION_DEBUG") == "1":
                import traceback
                traceback.print_exc()
            self.model = None
            self.tokenizer = None
            self.feature_extractor = None
    
    def describe_audio(self, audio_path: str, style: str = None, max_length: int = 100) -> str:
        """Generate a caption describing the audio content.
        
        Args:
            audio_path: Path to audio file (mp3, wav, ogg, etc.)
            style: Caption style override (clotho, audiocaps, audioset)
            max_length: Maximum length of generated caption
            
        Returns:
            Audio description string, or empty string if model unavailable
        """
        if self.model is None or self.tokenizer is None or self.feature_extractor is None:
            return ""
        
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=self.feature_extractor.sampling_rate)
            input_features = self.feature_extractor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Generate caption without forced prefix
            # The model was fine-tuned to generate captions, so it should work without forced IDs
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features=input_features.to(device),
                    max_length=max_length,
                    num_beams=1,  # Set to 1 for faster generation, increase for better quality
                    task="transcribe",  # For caption generation
                )
            
            # Decode and postprocess
            caption = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up the caption
            # Remove transcription markers and codec prefixes
            caption = caption.replace("<|startoftranscript|>", "").strip()

            # Remove style prefixes if present (these are exact patterns from model training)
            for prefix in self.style_prefixes.values():
                if caption.startswith(prefix):
                    caption = caption[len(prefix):].strip()
                    break

            # Also remove "clotho>" prefix variations just in case
            if caption.startswith("clotho"):
                caption = caption[6:].strip()
                if caption.startswith(">"):
                    caption = caption[1:].strip()

            # Strip any model-header artefacts: handles doubled prefixes, "are" vs ":" etc.
            # e.g. "audiosetaudioset keywords are X" or "clotho > caption: X"
            import re
            caption = re.sub(
                r'^(?:(?:audioset|audiocaps|clotho)\s*[>:]?\s*){1,3}'
                r'(?:keywords?|caption|tags?)?\s*(?:are|is|:)?\s*',
                '', caption, flags=re.IGNORECASE
            ).strip()

            return caption
        
        except Exception as e:
            print(f"[AUDIO] Error describing audio: {e}")
            if os.getenv("AUDIO_CAPTION_DEBUG") == "1":
                import traceback
                traceback.print_exc()
            return ""
    
    def describe_video_audio(self, video_path: str, style: str = None) -> str:
        """Extract and describe audio from a video file.
        
        Args:
            video_path: Path to video file
            style: Caption style override
            
        Returns:
            Audio description string
        """
        if self.model is None:
            return ""
        
        try:
            import tempfile
            import os
            
            # Use moviepy to extract audio (more reliable than ffmpeg subprocess)
            try:
                from moviepy.editor import VideoFileClip
                
                video = VideoFileClip(str(video_path))
                if video.audio is None:
                    print(f"[AUDIO] Video has no audio track")
                    return ""
                
                # Create temporary audio file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_audio_path = tmp.name
                
                try:
                    # Write audio to temporary file
                    print(f"[AUDIO] Extracting audio from video...")
                    video.audio.write_audiofile(
                        temp_audio_path,
                        verbose=False,
                        logger=None
                    )
                    
                    # Verify the file was created
                    if not os.path.exists(temp_audio_path):
                        print(f"[AUDIO] Failed to extract audio to file")
                        return ""
                    
                    file_size = os.path.getsize(temp_audio_path)
                    if file_size < 1000:
                        print(f"[AUDIO] Audio file too small ({file_size} bytes)")
                        return ""
                    
                    print(f"[AUDIO] Extracted audio: {file_size} bytes")
                    
                    # Describe the extracted audio
                    description = self.describe_audio(temp_audio_path, style=style)
                    return description
                finally:
                    # Clean up
                    video.close()
                    if os.path.exists(temp_audio_path):
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
            
            except ImportError:
                print("[AUDIO] moviepy not available, trying ffmpeg fallback...")
                # Fallback to ffmpeg if moviepy not available
                return self._extract_audio_ffmpeg(video_path, style)
        
        except Exception as e:
            print(f"[AUDIO] Error extracting audio from video: {e}")
            if os.getenv("AUDIO_CAPTION_DEBUG") == "1":
                import traceback
                traceback.print_exc()
            return ""

    def describe_video_audio_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        style: str = None,
    ) -> str:
        """Extract and describe the audio for a specific time range of a video.

        This method trims the video to the requested segment (using ffmpeg) and
        then runs the normal ``describe_audio`` pipeline on the resulting WAV file.
        It is useful for generating context-aware descriptions for individual
        segments rather than the entire video.

        Args:
            video_path: Path to the source video file.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.
            style: Caption style override.

        Returns:
            Audio description for that segment, or empty string on failure.
        """
        if self.model is None:
            return ""

        try:
            import tempfile
            import os
            import subprocess
            import platform
            import time

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio_path = tmp.name

            try:
                # Build ffmpeg command to extract the specified interval
                if platform.system() == "Windows":
                    cmd = (
                        f'ffmpeg -i "{video_path}" -ss {start_time} -to {end_time}'
                        f' -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_audio_path}"'
                    )
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=120,
                    )
                else:
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(video_path),
                        "-ss",
                        str(start_time),
                        "-to",
                        str(end_time),
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-y",
                        temp_audio_path,
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=120,
                    )

                time.sleep(0.5)

                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) < 1000:
                    print(f"[AUDIO] segment extraction failed for {start_time}-{end_time}")
                    return ""

                file_size = os.path.getsize(temp_audio_path)
                print(f"[AUDIO] Extracted segment audio ({start_time}-{end_time}s): {file_size} bytes")

                description = self.describe_audio(temp_audio_path, style=style)
                return description
            finally:
                if os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                    except:
                        pass

        except Exception as e:
            print(f"[AUDIO] Error describing video segment: {e}")
            return ""
    
    def _extract_audio_ffmpeg(self, video_path: str, style: str = None) -> str:
        """Fallback audio extraction using ffmpeg directly."""
        try:
            import subprocess
            import tempfile
            import os
            import time
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio_path = tmp.name
            
            try:
                # Use ffmpeg via shell command
                import platform
                if platform.system() == "Windows":
                    cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_audio_path}"'
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=120,
                    )
                else:
                    cmd = [
                        "ffmpeg", "-i", str(video_path),
                        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                        "-y", temp_audio_path
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=120,
                    )
                
                time.sleep(0.5)
                
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) < 1000:
                    print(f"[AUDIO] ffmpeg extraction failed")
                    return ""
                
                file_size = os.path.getsize(temp_audio_path)
                print(f"[AUDIO] Extracted audio via ffmpeg: {file_size} bytes")
                
                description = self.describe_audio(temp_audio_path, style=style)
                return description
            finally:
                if os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                    except:
                        pass
        
        except Exception as e:
            print(f"[AUDIO] ffmpeg extraction failed: {e}")
            return ""
