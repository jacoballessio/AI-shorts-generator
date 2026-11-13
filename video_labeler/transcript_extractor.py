"""
Transcript extraction using OpenAI Whisper with optional speaker diarization.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from .models import TranscriptSegment


class TranscriptExtractor:
    """Extracts transcripts from video audio using Whisper."""
    
    def __init__(
        self,
        model: Optional[str] = "base",
        use_speaker_diarization: bool = False,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
    ):
        """
        Initialize transcript extractor.
        
        Args:
            model: Whisper model size (tiny, base, small, medium, large)
                   None to disable
            use_speaker_diarization: Whether to run pyannote diarization
            diarization_model: Hugging Face pyannote diarization pipeline ID
        """
        self.model = model
        self.whisper_model = None
        self.use_speaker_diarization = use_speaker_diarization
        self.diarization_model_name = diarization_model
        self.diarization_pipeline = None
        
        if model:
            try:
                import whisper
                print(f"   Loading Whisper model: {model}")
                self.whisper_model = whisper.load_model(model)
            except ImportError:
                print("⚠️  Warning: openai-whisper not installed. Transcription disabled.")
                print("   Install with: pip install openai-whisper")
                self.model = None

        if self.use_speaker_diarization:
            self._init_diarization()

    def _init_diarization(self):
        """Initialize pyannote diarization pipeline when available."""
        # Avoid repeated non-fatal Hugging Face symlink warnings on Windows.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        env_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("PYANNOTE_TOKEN")
        )
        try:
            # Prefer explicit on-disk login token from `hf auth login`.
            # huggingface_hub.get_token() prioritizes HF_TOKEN env var, which can
            # be stale and cause gated-model 403 despite a valid CLI login.
            from huggingface_hub.utils._auth import _get_token_from_file
            cached_token = _get_token_from_file()
        except Exception:
            cached_token = None

        token = cached_token or env_token
        if cached_token and env_token and cached_token != env_token:
            print("[INFO] Using cached HF login token from disk (overriding HF_TOKEN env var).")
        if not token:
            print("⚠️  Warning: Speaker diarization requested but no HF token found.")
            print("   Set HF_TOKEN (or HUGGINGFACE_TOKEN), or run `hf auth login`.")
            self.use_speaker_diarization = False
            return

        try:
            import warnings as _warnings
            _warnings.filterwarnings(
                "ignore",
                message=r"^\s*torchcodec is not installed correctly so built-in audio decoding will fail\..*",
                category=UserWarning,
            )
            from pyannote.audio import Pipeline
            import torch

            print(f"   Loading speaker diarization model: {self.diarization_model_name}")
            def _load_pipeline(auth_token: str):
                try:
                    # Newer pyannote versions
                    return Pipeline.from_pretrained(
                        self.diarization_model_name,
                        token=auth_token,
                    )
                except TypeError:
                    # Older pyannote versions
                    return Pipeline.from_pretrained(
                        self.diarization_model_name,
                        use_auth_token=auth_token,
                    )

            try:
                self.diarization_pipeline = _load_pipeline(token)
            except Exception as first_error:
                # If an env token is stale/mismatched, fall back to HF CLI login token.
                first_msg = str(first_error)
                can_retry_with_cached = (
                    cached_token
                    and env_token
                    and cached_token != env_token
                    and ("403" in first_msg or "gated" in first_msg or "Cannot access gated repo" in first_msg)
                )
                if not can_retry_with_cached:
                    raise
                print("[INFO] Environment HF token denied gated repo; retrying with cached HF login token...")
                self.diarization_pipeline = _load_pipeline(cached_token)

            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
        except ImportError:
            print("⚠️  Warning: pyannote.audio not installed. Speaker diarization disabled.")
            print("   Install with: pip install pyannote.audio")
            self.use_speaker_diarization = False
        except Exception as e:
            msg = str(e)
            if (
                "segmentation-3.0" in msg
                or "speaker-diarization-community-1" in msg
                or "gated" in msg
                or "403" in msg
            ):
                print("⚠️  Warning: pyannote gated model access is missing.")
                print("   1. Visit https://huggingface.co/pyannote/segmentation-3.0 and accept terms")
                print("   2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept terms")
                print("   3. Visit https://huggingface.co/pyannote/speaker-diarization-community-1 and accept terms")
                print("   4. Ensure your HF login/token belongs to the same account with accepted access")
            print(f"⚠️  Warning: Failed to load diarization model: {e}")
            self.use_speaker_diarization = False
    
    def extract(
        self,
        video_path: str,
        output_path: Optional[Path] = None
    ) -> Tuple[List[TranscriptSegment], str]:
        """
        Extract transcript from video.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save transcript JSON
            
        Returns:
            (List of transcript segments, full transcript text)
        """
        if not self.whisper_model:
            print("   Skipping transcript extraction (Whisper not available)")
            return [], ""
        
        # Transcribe with word-level timestamps
        result = self.whisper_model.transcribe(
            video_path,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract segments and filter hallucinations
        segments = []
        for segment in result['segments']:
            text = segment['text'].strip()
            
            # Filter out likely hallucinations
            if self._is_hallucination(text, segment):
                continue
            
            seg = TranscriptSegment(
                text=text,
                start_time=segment['start'],
                end_time=segment['end'],
                confidence=segment.get('confidence')
            )
            segments.append(seg)

        if self.use_speaker_diarization and self.diarization_pipeline and segments:
            self._apply_speaker_diarization(video_path, segments)
        
        # Full transcript
        full_transcript = " ".join([s.text for s in segments])
        
        # Save if output path provided
        if output_path:
            data = {
                'full_transcript': full_transcript,
                'segments': [s.to_dict() for s in segments]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        return segments, full_transcript

    def _apply_speaker_diarization(self, video_path: str, segments: List[TranscriptSegment]):
        """Assign speaker labels to transcript segments using overlap with diarization turns."""
        # pyannote 4.x has a broken AudioDecoder import from torchcodec that crashes on any
        # file path input regardless of format. Pass a waveform tensor directly to bypass
        # all of pyannote's internal file I/O.
        try:
            import torch
            import whisper

            # Use Whisper's FFmpeg-backed loader for consistent, robust 16k mono audio.
            waveform_np = whisper.load_audio(video_path)
            waveform_tensor = torch.from_numpy(waveform_np.astype("float32")).unsqueeze(0)
            audio_input = {"waveform": waveform_tensor, "sample_rate": 16000}
        except Exception as e:
            print(f"⚠️  Warning: Could not extract audio waveform for diarization: {e}")
            return

        try:
            diarization = self.diarization_pipeline(audio_input)
        except Exception as e:
            print(f"⚠️  Warning: Speaker diarization failed: {e}")
            return

        speaker_turns = self._extract_speaker_turns(diarization)

        if not speaker_turns:
            print("⚠️  Warning: Speaker diarization produced no speaker turns. Falling back to SPEAKER_1 for all segments.")
            for seg in segments:
                seg.speaker = "SPEAKER_1"
                cleaned_text = re.sub(r"^\[SPEAKER_\d+\]\s*", "", seg.text).strip()
                seg.text = f"[SPEAKER_1] {cleaned_text}" if cleaned_text else "[SPEAKER_1]"
            return

        speaker_map: Dict[str, str] = {}
        speaker_counter = 0

        for seg in segments:
            best_speaker = None
            best_overlap = 0.0
            for turn_start, turn_end, turn_speaker in speaker_turns:
                overlap = max(0.0, min(seg.end_time, turn_end) - max(seg.start_time, turn_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn_speaker

            if not best_speaker or best_overlap <= 0:
                continue

            if best_speaker not in speaker_map:
                speaker_counter += 1
                speaker_map[best_speaker] = f"SPEAKER_{speaker_counter}"

            speaker_label = speaker_map[best_speaker]
            seg.speaker = speaker_label
            cleaned_text = re.sub(r"^\[SPEAKER_\d+\]\s*", "", seg.text).strip()
            seg.text = f"[{speaker_label}] {cleaned_text}" if cleaned_text else f"[{speaker_label}]"

    def _extract_speaker_turns(self, diarization_output) -> List[Tuple[float, float, str]]:
        """Normalize pyannote diarization outputs across versions to (start, end, speaker)."""
        recognized_output = False

        def _from_itertracks(obj) -> List[Tuple[float, float, str]]:
            turns: List[Tuple[float, float, str]] = []
            if not hasattr(obj, "itertracks"):
                return turns
            for turn, _, speaker in obj.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(speaker)))
            return turns

        def _safe_getattr(obj, name: str):
            try:
                return getattr(obj, name)
            except Exception:
                return None

        # Preferred path for newer pyannote dataclasses/wrappers.
        if hasattr(diarization_output, "serialize"):
            recognized_output = True
            try:
                payload = diarization_output.serialize()
                if isinstance(payload, dict):
                    for key in ("diarization", "exclusive_diarization", "exclusiveDiarization"):
                        seq = payload.get(key)
                        if not isinstance(seq, list):
                            continue
                        parsed: List[Tuple[float, float, str]] = []
                        for item in seq:
                            if not isinstance(item, dict):
                                continue
                            start = item.get("start")
                            end = item.get("end")
                            speaker = item.get("speaker") or item.get("label")
                            if start is None or end is None or speaker is None:
                                continue
                            parsed.append((float(start), float(end), str(speaker)))
                        if parsed:
                            return parsed
            except Exception:
                pass

        # Legacy path: pyannote.core.Annotation
        turns = _from_itertracks(diarization_output)
        if turns:
            return turns

        # Newer wrappers sometimes expose underlying annotation via attributes.
        for attr in ("speaker_diarization", "exclusive_speaker_diarization", "annotation", "diarization"):
            nested = _safe_getattr(diarization_output, attr)
            if nested is None:
                continue
            recognized_output = True
            turns = _from_itertracks(nested)
            if turns:
                return turns

        # Dict-like outputs used by some pyannote variants.
        if isinstance(diarization_output, dict):
            recognized_output = True
            for key in ("speaker_diarization", "annotation", "diarization"):
                nested = diarization_output.get(key)
                turns = _from_itertracks(nested)
                if turns:
                    return turns

            seq = diarization_output.get("segments") or diarization_output.get("turns")
            if isinstance(seq, list):
                parsed: List[Tuple[float, float, str]] = []
                for item in seq:
                    if not isinstance(item, dict):
                        continue
                    start = item.get("start")
                    end = item.get("end")
                    speaker = item.get("speaker") or item.get("label")
                    if start is None or end is None or speaker is None:
                        continue
                    parsed.append((float(start), float(end), str(speaker)))
                if parsed:
                    return parsed

        # Last resort: sequence of segment-like objects.
        if isinstance(diarization_output, list):
            recognized_output = True
            parsed: List[Tuple[float, float, str]] = []
            for item in diarization_output:
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                speaker = getattr(item, "speaker", None) or getattr(item, "label", None)
                if start is None or end is None or speaker is None:
                    continue
                parsed.append((float(start), float(end), str(speaker)))
            if parsed:
                return parsed

        # Generic catch-all: inspect fields/attributes and use the first Annotation-like value.
        try:
            attrs = vars(diarization_output)
        except Exception:
            attrs = None
        if isinstance(attrs, dict):
            recognized_output = True
            for value in attrs.values():
                turns = _from_itertracks(value)
                if turns:
                    return turns

        # Some wrappers use __slots__ / properties and do not expose __dict__.
        # Inspect public attributes and try Annotation-like values.
        try:
            for attr in dir(diarization_output):
                if attr.startswith("_"):
                    continue
                value = _safe_getattr(diarization_output, attr)
                if value is not None:
                    recognized_output = True
                turns = _from_itertracks(value)
                if turns:
                    return turns
        except Exception:
            pass

        # pyannote 4.x returns DiarizeOutput. Even when no turns are found,
        # this is still a recognized output type and should not be reported as unsupported.
        try:
            if type(diarization_output).__name__ == "DiarizeOutput":
                recognized_output = True
        except Exception:
            pass

        if not recognized_output:
            try:
                kind = type(diarization_output).__name__
                print(f"[INFO] Unsupported diarization output type: {kind}")
            except Exception:
                pass
        return []
    
    def _is_hallucination(self, text: str, segment: dict) -> bool:
        """Detect if transcript segment is likely a Whisper hallucination."""
        import re
        
        # Empty or whitespace-only
        if not text or not text.strip():
            return True
        
        text_lower = text.lower().strip()
        
        # Very high no-speech probability (Whisper is unsure)
        no_speech_prob = segment.get('no_speech_prob', 0)
        if no_speech_prob > 0.8:
            return True
        
        # Single character or very short
        if len(text) <= 2:
            return True
        
        # Repeated characters pattern (P-P, A-A, etc)
        if re.match(r'^([A-Z]-?){2,}$', text.strip()):
            return True
        
        # Repeated words pattern (ha ha ha, no no no, etc) - likely on-screen text
        words = text_lower.split()
        if len(words) >= 3:
            # Check if same word repeated 3+ times
            if len(set(words)) == 1:  # All same word
                return True
            # Check if same short word (1-3 chars) repeated 4+ times
            word_counts = {}
            for word in words:
                if len(word) <= 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
            if any(count >= 4 for count in word_counts.values()):
                return True
        
        # Contains mostly non-ASCII (Cyrillic, etc) for short segments
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if ascii_ratio < 0.5 and len(text) < 20:
            return True
        
        # Common hallucination phrases
        hallucination_patterns = [
            r'^[\.\,\!\?]+$',  # Just punctuation
            r'^(uh|um|hmm|mm|ah)+$',  # Just filler sounds
            r'^thank you\.$',  # Common hallucination at end
            r'^bye\.$',  # Common hallucination
        ]
        for pattern in hallucination_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
