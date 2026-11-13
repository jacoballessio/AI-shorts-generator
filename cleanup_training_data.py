"""
Post-process training dataset JSON files:
  1. Fix mojibake titles (multi-encoded UTF-8 decoded as latin-1)
  2. Strip "At Xs," timestamp markers and audio info from scene_description
  3. Convert prose ui_elements_html to actual HTML/CSS via a coding LLM

Usage:
  python cleanup_training_data.py [--dir data/training_dataset] [--model rombodawg/rombos_Replete-Coder-Llama3-8B]
  python cleanup_training_data.py --video UDTvVTjKLIA        # single file
  python cleanup_training_data.py --skip-ui                  # only fix titles/scene descriptions
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. Audio garbage detection (shared with dataset_builder)
# ---------------------------------------------------------------------------

_MIN_UNIQUE_RATIO = 0.15  # unique_chars / total_chars below this = repetitive garbage

# Garbage prefix tokens the audio captioner sometimes prepends to valid content
_GARBAGE_PREFIX_RE = re.compile(r'^(?:audles?|aud\s)\s*', re.IGNORECASE)


def _clean_audio_description(text: str) -> str | None:
    """
    Strip known garbage prefix tokens, then validate the remaining content.
    Returns the cleaned string, or None if entirely garbage.
    """
    if not text:
        return None
    t = text.strip()
    if len(t) < 3:
        return None
    # Strip known garbage prefix (e.g. "audles" prepended directly before valid words)
    stripped = _GARBAGE_PREFIX_RE.sub('', t).strip()
    was_prefixed = stripped != t
    if not stripped or len(stripped) < 3:
        return None
    if was_prefixed:
        stripped = stripped[0].upper() + stripped[1:]
    s = stripped.lower()
    if len(s) > 10 and len(set(s)) / len(s) < _MIN_UNIQUE_RATIO:
        return None
    if re.search(r"(.{2,4})\1{4,}", s):
        return None
    return stripped


def _is_garbage_audio(text: str) -> bool:
    """Return True if the string cannot be salvaged as a valid audio description."""
    return _clean_audio_description(text) is None


# ---------------------------------------------------------------------------
# 1. Title mojibake fix
# ---------------------------------------------------------------------------

def fix_mojibake(text: str) -> str:
    """
    Fix titles with multi-level encoding corruption (UTF-8 bytes decoded as
    Windows-1252 one or more times). Uses ftfy which handles all common cases.
    """
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        pass
    # Fallback: single-pass cp1252 → utf-8
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# ---------------------------------------------------------------------------
# 2. Scene description cleanup
# ---------------------------------------------------------------------------

# "At 0s, " / "At 3.5s, " anywhere in text
_TIMESTAMP_RE = re.compile(r"\bAt \d+(?:\.\d+)?s,\s*", re.IGNORECASE)

# Audio annotation embedded via dash within a sentence, e.g.:
#   "...and a light, cheerful instrumental track plays-a simple upbeat melody
#    reminiscent of...-with no narration or dialogue."
# Captures the "and a <descriptor> <music-word> plays..." clause up to next period or
# the end of an em-dash-terminated clause.
_EMBEDDED_AUDIO_RE = re.compile(
    r"""
    (?:,\s*)?                           # optional leading comma+space
    \band\s+a\s+                        # "and a"
    (?:\w+[\s,]+){0,5}?                 # up to 5 descriptor words
    (?:instrumental|musical|melody|tune|track|score|theme|jingle|music)\s+
    (?:\w+\s+)*?plays                   # "[words] plays"
    [^.]*?                              # rest of audio clause
    (?=-\w|\.\s|\Z|,\s*[\w\s]+(?:plays|audio|music))  # lookahead to avoid eating visual text
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sentences that are predominantly about audio (standalone)
_AUDIO_SENTENCE_INDICATORS = re.compile(
    r"""
    \b(?:
        (?:the\s+)?audio\s+(?:feature|include|consist|play|stay|remain|throughout|sole)
        |an?\s+(?:upbeat|cheerful|whimsical|bouncy|playful|jazzy|festive|light|gentle|
                  catchy|simple|lively|uplifting)\s+(?:\w+\s+)*?(?:track|melody|tune|score|theme|jingle|piece)
        |(?:instrumental|musical)\s+(?:track|piece|score|theme|tune|composition)
        |(?:the\s+)?(?:music|bg\s+music|background\s+music)\s+(?:play|fill|create|stay|remain)
        |(?:no\s+)?(?:narration|dialogue)\s+(?:is\s+)?(?:present|audible|heard)
        |with\s+no\s+(?:narration|additional\s+sound|dialogue)
        |sound\s+effect
        |the\s+beat\s+(?:of|from)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Additional: "At Xs, the audio..." → entire sentence is audio-only
_AUDIO_AT_TS_RE = re.compile(
    r'\bAt \d+(?:\.\d+)?s,\s+[^.]*?\b(?:audio|music|melody|track|instrumental|sound)\b[^.]*\.',
    re.IGNORECASE,
)


def _split_sentences(text: str):
    """Simple sentence splitter on '. ', '! ', '? '."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p for p in parts if p.strip()]


def clean_scene_description(desc: str) -> str:
    """
    Remove timestamp markers and audio info from a scene description.
    """
    if not desc:
        return desc

    # Step 1: remove "At Xs, the audio..." sentences before tokenising
    text = _AUDIO_AT_TS_RE.sub("", desc)

    # Step 2: remove inline timestamp markers (keep the rest of the sentence)
    text = _TIMESTAMP_RE.sub("", text)

    # Step 3: remove embedded "and a [music] plays ..." clauses
    text = _EMBEDDED_AUDIO_RE.sub("", text)

    # Step 4: normalise whitespace artefacts (double spaces, leading commas)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^\s*[,;]\s*", "", text)
    text = re.sub(r"\s+([,;.])", r"\1", text)

    # Step 5: split to sentences and drop any that are audio-only
    sentences = _split_sentences(text)
    kept = [s for s in sentences if not _AUDIO_SENTENCE_INDICATORS.search(s)]

    result = " ".join(kept).strip()

    # Step 6: capitalise first letter if lost
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result or desc  # fall back to original if we over-stripped


# ---------------------------------------------------------------------------
# 3. Coding LLM for UI HTML generation
# ---------------------------------------------------------------------------

_UI_PROMPT = """\
Task: Act as a UI Developer. Your goal is to generate HTML/CSS for a Video Graphic Overlay based on a description.

Video Description:
{description}

Strict Extraction Rules:

Render ONLY UI/Text: Only generate HTML elements for text that is explicitly described as "text appearing over" or "on-screen titles."

Discard Scenery/People: Do NOT create <div>, <img>, or <span> elements for people, buildings, trees, or clothes. If it is a physical object in the video, ignore it.

Transparent Canvas: * Set html and body to margin: 0; padding: 0; background: transparent;.

The main container must be width: 100vw; height: 100vh; background: transparent; position: relative; overflow: hidden;.

UI Styling (Text Overlay):

Font: sans-serif, bold, uppercase.

Color: White.

Outline: Recreate the "black outline" using text-shadow: 3px 3px 0 #000, -3px -3px 0 #000, 3px -3px 0 #000, -3px 3px 0 #000;.

Placement: Position the text exactly where described (e.g., "appears over them" usually implies center-screen). Use display: flex or absolute positioning to center it.

No Hallucinations: Do not include external image URLs or placeholders for the "Person."

Output: Provide only the code.\
"""

_EMPTY_OVERLAY_HTML = (
    "<!DOCTYPE html><html><head><style>"
    "html,body{margin:0;padding:0;background:transparent;}"
    ".container{width:100vw;height:100vh;background:transparent;"
    "position:relative;overflow:hidden;}"
    "</style></head><body><div class=\"container\"></div></body></html>"
)


def _is_prose(text: str) -> bool:
    """Return True if the string is a prose description rather than HTML."""
    stripped = text.strip()
    if not stripped:
        return False
    return not stripped.startswith("<")


def _extract_html_block(text: str) -> str:
    """Extract HTML from a ```html ... ``` or ``` ... ``` fenced block, or return raw text."""
    m = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some models emit the code without fences
    if "<html" in text.lower() or "<!doctype" in text.lower():
        start = re.search(r"(?i)<!doctype|<html", text)
        if start:
            return text[start.start():].strip()
    return text.strip()


def _indicates_no_overlay(description: str) -> bool:
    """Return True when the prose explicitly says there are no text overlays."""
    markers = [
        "no additional text overlay",
        "no text overlay",
        "no overlay",
        "no on-screen text",
        "no title",
        "no caption",
        "no text appear",
        "no text is visible",
        "no visible text",
    ]
    desc_lower = description.lower()
    return any(m in desc_lower for m in markers)


class CodingLLM:
    """Wrapper around a causal LM loaded via transformers for HTML generation."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"\nLoading coding LLM: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[OK] Coding LLM loaded on {next(self.model.parameters()).device}")

    def generate_html(self, description: str) -> str:
        import torch

        if _indicates_no_overlay(description):
            return _EMPTY_OVERLAY_HTML

        prompt = _UI_PROMPT.format(description=description)

        # Use chat template if the model supports it
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _extract_html_block(raw) or _EMPTY_OVERLAY_HTML


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(path: Path, llm: "CodingLLM | None", dry_run: bool = False) -> bool:
    """
    Clean a single example JSON file.
    Returns True if it was modified.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False

    # --- Title fix ---
    title = data.get("input", {}).get("title", "")
    if title:
        fixed = fix_mojibake(title)
        if fixed != title:
            print(f"  Title:  {title!r}")
            print(f"       -> {fixed!r}")
            if not dry_run:
                data["input"]["title"] = fixed
            modified = True

    # --- Segment cleanup ---
    for seg in data.get("output", {}).get("segments", []):
        ts = seg.get("start_time", 0)

        # Scene description
        scene = seg.get("scene_description", "")
        cleaned_scene = clean_scene_description(scene)
        if cleaned_scene != scene:
            if not dry_run:
                seg["scene_description"] = cleaned_scene
            modified = True
            print(f"  [{ts:.1f}s] scene_description cleaned")

        # Audio description: remove garbage (repetitive hallucinations)
        audio = seg.get("audio_description", "")
        if audio:
            cleaned_audio = _clean_audio_description(audio)
            if cleaned_audio is None:
                print(f"  [{ts:.1f}s] audio_description removed (garbage: {audio[:60]!r})")
                if not dry_run:
                    seg.pop("audio_description", None)
                modified = True
            elif cleaned_audio != audio:
                print(f"  [{ts:.1f}s] audio_description fixed: {audio[:40]!r} → {cleaned_audio[:40]!r}")
                if not dry_run:
                    seg["audio_description"] = cleaned_audio
                modified = True

        # UI HTML: convert prose → HTML via LLM
        if llm is not None:
            ui = seg.get("ui_elements_html", "")
            if _is_prose(ui):
                print(f"  [{ts:.1f}s] Generating UI HTML from prose ({len(ui)} chars)...")
                if not dry_run:
                    html = llm.generate_html(ui)
                    seg["ui_elements_html"] = html
                modified = True

    if modified and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {path.name}")

    return modified


def main():
    parser = argparse.ArgumentParser(description="Clean training dataset JSON files")
    parser.add_argument(
        "--dir",
        default="data/training_dataset",
        help="Directory containing *_example.json files",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Process only this video ID (e.g. UDTvVTjKLIA)",
    )
    parser.add_argument(
        "--model",
        default="rombodawg/rombos_Replete-Coder-Llama3-8B",
        help="HuggingFace model ID for UI HTML generation",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip UI HTML generation (only fix titles and scene descriptions)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dir)
    if not dataset_dir.exists():
        print(f"[ERROR] Directory not found: {dataset_dir}")
        sys.exit(1)

    # Collect files
    if args.video:
        paths = list(dataset_dir.glob(f"{args.video}_example.json"))
        if not paths:
            # Try without suffix
            paths = list(dataset_dir.glob(f"{args.video}*.json"))
        if not paths:
            print(f"[ERROR] No file found for video: {args.video}")
            sys.exit(1)
    else:
        paths = sorted(dataset_dir.glob("*_example.json"))

    print(f"Processing {len(paths)} file(s) in {dataset_dir}")
    if args.dry_run:
        print("[DRY RUN — no files will be written]")

    # Load LLM only if needed
    llm = None
    if not args.skip_ui:
        llm = CodingLLM(args.model)

    modified_count = 0
    for path in paths:
        print(f"\n[{path.name}]")
        changed = process_file(path, llm=llm, dry_run=args.dry_run)
        if changed:
            modified_count += 1

    print(f"\nDone. {modified_count}/{len(paths)} files modified.")


if __name__ == "__main__":
    main()
