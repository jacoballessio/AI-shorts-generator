"""
Generate synthetic training examples in the exact *_example.json format.

Supports:
- OpenAI API
- Local Hugging Face transformers inference

Usage:
  python src/training/generate_synthetic_examples.py \
    --count 200 \
    --creator "Daniel Hentschel" \
    --topic "People & Blogs" \
    --out-dir data/training_dataset_synth

Then pre-run training on synthetic data, followed by real data:
  python src/training/train_carmind_lora.py --data data/training_dataset_synth --output models/carmind2_lora_synth
  python src/training/train_carmind_lora.py --data data/training_dataset --output models/carmind2_lora --epochs 1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI

load_dotenv()


def load_seed_examples(seed_dir: Path, creator: str, topic: str, limit: int) -> list[dict[str, Any]]:
    """Load a few existing examples as style references."""
    if limit <= 0 or not seed_dir.exists():
        return []

    paths = sorted(seed_dir.glob("*_example.json"))
    matches: list[dict[str, Any]] = []

    for p in paths:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        inp = obj.get("input", {})
        c = str(inp.get("creator", "")).strip().lower()
        t = str(inp.get("topic", "")).strip().lower()
        if (not creator or c == creator.lower()) and (not topic or t == topic.lower()):
            matches.append(obj)
        if len(matches) >= limit:
            break

    return matches


def load_seed_pool(seed_dir: Path) -> list[dict[str, Any]]:
    """Load all seed examples for random creator/topic sampling."""
    if not seed_dir.exists():
        return []

    pool: list[dict[str, Any]] = []
    for p in sorted(seed_dir.glob("*_example.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("input"), dict):
            pool.append(obj)
    return pool


def pick_creator_topic(pool: list[dict[str, Any]], fallback_creator: str, fallback_topic: str) -> tuple[str, str]:
    """Pick creator/topic randomly from seed pool if not explicitly provided."""
    if pool:
        pick = random.choice(pool)
        inp = pick.get("input", {})
        creator = str(inp.get("creator", "")).strip() or fallback_creator
        topic = str(inp.get("topic", "")).strip() or fallback_topic
        if creator and topic:
            return creator, topic
    return fallback_creator or "Unknown Creator", fallback_topic or "General"


def pick_duration(pool: list[dict[str, Any]], fallback_duration: int | None) -> int:
    """Pick duration from seed pool when not explicitly provided."""
    if fallback_duration is not None and fallback_duration > 0:
        return int(fallback_duration)

    durations: list[int] = []
    for ex in pool:
        try:
            d = int(ex.get("input", {}).get("duration", 0))
        except Exception:
            d = 0
        if 5 <= d <= 120:
            durations.append(d)

    if durations:
        return random.choice(durations)

    # Fallback when no seeds are available.
    return random.randint(15, 60)


def make_seed_snippet(seeds: list[dict[str, Any]]) -> str:
    """Build compact seed snippet for prompting."""
    if not seeds:
        return "No seed examples provided."

    compact = []
    for ex in seeds:
        inp = ex.get("input", {})
        out = ex.get("output", {})
        compact.append(
            {
                "input": {
                    "duration": inp.get("duration", 20),
                    "creator": inp.get("creator", ""),
                    "topic": inp.get("topic", ""),
                    "title": inp.get("title", ""),
                    "sources": inp.get("sources", []),
                },
                "output": {
                    "segments": out.get("segments", [])[:2]
                },
            }
        )

    return json.dumps(compact, ensure_ascii=False, indent=2)


def validate_example(obj: dict[str, Any]) -> tuple[bool, str]:
    """Validate strict training-example structure."""
    if not isinstance(obj, dict):
        return False, "Top-level must be object"

    if "input" not in obj or "output" not in obj:
        return False, "Missing input/output"

    inp = obj["input"]
    out = obj["output"]
    if not isinstance(inp, dict) or not isinstance(out, dict):
        return False, "input/output must be objects"

    for key in ["sources", "duration", "creator", "topic", "title"]:
        if key not in inp:
            return False, f"input.{key} missing"

    if not isinstance(inp["sources"], list):
        return False, "input.sources must be array"

    if "segments" not in out or not isinstance(out["segments"], list) or len(out["segments"]) == 0:
        return False, "output.segments must be non-empty array"

    required_seg = ["start_time", "end_time", "scene_description", "ui_elements_html", "transcript", "duration"]
    prev_end = -1.0
    for i, seg in enumerate(out["segments"]):
        if not isinstance(seg, dict):
            return False, f"segment[{i}] must be object"
        for key in required_seg:
            if key not in seg:
                return False, f"segment[{i}].{key} missing"

        try:
            st = float(seg["start_time"])
            et = float(seg["end_time"])
            dur = float(seg["duration"])
        except Exception:
            return False, f"segment[{i}] time fields must be numeric"

        if st < 0 or et <= st or dur <= 0:
            return False, f"segment[{i}] invalid time bounds"
        if abs((et - st) - dur) > 0.25:
            return False, f"segment[{i}] duration mismatch"
        if st + 1e-6 < prev_end:
            return False, f"segment[{i}] not monotonic"
        prev_end = et

    return True, "ok"


def normalize_example(obj: dict[str, Any], synthetic_id: str) -> dict[str, Any]:
    """Normalize optional fields and metadata."""
    if "metadata" not in obj or not isinstance(obj["metadata"], dict):
        obj["metadata"] = {}

    meta = obj["metadata"]
    if not meta.get("video_id"):
        meta["video_id"] = synthetic_id
    if "upload_date" not in meta:
        meta["upload_date"] = "20260318"
    if "view_count" not in meta:
        meta["view_count"] = 0

    # Keep sources clean and bounded for easier schema learning.
    cleaned_sources = []
    for src in obj.get("input", {}).get("sources", []):
        if not isinstance(src, dict):
            continue
        cleaned_sources.append(
            {
                "title": str(src.get("title", ""))[:120],
                "description": str(src.get("description", ""))[:400],
                "text": str(src.get("text", ""))[:800],
                "url": str(src.get("url", ""))[:300],
                "type": str(src.get("type", "article"))[:40],
                "source_name": str(src.get("source_name", ""))[:80],
                "published_at": str(src.get("published_at", ""))[:40],
            }
        )
    obj["input"]["sources"] = cleaned_sources

    return obj


def extract_json_object_text(text: str) -> str:
    """Extract first JSON object from model output, handling code fences/noise."""
    if not text:
        return "{}"

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    # Fast path
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Find first balanced object.
    start = stripped.find("{")
    if start < 0:
        return stripped

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]

    return stripped


def make_openai_client():
    """Create OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Install with: pip install openai") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def make_hf_text_generator(
    model: str,
    trust_remote_code: bool = False,
    device_map: str = "auto",
):
    """Create local Hugging Face text-generation pipeline."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install with: pip install transformers") from exc

    # Windows symlink warning is noisy and non-fatal.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    resolved_device_map = device_map
    if device_map in ("cuda", "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("--hf-device-map cuda requested, but CUDA is not available.")
        resolved_device_map = {"": 0}

    def _load_tokenizer_with_retry(model_id: str):
        try:
            return AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )
        except Exception as tok_err:
            # Some transformers/huggingface_hub combos can fail on remote
            # additional_chat_templates lookup (404). Retry from local snapshot.
            if "additional_chat_templates" not in str(tok_err):
                raise
            try:
                from huggingface_hub import snapshot_download

                local_dir = snapshot_download(
                    repo_id=model_id,
                    token=hf_token,
                )
                return AutoTokenizer.from_pretrained(
                    local_dir,
                    trust_remote_code=trust_remote_code,
                    local_files_only=True,
                )
            except Exception:
                raise tok_err

    try:
        return pipeline(
            task="text-generation",
            model=model,
            dtype="auto",
            device_map=resolved_device_map,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
    except Exception as first_error:
        # Fallback for some Mistral variants with incomplete config (e.g., missing head_dim).
        try:
            config = AutoConfig.from_pretrained(
                model,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )

            if getattr(config, "head_dim", None) is None:
                hidden_size = getattr(config, "hidden_size", None)
                num_attention_heads = getattr(config, "num_attention_heads", None)
                if hidden_size and num_attention_heads:
                    config.head_dim = hidden_size // num_attention_heads

            tokenizer = _load_tokenizer_with_retry(model)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                config=config,
                dtype="auto",
                device_map=resolved_device_map,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )
            return pipeline("text-generation", model=model_obj, tokenizer=tokenizer)
        except Exception as second_error:
            raise RuntimeError(
                "Failed to load Hugging Face model for local generation. "
                "This checkpoint may be incompatible with your installed transformers version or requires custom code. "
                "Try --trust-remote-code, use a different instruct model, or update transformers. "
                f"First error: {first_error}; Fallback error: {second_error}"
            ) from second_error


def generate_one_openai(
    client,
    model: str,
    creator: str,
    topic: str,
    duration: int,
    seed_snippet: str,
    synthetic_id: str,
) -> dict[str, Any]:
    """Generate one synthetic example via OpenAI JSON mode."""
    sys = (
        "You generate synthetic short-video training examples. "
        "Return ONLY valid JSON with exact schema: "
        "{input:{sources,duration,creator,topic,title},output:{segments:[...]},metadata:{video_id,upload_date,view_count}}. "
        "Do not include markdown. Do not include chat logs, XML, HTML tags, usernames, or profanity. "
        "Segments must be temporally consistent and sum close to duration."
    )

    user = (
        f"Create one training example for creator '{creator}' on topic '{topic}' with duration {duration} seconds.\n"
        "Important quality constraints:\n"
        "- 4 to 8 segments\n"
        "- Each segment has: start_time,end_time,scene_description,ui_elements_html,transcript,duration\n"
        "- Use empty string for ui_elements_html unless needed\n"
        "- Keep sources realistic and concise (or empty list)\n"
        "- Keep transcript coherent and safe\n"
        "- Ensure strict JSON\n\n"
        f"Seed examples (style reference only):\n{seed_snippet}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.7,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )

    text = resp.choices[0].message.content or "{}"
    json_text = extract_json_object_text(text)
    obj = json.loads(json_text)

    # Some models may nest under "example".
    if isinstance(obj, dict) and "example" in obj and isinstance(obj["example"], dict):
        obj = obj["example"]

    ok, reason = validate_example(obj)
    if not ok:
        raise ValueError(f"Validation failed: {reason}")

    obj = normalize_example(obj, synthetic_id)
    return obj


def _build_segment_timeline(total_duration: int, segment_count: int) -> list[tuple[float, float]]:
    """Build deterministic monotonic timeline with slight randomization."""
    if segment_count <= 0:
        return [(0.0, float(total_duration))]

    weights = [random.uniform(0.85, 1.2) for _ in range(segment_count)]
    weight_sum = sum(weights)
    raw = [float(total_duration) * w / weight_sum for w in weights]

    starts_ends: list[tuple[float, float]] = []
    cursor = 0.0
    for i, d in enumerate(raw):
        if i == segment_count - 1:
            end = float(total_duration)
        else:
            end = round(cursor + d, 2)
            if end <= cursor:
                end = round(cursor + 0.8, 2)
        starts_ends.append((round(cursor, 2), round(end, 2)))
        cursor = end

    # Ensure final end is exact and monotonic.
    starts_ends[-1] = (starts_ends[-1][0], float(total_duration))
    fixed: list[tuple[float, float]] = []
    prev_end = 0.0
    for i, (start, end) in enumerate(starts_ends):
        start = max(start, prev_end)
        if i < len(starts_ends) - 1:
            end = max(end, round(start + 0.5, 2))
        else:
            end = float(total_duration)
        fixed.append((round(start, 2), round(end, 2)))
        prev_end = end
    return fixed


def _extract_tag_value(text: str, tag_name: str) -> str:
    """Extract a simple XML-like tag value, case-insensitive."""
    m = re.search(rf"<{tag_name}>\\s*(.*?)\\s*</{tag_name}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _ensure_speaker_prefix(transcript: str, speaker_label: str) -> str:
    """Ensure transcript starts with [SPEAKER_X] tag."""
    cleaned = re.sub(r"^\s*\[SPEAKER_\d+\]\s*", "", transcript or "").strip()
    if not cleaned:
        return f"[{speaker_label}]"
    return f"[{speaker_label}] {cleaned}"


def _trim_prompt_echo(text: str) -> str:
    """Remove common prompt-echo prefixes from local-model outputs."""
    cleaned = text.strip()
    markers = ["ANSWER:", "Output:", "Response:", "Assistant:"]
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1].strip()

    prefix_patterns = [
        r"^The title is:\s*",
        r"^Scene description:\s*",
        r"^Transcript:\s*",
        r"^The scene description for scene \d+.*?:\s*",
    ]
    for pattern in prefix_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def _looks_like_bad_generation(text: str, field_kind: str) -> bool:
    """Detect prompt echo, instructions, and pathological low-quality outputs."""
    if not text or not text.strip():
        return True

    lowered = text.lower()
    bad_markers = [
        "return only",
        "no quotes",
        "create the transcript",
        "create the scene description",
        "create a short video title",
        "time window:",
        "topic:",
        "creator:",
        "scene ",
        "no punctuation",
    ]
    if any(marker in lowered for marker in bad_markers):
        return True

    if field_kind == "title":
        return len(text.split()) > 14

    if field_kind == "scene":
        return len(text) > 180 or text.count(".") > 3

    if field_kind == "transcript":
        words = text.split()
        if len(words) < 3 or len(words) > 28:
            return True
        if len(set(words[: min(8, len(words))])) <= 2:
            return True

    return False


def _run_hf_prompt(
    hf_generator,
    prompt: str,
    max_new_tokens: int,
    field_kind: str,
    temperature: float = 0.25,
    do_sample: bool = False,
    retries: int = 3,
) -> str:
    """Run one Hugging Face prompt and return a cleaned single-line response."""
    last_text = ""
    for attempt in range(retries):
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_full_text": False,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature if attempt == 0 else 0.1
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        outputs = hf_generator(prompt, **gen_kwargs)
        text = ""
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                text = str(first.get("generated_text", ""))

        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[0]

        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()

        text = _trim_prompt_echo(re.sub(r"\s+", " ", text).strip())
        last_text = text
        if not _looks_like_bad_generation(text, field_kind):
            return text

    return last_text


def _run_hf_prompts_batch(
    hf_generator,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    batch_size: int,
) -> list[str]:
    """Run a batch of prompts through HF pipeline and return cleaned first-line outputs."""
    if not prompts:
        return []

    outputs = hf_generator(
        prompts,
        **({
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_full_text": False,
            "batch_size": max(1, int(batch_size)),
        } | ({
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
        } if do_sample else {}))
    )

    results: list[str] = []
    for out in outputs:
        item = out
        if isinstance(item, list) and item:
            item = item[0]
        text = ""
        if isinstance(item, dict):
            text = str(item.get("generated_text", ""))
        text = _trim_prompt_echo(re.sub(r"\s+", " ", (text or "").strip()))
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        results.append(text)
    return results


def generate_one_huggingface(
    hf_generator,
    creator: str,
    topic: str,
    duration: int,
    seed_snippet: str,
    synthetic_id: str,
    max_new_tokens: int,
    min_segments: int,
    max_segments: int,
    hf_temperature: float,
    hf_do_sample: bool,
    hf_retries: int,
    hf_batch_size: int,
) -> dict[str, Any]:
    """Generate one synthetic example via local Hugging Face inference.

    Strategy: code owns schema/timings; LLM fills each field individually.
    """
    duration_based_max = max(4, duration // 2 if duration >= 8 else 4)
    dynamic_max = max(4, min(8, duration_based_max, max_segments))
    dynamic_min = max(2, min(min_segments, dynamic_max))
    segment_count = random.randint(dynamic_min, dynamic_max)
    timeline = _build_segment_timeline(duration, segment_count)

    title_prompt = (
        "Task: write only the final title text.\n"
        "Bad: The title is: Election breakdown\n"
        "Good: Election Issues Explained\n"
        f"Creator: {creator}\n"
        f"Topic: {topic}\n"
        "Rules: max 8 words, no quotes, no prefix, no explanation.\n"
        "ANSWER:"
    )
    title = _run_hf_prompt(
        hf_generator,
        title_prompt,
        max_new_tokens=min(30, max_new_tokens),
        field_kind="title",
        temperature=hf_temperature,
        do_sample=hf_do_sample,
        retries=hf_retries,
    )
    if _looks_like_bad_generation(title, "title"):
        rescue_title_prompt = (
            f"Write only a clean short title for a {topic} video by {creator}. "
            "Output title only. No prefix, no quotes, max 8 words. ANSWER:"
        )
        title = _run_hf_prompt(
            hf_generator,
            rescue_title_prompt,
            max_new_tokens=min(24, max_new_tokens),
            field_kind="title",
            temperature=max(0.05, hf_temperature),
            do_sample=False,
            retries=max(2, hf_retries),
        )
    if _looks_like_bad_generation(title, "title"):
        raise ValueError("Could not generate a valid LLM title")
    title = title[:120]

    # Sources are optional for synthetic generation; keep deterministic empty list.
    sources = []

    # Transcript fallback remains to avoid dropping complete samples on single-line failures.
    fallback_transcript_templates = [
        "Quick context before we dive in.",
        "Here is the key point most people miss.",
        "Now watch what changes when we do it this way.",
        "This is the part that makes the biggest difference.",
        "If you only remember one thing, remember this.",
        "That is the full breakdown in simple terms.",
    ]

    speaker_pool_size = random.choice([1, 2])
    previous_transcript_tail = ""

    scene_prompts: list[str] = []
    for idx, (start, end) in enumerate(timeline, start=1):
        scene_prompt = (
            "Task: write only the scene description sentence.\n"
            "Bad: The scene description for scene 2 is a creator talking to camera.\n"
            "Good: Parker speaks directly to camera while introducing the policy debate.\n"
            f"Video title: {title}\n"
            f"Creator: {creator}\n"
            f"Topic: {topic}\n"
            f"Scene number: {idx} of {segment_count}\n"
            f"Time window: {start:.2f}s to {end:.2f}s\n"
            "Rules: one concise sentence, no quotes, no meta commentary, no instruction text.\n"
            "ANSWER:"
        )
        scene_prompts.append(scene_prompt)

    scene_outputs = _run_hf_prompts_batch(
        hf_generator,
        scene_prompts,
        max_new_tokens=min(45, max_new_tokens),
        temperature=hf_temperature,
        do_sample=hf_do_sample,
        batch_size=hf_batch_size,
    )

    segments = []
    for idx, (start, end) in enumerate(timeline, start=1):
        scene = scene_outputs[idx - 1] if idx - 1 < len(scene_outputs) else ""
        if _looks_like_bad_generation(scene, "scene"):
            rescue_scene_prompt = (
                "Write only one concise scene description sentence. "
                f"Video title: {title}. Creator: {creator}. Topic: {topic}. "
                f"Scene {idx}/{segment_count}, {start:.2f}s to {end:.2f}s. "
                "No prefix, no quotes, no instruction text. ANSWER:"
            )
            scene = _run_hf_prompt(
                hf_generator,
                rescue_scene_prompt,
                max_new_tokens=min(40, max_new_tokens),
                field_kind="scene",
                temperature=max(0.05, hf_temperature),
                do_sample=False,
                retries=max(2, hf_retries),
            )
        if _looks_like_bad_generation(scene, "scene"):
            raise ValueError(f"Could not generate valid LLM scene description for segment {idx}")

        speaker_idx = 1 if speaker_pool_size == 1 else (1 if idx % 2 == 1 else 2)
        if previous_transcript_tail:
            transcript_prompt = (
                "Task: write only the spoken transcript line.\n"
                f"Video title: {title}\n"
                f"Scene number: {idx} of {segment_count}\n"
                f"Continue naturally from these ending words: {previous_transcript_tail}\n"
                f"Rules: one sentence, 8-20 words, start exactly with [SPEAKER_{speaker_idx}], no quotes, no instruction text.\n"
                f"Example good output: [SPEAKER_{speaker_idx}] That is where the story starts to change for everyone involved.\n"
                "ANSWER:"
            )
        else:
            transcript_prompt = (
                "Task: write only the spoken transcript line.\n"
                f"Video title: {title}\n"
                f"Scene number: {idx} of {segment_count}\n"
                f"Rules: one sentence, 8-20 words, start exactly with [SPEAKER_{speaker_idx}], no quotes, no instruction text.\n"
                f"Example good output: [SPEAKER_{speaker_idx}] Here is the quick version of what happened today.\n"
                "ANSWER:"
            )

        transcript = _run_hf_prompt(
            hf_generator,
            transcript_prompt,
            max_new_tokens=min(40, max_new_tokens),
            field_kind="transcript",
            temperature=hf_temperature,
            do_sample=hf_do_sample,
            retries=hf_retries,
        )
        if not transcript:
            transcript = fallback_transcript_templates[(idx - 1) % len(fallback_transcript_templates)]
        transcript = _ensure_speaker_prefix(transcript, f"SPEAKER_{speaker_idx}")

        cleaned_transcript = re.sub(r"^\s*\[SPEAKER_\d+\]\s*", "", transcript).strip()
        tail_words = cleaned_transcript.split()
        previous_transcript_tail = " ".join(tail_words[-4:]) if tail_words else ""

        seg_duration = round(float(end) - float(start), 2)
        segments.append(
            {
                "start_time": float(start),
                "end_time": float(end),
                "scene_description": scene,
                "ui_elements_html": "",
                "transcript": transcript,
                "duration": float(seg_duration),
            }
        )

    obj = {
        "input": {
            "sources": sources,
            "duration": int(duration),
            "creator": creator,
            "topic": topic,
            "title": title,
        },
        "output": {
            "segments": segments,
        },
        "metadata": {
            "video_id": synthetic_id,
            "upload_date": "20260318",
            "view_count": 0,
        },
    }

    ok, reason = validate_example(obj)
    if not ok:
        raise ValueError(f"Validation failed after template fill: {reason}")

    obj = normalize_example(obj, synthetic_id)
    return obj


def sanitize_filename(name: str) -> str:
    """Create safe file name chunk."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:40] or "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic *_example.json training data using OpenAI or local Hugging Face inference")
    parser.add_argument("--count", type=int, required=True, help="Number of examples to generate")
    parser.add_argument(
        "--creator",
        type=str,
        default="",
        help="Creator name (optional; random from --seed-dir if omitted)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Topic/category (optional; random from --seed-dir if omitted)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Fixed duration seconds (optional; random from --seed-dir by default)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "huggingface"],
        help="LLM provider: openai or huggingface",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name for selected provider")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face models that require custom remote code",
    )
    parser.add_argument(
        "--hf-device-map",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="HF model placement: auto (may offload), cuda (force GPU), or cpu",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1800, help="Max generated tokens for Hugging Face provider")
    parser.add_argument("--hf-min-segments", type=int, default=4, help="Minimum segments per synthetic example (HF mode)")
    parser.add_argument("--hf-max-segments", type=int, default=6, help="Maximum segments per synthetic example (HF mode)")
    parser.add_argument("--hf-temperature", type=float, default=0.15, help="HF decoding temperature")
    parser.add_argument("--hf-retries", type=int, default=1, help="Retries per HF field if output quality is poor")
    parser.add_argument("--hf-batch-size", type=int, default=8, help="Batch size for HF batched scene generation")
    parser.add_argument(
        "--hf-do-sample",
        action="store_true",
        help="Enable sampling in HF generation (greedy by default for speed/stability)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_dataset_synth"), help="Output directory")
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("data/training_dataset"),
        help=(
            "Directory of real *_example.json files used as style references and random "
            "creator/topic source"
        ),
    )
    parser.add_argument("--seed-limit", type=int, default=8, help="How many seed examples to include")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per sample")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel generation workers (default: 1)",
    )
    args = parser.parse_args()

    client = None
    hf_generator = None
    if args.provider == "openai":
        client = make_openai_client()
    else:
        hf_generator = make_hf_text_generator(
            args.model,
            trust_remote_code=args.trust_remote_code,
            device_map=args.hf_device_map,
        )
        if args.workers > 1:
            print("[INFO] Hugging Face provider uses a single local model instance; forcing --workers=1 for stability.")
            args.workers = 1

    seed_pool = load_seed_pool(args.seed_dir)
    if not seed_pool and (not args.creator or not args.topic):
        raise RuntimeError(
            "No seed examples found in --seed-dir. Provide --creator and --topic explicitly "
            "or point --seed-dir to a dataset with *_example.json files."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_filename(f"{args.creator or 'random'}_{args.topic or 'random'}")

    run_stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    id_lock = threading.Lock()
    id_counter = {"value": 0}

    def next_synthetic_id() -> str:
        with id_lock:
            id_counter["value"] += 1
            return f"synthetic_{run_stamp}_{id_counter['value']:05d}"

    def process_index(idx: int) -> tuple[bool, int, str]:
        sample_creator, sample_topic = pick_creator_topic(seed_pool, args.creator, args.topic)
        sample_duration = pick_duration(seed_pool, args.duration)
        seeds = load_seed_examples(args.seed_dir, sample_creator, sample_topic, args.seed_limit)
        seed_snippet = make_seed_snippet(seeds)
        synthetic_id = next_synthetic_id()

        last_err = ""
        retries_used = 0
        for attempt in range(1, args.max_retries + 1):
            retries_used = attempt
            try:
                obj = generate_one_openai(
                    client=client,
                    model=args.model,
                    creator=sample_creator,
                    topic=sample_topic,
                    duration=sample_duration,
                    seed_snippet=seed_snippet,
                    synthetic_id=synthetic_id,
                ) if args.provider == "openai" else generate_one_huggingface(
                    hf_generator=hf_generator,
                    creator=sample_creator,
                    topic=sample_topic,
                    duration=sample_duration,
                    seed_snippet=seed_snippet,
                    synthetic_id=synthetic_id,
                    max_new_tokens=args.max_new_tokens,
                    min_segments=args.hf_min_segments,
                    max_segments=args.hf_max_segments,
                    hf_temperature=args.hf_temperature,
                    hf_do_sample=args.hf_do_sample,
                    hf_retries=args.hf_retries,
                    hf_batch_size=args.hf_batch_size,
                )
                file_path = args.out_dir / f"{base_name}_{synthetic_id}_example.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
                return (
                    True,
                    retries_used,
                    f"{file_path} [{sample_creator} | {sample_topic} | {sample_duration}s]",
                )
            except Exception as exc:
                last_err = str(exc)

        return False, retries_used, f"sample {idx}: {last_err}"

    indices = list(range(1, args.count + 1))
    created = 0
    attempted = 0
    max_workers = max(1, int(args.workers))

    if max_workers == 1:
        for i, idx in enumerate(indices, start=1):
            ok, attempts_used, msg = process_index(idx)
            attempted += attempts_used
            if ok:
                created += 1
                print(f"[OK] {created}/{args.count} -> {msg}")
            else:
                print(f"[WARN] Failed {msg}")
    else:
        print(f"Using {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(process_index, idx): idx for idx in indices}
            for fut in as_completed(future_map):
                ok, attempts_used, msg = fut.result()
                attempted += attempts_used
                if ok:
                    created += 1
                    print(f"[OK] {created}/{args.count} -> {msg}")
                else:
                    print(f"[WARN] Failed {msg}")

    print(f"\nDone. Created {created} examples (attempted {attempted} generation attempts).")
    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
