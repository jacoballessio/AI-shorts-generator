"""
Generate a training XML-ish example using a local fine-tuned model.

Input fields:
- sources (from --sources-file JSON and/or repeated --source)
- duration
- creator
- topic

Output:
- Writes an XML-ish file with schema:
    <example>
        <input>...</input>
        <output><segments>...</segments></output>
    </example>
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import html
import inspect
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
import random
try:
    from src.training.convert_xmlish_to_examples import (
        _parse_example as _xe_parse_example,
        _parse_segment_fragment as _xe_parse_fragment,
        _normalize_spaced_xml_tags as _xe_normalize_tags,
        EXAMPLE_BLOCK_RE as _XE_EXAMPLE_BLOCK_RE,
    )
    _XMLISH_JSON_AVAILABLE = True
except ImportError:
    _XMLISH_JSON_AVAILABLE = False

DEFAULT_BASE_MODEL = "SuperAGI/mistral-7B-PoSE-32k"

_NLP_SCORER_TOKENIZER: Any | None = None
_NLP_SCORER_MODEL: Any | None = None
_CLOUD_ENDPOINT_CFG: dict[str, Any] | None = None

torch = None
snapshot_download = None
PeftModel = None
AutoConfig = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None


def _ensure_ml_imports() -> None:
    """Lazily import heavy ML dependencies required only for local generation."""
    global torch, snapshot_download, PeftModel
    global AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if torch is not None:
        return

    import torch as _torch
    from huggingface_hub import snapshot_download as _snapshot_download
    from peft import PeftModel as _PeftModel
    from transformers import (
        AutoConfig as _AutoConfig,
        AutoModelForCausalLM as _AutoModelForCausalLM,
        AutoTokenizer as _AutoTokenizer,
        BitsAndBytesConfig as _BitsAndBytesConfig,
    )

    torch = _torch
    snapshot_download = _snapshot_download
    PeftModel = _PeftModel
    AutoConfig = _AutoConfig
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    BitsAndBytesConfig = _BitsAndBytesConfig

_ALLOWED_XMLISH_TAGS = {
    "example",
    "topic",
    "creator",
    "title",
    "input",
    "output",
    "segments",
    "segment",
    "start",
    "end",
    "duration",
    "scene",
    "transcript",
    "audio",
    "ui_description",
}


def _configure_stdio() -> None:
    """Prefer UTF-8 console output on Windows so logging generated text cannot crash."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def fix_punctuation_spacing(text: str) -> str:
    """Fix spacing around punctuation marks.
    
    Fixes common generation artifacts:
    - "Hello.World" -> "Hello. World"
    - "Word,another" -> "Word, another"
    - ".The text" -> ". The text"
    - 'saying"Sorry' -> 'saying "Sorry'
    - Multiple spaces -> single space
    """
    if not text:
        return text
    
    # Fix space before punctuation (remove extra spaces)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Add space after period/question mark/exclamation followed by capital letter (if missing)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Fix space after other punctuation followed by lowercase (common in generated text)
    text = re.sub(r'([.!?])([a-z])', r'\1 \2', text)
    
    # Fix space after comma if missing
    text = re.sub(r',([A-Za-z])', r', \1', text)
    
    # Fix space before/after quotes
    text = re.sub(r'\s*"\s*', ' "', text)
    text = re.sub(r'"\s+([.,:!?])', r'"\1', text)
    text = re.sub(r'"([A-Za-z])', r'" \1', text)
    
    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix ]. and ][ patterns
    text = re.sub(r'\]\s*\[', '][', text)
    
    return text.strip()


def compact_sources(sources: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Build a bounded source summary so generation prompt stays manageable."""
    blocks: list[str] = []
    for idx, source in enumerate(sources, start=1):
        title = str(source.get("title", "")).strip() or f"Source {idx}"
        text = str(source.get("text", "")).strip()
        description = str(source.get("description", "")).strip()
        body = text if text else description
        if not body:
            body = "(No source text available)"
        blocks.append(f"[{idx}] {title}\\n{body}")

    joined = "\\n\\n".join(blocks)
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 3] + "..."


def apply_chat_template_safe(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply chat template when available, otherwise use plain fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        chunks: list[str] = []
        for message in messages:
            role = message["role"].upper()
            chunks.append(f"{role}:\\n{message['content']}")
        chunks.append("ASSISTANT:\n")
        return "\\n\\n".join(chunks)


def normalize_source(raw: Any) -> dict[str, str]:
    """Normalize source entries to the training schema."""
    if isinstance(raw, str):
        url = raw.strip()
        return {
            "title": "",
            "description": "",
            "text": "",
            "url": url,
            "type": "article",
            "source_name": "",
            "published_at": "",
        }

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported source entry type: {type(raw)}")

    return {
        "title": str(raw.get("title", "")),
        "description": str(raw.get("description", "")),
        "text": str(raw.get("text", "")),
        "url": str(raw.get("url", "")),
        "type": str(raw.get("type", raw.get("source_type", "article"))),
        "source_name": str(raw.get("source_name", "")),
        "published_at": str(raw.get("published_at", raw.get("publish_date", ""))),
    }


def load_sources(sources_file: Path | None, inline_sources: list[str]) -> list[dict[str, str]]:
    """Load and normalize sources from JSON file plus repeated --source values."""
    all_sources: list[dict[str, str]] = []

    if sources_file:
        with open(sources_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("--sources-file must contain a JSON list")
        all_sources.extend(normalize_source(item) for item in payload)

    all_sources.extend(normalize_source(item) for item in inline_sources)

    filtered = []
    for source in all_sources:
        if source["url"] or source["text"] or source["description"]:
            filtered.append(source)

    # Allow generating examples with zero sources
    if not filtered:
        return []  # Return an empty list instead of raising an error

    return filtered


def _xml_text(value: Any) -> str:
    """Escape element text for XML-like output."""
    return html.escape(str(value), quote=False)


def _xml_attr(value: Any) -> str:
    """Escape attribute value for XML-like output."""
    return html.escape(str(value), quote=True)


def _escape_bare_ampersands(text: str) -> str:
    """Escape '&' that are not already part of XML/HTML entities."""
    return re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)", "&amp;", text)


def build_xmlish_prefix_prompt(
    creator: str,
    topic: str,
    duration: int,
    sources: list[dict[str, str]],
    title: str,
) -> str:
    """Build XML-ish prefix that the model should continue with <segment> blocks."""
    lines: list[str] = ["<example>", "<input>"]
    creator_xml = _xml_text(creator)
    topic_xml = _xml_text(topic)
    if duration is None:
        duration = random.randint(5, 30)
    if title:
        lines.extend(
            [
                f"<duration>{duration}</duration>",
                f"<creator>{creator_xml}</creator>",
                f"<topic>{topic_xml}</topic>",
                f"<title>{title}</title>",
                "</input>",
                "<output>",
                "<segments>",
            ]
        )
    elif creator:
        if topic:
            lines.extend(
                [
                    f"<duration>{duration}</duration>",
                    f"<creator>{creator_xml}</creator>",
                    f"<topic>{topic_xml}</topic>",
                    f"<title>"
                ]
            )
        else:
            lines.extend(
                [
                    f"<duration>{duration}</duration>",
                    f"<creator>{creator_xml}</creator>",
                ]
            )
    elif duration:
        lines.append(f"<duration>{duration}</duration>")
    else:
        pass

    return "\n".join(lines) + "\n"


def _sanitize_generated_title(text: str, *, fallback_topic: str) -> str:
    """Normalize a raw model completion into a short single-line title."""
    raw = (text or "").strip()
    if not raw:
        return f"{fallback_topic} Short" if fallback_topic else "Untitled Short"

    raw = raw.replace("\r", "\n")
    raw = _strip_markdown_fence(raw)
    raw = re.sub(r"<\s*/?\s*(example|input|output|segments?|segment|duration|creator|topic|title|scene|transcript|audio)\s*>", " ", raw, flags=re.IGNORECASE)
    raw = raw.split("\n", 1)[0].strip()
    raw = re.sub(r"^\s*title\s*:\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.strip(" \t\r\n\"'`[](){}<>:;,.!?")
    raw = re.sub(r"\s+", " ", raw).strip()

    if not raw:
        return f"{fallback_topic} Short" if fallback_topic else "Untitled Short"

    if len(raw) > 100:
        raw = raw[:100].rsplit(" ", 1)[0].strip() or raw[:100].strip()

    return raw


def generate_title_with_llm(
    model: Any,
    tokenizer: Any,
    *,
    creator: str,
    topic: str,
    duration: int | None,
    sources: list[dict[str, str]],
) -> str:
    """Generate a short title with the loaded LLM when none is provided."""
    source_summary = compact_sources(sources, max_chars=1600) if sources else ""
    random_letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    prompt_parts = [
        "<example>",
        "<input>",
        "<duration>" + str(duration) + "</duration>",
        "<creator>" + creator + "</creator>",
        "<topic>" + topic + "</topic>",
        "<title>"+random_letter,
    ]
    if source_summary:
        prompt_parts.append("Relevant source context:")
        prompt_parts.append(source_summary)

    prompt_text = "\n".join(prompt_parts)

    # Remove trailing newline
    prompt_text = prompt_text.rstrip("\n")
    completion = generate_text(
        model,
        tokenizer,
        prompt_text,
        max_new_tokens=24,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
    )
    if "</title>" in completion:
        title_match = re.search(r"<\s*title\s*>([\s\S]*?)</\s*title\s*>", completion, flags=re.IGNORECASE)
        if title_match:
            return random_letter + _sanitize_generated_title(title_match.group(1), fallback_topic=topic or "General")
        else:
            # Get everything before </title>
            pre_title = completion.split("</title>", 1)[0]
            return random_letter + _sanitize_generated_title(pre_title, fallback_topic=topic or "General")

    return random_letter + _sanitize_generated_title(completion, fallback_topic=topic or "General")


def _strip_markdown_fence(text: str) -> str:
    """Remove optional markdown code fences from model output."""
    match = re.search(r"```(?:xml)?\\s*([\\s\\S]*?)\\s*```", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _sanitize_segments_markers_in_completion(text: str) -> str:
    """Completion must not emit <segments>; allow at most one </segments>."""
    sanitized = re.sub(r"<\s*s\s*e\s*g\s*m\s*e\s*n\s*t\s*s\s*>", "", text, flags=re.IGNORECASE)

    close_pattern = re.compile(r"<\s*/\s*s\s*e\s*g\s*m\s*e\s*n\s*t\s*s\s*>", flags=re.IGNORECASE)
    close_matches = list(close_pattern.finditer(sanitized))
    if len(close_matches) <= 1:
        return sanitized

    parts: list[str] = []
    cursor = 0
    last = close_matches[-1]
    for match in close_matches:
        parts.append(sanitized[cursor:match.start()])
        if match.start() == last.start() and match.end() == last.end():
            parts.append("</segments>")
        cursor = match.end()
    parts.append(sanitized[cursor:])
    return "".join(parts)


def _looks_repetition_loop(text: str) -> bool:
    """Detect common collapse loops seen in malformed generations."""
    if not text:
        return False

    if "<0x0A>" in text:
        return True

    close_example_count = len(re.findall(r"</example>", text, flags=re.IGNORECASE))
    if close_example_count >= 3:
        return True

    repeated_stub_count = len(
        re.findall(r"<segments>\s*1\s*</segments>", text, flags=re.IGNORECASE)
    )
    if repeated_stub_count >= 2:
        return True

    # Spaced opening tags like '< example ...>' are a strong sign of token collapse.
    if re.search(r"<\s+example\b", text, flags=re.IGNORECASE):
        return True

    return False


def _fragmentation_score(text: str) -> int:
    """Higher score means the text looks more token-fragmented."""
    score = 0
    score += text.count("▁") * 4
    score += len(re.findall(r"<\s*/?\s+[A-Za-z_]", text)) * 8
    score += len(re.findall(r"\b[a-zA-Z]{1,2}\s+[a-zA-Z]{1,3}\s+[a-zA-Z]{1,3}\b", text))
    return score


def _normalize_generated_text(text: str) -> str:
    """Normalize common tokenized artifacts in generated XML-ish text."""
    text = text.replace("▁", " ")
    text = text.replace(r"\_", "_")

    # Strip repeated leading chevrons that commonly appear before tags/timing values.
    text = re.sub(r"(?m)^\s*>+(?=\s*<)", "", text)
    text = re.sub(r"(?m)^\s*>+(?=\s*[-(\d])", "", text)
    text = re.sub(r"(?m)^\s*[A-Za-z]>\s*(?=<)", "", text)
    text = re.sub(r"(?<=>)\s*>+(?=\s*<)", "", text)
    text = re.sub(r"(?<=>)\s*>+(?=\s*[-(\d])", "", text)

    def _fix_tag(match: re.Match[str]) -> str:
        slash = match.group(1)
        name = re.sub(r"\s+", "", match.group(2))
        return f"<{slash}{name}>"

    text = re.sub(r"<\s*(/?)\s*([A-Za-z_][A-Za-z0-9_\s]*)\s*>", _fix_tag, text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _normalize_known_xmlish_tags(text: str) -> str:
    """Fix common near-miss tags emitted by the model to keep parsing robust."""
    if not text:
        return text

    fixed = text

    # Transcript tag misspellings seen in generations.
    transcript_tag_typos = (
        "transcription",
        "trascription",
        "transscript",
        "trascript",
        "trancript",
        "transciprt",
        "transcrpt",
        "transccript",
        "transit",
    )
    for typo in transcript_tag_typos:
        fixed = re.sub(
            rf"<\s*/?\s*{typo}\s*>",
            lambda m: "</transcript>" if "/" in m.group(0) else "<transcript>",
            fixed,
            flags=re.IGNORECASE,
        )

    # Some outputs emit <transition>/<scan> where a scene payload is intended.
    fixed = re.sub(r"<\s*/?\s*transition\s*>", lambda m: "</scene>" if "/" in m.group(0) else "<scene>", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"<\s*/?\s*scan\s*>", lambda m: "</scene>" if "/" in m.group(0) else "<scene>", fixed, flags=re.IGNORECASE)

    # Some outputs emit <transform> for transcript content.
    fixed = re.sub(r"<\s*/?\s*transform\s*>", lambda m: "</transcript>" if "/" in m.group(0) else "<transcript>", fixed, flags=re.IGNORECASE)

    # Normalize common plural typo for segment container.
    fixed = re.sub(r"<\s*/?\s*seguments\s*>", lambda m: "</segments>" if "/" in m.group(0) else "<segments>", fixed, flags=re.IGNORECASE)

    # Occasional tail artifact: </transcript>s>
    fixed = re.sub(r"(</\s*transcript\s*>)\s*s\s*>", r"\1", fixed, flags=re.IGNORECASE)

    # Some generations open <transcript> and immediately start a <scene> block.
    # Drop the orphan transcript opener to avoid illegal nesting.
    fixed = re.sub(r"<\s*transcript\s*>\s*(?=<\s*scene\s*>)", "", fixed, flags=re.IGNORECASE)

    # Repair malformed scene closers emitted as </transcript>.
    fixed = re.sub(r"(<\s*scene\s*>[\s\S]*?)</\s*transcript\s*>", r"\1</scene>", fixed, flags=re.IGNORECASE)

    # Repair malformed transcript closers emitted as </scene>.
    fixed = re.sub(r"(<\s*transcript\s*>[\s\S]*?)</\s*scene\s*>", r"\1</transcript>", fixed, flags=re.IGNORECASE)

    # If a scene block starts with a speaker marker, it is almost certainly transcript.
    fixed = re.sub(
        r"<\s*scene\s*>\s*(\[\s*SPEAK[A-Z_\s\-0-9]*\][\s\S]*?)</\s*scene\s*>",
        lambda m: f"<transcript>{m.group(1).strip()}</transcript>",
        fixed,
        flags=re.IGNORECASE,
    )

    # Remove repeated leading chevrons before tags and timing content.
    fixed = re.sub(r"(?m)^\s*>+(?=\s*<)", "", fixed)
    fixed = re.sub(r"(?m)^\s*>+(?=\s*[-(\d])", "", fixed)
    fixed = re.sub(r"(?m)^\s*[A-Za-z]>\s*(?=<)", "", fixed)
    fixed = re.sub(r"(?<=>)\s*>+(?=\s*<)", "", fixed)
    fixed = re.sub(r"(?<=>)\s*>+(?=\s*[-(\d])", "", fixed)
    fixed = re.sub(r"<\s*(start|end|duration)\s*>\s*>+", lambda m: f"<{m.group(1).lower()}>", fixed, flags=re.IGNORECASE)

    # Remove stray markdown/backtick artifacts that can break XML parsing.
    fixed = fixed.replace("`", "")

    # Drop XML comments injected by the model.
    fixed = re.sub(r"<!--.*?-->", "", fixed, flags=re.DOTALL)

    # Remove stray apostrophes and angle-brackets immediately after close tags.
    fixed = re.sub(r"(</\s*[A-Za-z_][A-Za-z0-9_]*\s*>)\s*'+", r"\1", fixed)
    fixed = re.sub(r"(</\s*[A-Za-z_][A-Za-z0-9_]*\s*>)\s*>+", r"\1", fixed)

    # Repair malformed timing tags like: <start>10.5 <end>20.0</end>
    # -> <start>10.5</start> <end>20.0</end>
    fixed = re.sub(
        r"<\s*start\s*>\s*([^<\r\n]+?)\s*<\s*end\s*>",
        r"<start>\1</start> <end>",
        fixed,
        flags=re.IGNORECASE,
    )

    # If a new segment body starts immediately with timing tags after a closed
    # segment, synthesize the missing opening <segment>.
    fixed = re.sub(
        r"</\s*segment\s*>\s*(?=<\s*start\s*>)",
        "</segment>\n<segment>",
        fixed,
        flags=re.IGNORECASE,
    )

    # Normalize adjacent segment boundaries to a stable line break.
    fixed = re.sub(
        r"</\s*segment\s*>\s*<\s*segment\b",
        "</segment>\n<segment>",
        fixed,
        flags=re.IGNORECASE,
    )

    # Repair malformed section closers like </section>, </sec>tion>, </sec>l> -> </segment>
    fixed = re.sub(r"</\s*section\s*>", "</segment>", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"</\s*sec\s*>\s*[A-Za-z_]*\s*>", "</segment>", fixed, flags=re.IGNORECASE)

    # Repair malformed audio opening tags such as: < Audio key words telephone, ringtone</audio>
    fixed = re.sub(
        r"<\s*audio\b\s*([^<>]*)</\s*audio\s*>",
        lambda m: f"<audio>{m.group(1).strip()}</audio>",
        fixed,
        flags=re.IGNORECASE,
    )

    # Convert mistaken <scene>[SPEAKER_X]...</transcript> blocks into transcript tags.
    fixed = re.sub(
        r"<\s*scene\s*>\s*(\[[^\]\r\n]+\][\s\S]*?)</\s*transcript\s*>",
        lambda m: f"<transcript>{m.group(1).strip()}</transcript>",
        fixed,
        flags=re.IGNORECASE,
    )

    # Canonicalize tag casing for known schema tags (XML is case-sensitive).
    canonical_tags = (
        "example",
        "input",
        "duration",
        "creator",
        "topic",
        "title",
        "output",
        "segments",
        "segment",
        "start",
        "end",
        "scene",
        "transcript",
        "audio",
    )
    for tag in canonical_tags:
        fixed = re.sub(
            rf"<\s*(/?)\s*{tag}\s*>",
            lambda m: f"</{tag}>" if m.group(1) else f"<{tag}>",
            fixed,
            flags=re.IGNORECASE,
        )

    # Remove duplicated immediate closing tags like </start></start>.
    dup_close_tags = ("start", "end", "duration", "scene", "transcript", "audio", "segment")
    for tag in dup_close_tags:
        fixed = re.sub(
            rf"</\s*{tag}\s*>\s*</\s*{tag}\s*>",
            rf"</{tag}>",
            fixed,
            flags=re.IGNORECASE,
        )

    # Collapse malformed transcript tails like:
    # <transcript>...</transcript>.At 1s ... </transcript>
    fixed = re.sub(
        r"</\s*transcript\s*>\s*[^<]{1,500}</\s*transcript\s*>",
        "</transcript>",
        fixed,
        flags=re.IGNORECASE,
    )

    return fixed


def _normalize_speaker_tags(text: str) -> str:
    """Normalize speaker markers like [SPEAKHER_1] -> [SPEAKER_1]."""
    if not text:
        return text

    def _fix(match: re.Match[str]) -> str:
        raw_name = match.group(1).upper()
        idx = match.group(2)
        compact = re.sub(r"[^A-Z]", "", raw_name)

        # Handle broad speaker-like variants.
        # Examples: SPEAKER, SPEAKERS, SPEAKHER, SPEAKAR, SPEAKRO,
        # SPEAKERY, SPEAKER-1, SPEAKER 1, SPEAKER1, SPEaker, etc.
        speaker_ratio = difflib.SequenceMatcher(a=compact, b="SPEAKER").ratio()
        if (
            compact.startswith("SPEAK")
            or compact.startswith("SPAKER")
            or compact.startswith("SPEKER")
            or compact.startswith("SPEAKER")
            or (compact.startswith("SP") and "EAK" in compact)
            or speaker_ratio >= 0.6
        ):
            return f"[SPEAKER_{idx}]"
        return match.group(0)

    # Normalize escaped underscores in speaker markers.
    text = text.replace(r"\_", "_")

    # Handle wildcard/missing index variants: [SPEAKER_*], [SPEAKER_?], [SPEAKER]
    text = re.sub(r"\[\s*SPEAK(?:ER|HER|ERS|ARY|ERY)?[_\s]*[*?]\s*\]", "[SPEAKER_1]", text, flags=re.IGNORECASE)

    # Handle alpha speaker variants like [SPEAKER B], [SPEAKER C].
    text = re.sub(r"\[\s*SPEAK(?:ER|HER|ERS|ARY|ERY)?[ _-]+[A-Z]\s*\]", "[SPEAKER_1]", text, flags=re.IGNORECASE)

    # Speaker zero is typically an invalid fallback marker.
    text = re.sub(r"\[\s*SPEAKER\s*_\s*0\s*\]", "[SPEAKER_1]", text, flags=re.IGNORECASE)

    return re.sub(r"\[\s*([A-Za-z][A-Za-z _-]*)\s*_?\s*(\d+)\s*\]", _fix, text)


def _extract_segment_stream(candidate: str) -> str:
    """Extract normalized segment stream for continuation retries."""
    if not candidate:
        return ""
    m = re.search(
        r"<\s*segments\s*>([\s\S]*?)</\s*segments\s*>",
        candidate,
        flags=re.IGNORECASE,
    )
    if not m:
        return candidate.strip()
    return m.group(1).strip()


def _drop_disallowed_xmlish_tags(text: str) -> str:
    """Normalize close-enough tag typos to canonical names; strip truly unknown tags and their content."""
    if not text:
        return text

    # Discover all tag names present in the text
    raw_names = set(m.lower() for m in re.findall(r"<\s*/?([A-Za-z_][A-Za-z0-9_]*)\b", text))

    normalize_map: dict[str, str] = {}
    truly_unknown: set[str] = set()
    for name in raw_names:
        if name in _ALLOWED_XMLISH_TAGS:
            continue
        matches = difflib.get_close_matches(name, _ALLOWED_XMLISH_TAGS, n=1, cutoff=0.75)
        if matches:
            normalize_map[name] = matches[0]
        else:
            truly_unknown.add(name)

    # Remove truly unknown tags together with their enclosed content
    for name in truly_unknown:
        text = re.sub(
            rf"<{re.escape(name)}\b[^>]*>.*?</{re.escape(name)}\s*>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove any remaining orphan open / close tags for the same name
        text = re.sub(rf"</?{re.escape(name)}\b[^>]*>", "", text, flags=re.IGNORECASE)

    # Canonicalize allowed tags and rename close-enough typos
    def _fix_tag(match: re.Match[str]) -> str:
        slash = match.group(1)
        name = match.group(2).lower()
        if name in _ALLOWED_XMLISH_TAGS:
            return f"</{name}>" if slash else f"<{name}>"
        if name in normalize_map:
            canonical = normalize_map[name]
            return f"</{canonical}>" if slash else f"<{canonical}>"
        return ""

    return re.sub(r"<\s*(/?)\s*([A-Za-z_][A-Za-z0-9_]*)\b[^>]*>", _fix_tag, text)


def _format_short_number(value: float) -> str:
    """Format numeric timing values to <=5 characters when possible."""
    for decimals in (3, 2, 1, 0):
        s = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        if len(s) <= 5:
            return s
    iv = str(int(round(value)))
    if len(iv) <= 5:
        return iv
    return iv[:5]


def _extract_first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _has_empty_scene_tags(text: str) -> bool:
    """Return True if XML-ish text contains <scene> with no content."""
    if not text:
        return False
    return bool(
        re.search(
            r"<\s*scene\s*>\s*</\s*scene\s*>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _seed_first_missing_or_empty_scene_segment(segment_stream: str) -> str | None:
    """Return a retry seed stream at first segment missing/empty scene, else None.

    The returned text truncates everything from the first bad segment onward and
    reintroduces that segment's timing scaffold followed by '<scene>a ' so the
    next continuation pass completes the scene content instead of propagating
    empty scene segments.
    """
    if not segment_stream:
        return None

    seg_re = re.compile(r"<\s*segment\s*>([\s\S]*?)</\s*segment\s*>", flags=re.IGNORECASE)
    for m in seg_re.finditer(segment_stream):
        body = m.group(1)
        scene_m = re.search(r"<\s*scene\s*>([\s\S]*?)</\s*scene\s*>", body, flags=re.IGNORECASE)
        has_scene = scene_m is not None
        scene_text = (scene_m.group(1).strip() if scene_m is not None else "")
        if has_scene and scene_text:
            continue

        def _tag_value(tag: str) -> str:
            mm = re.search(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", body, flags=re.IGNORECASE | re.DOTALL)
            if not mm:
                return ""
            return (mm.group(1) or "").strip()

        start_s = _tag_value("start")
        end_s = _tag_value("end")
        dur_s = _tag_value("duration")

        # Keep all valid content before the first bad segment.
        prefix = segment_stream[: m.start()].rstrip()
        if prefix:
            prefix += "\n"

        seeded = (
            "<segment>\n"
            f"<start>{start_s}</start>\n"
            f"<end>{end_s}</end>\n"
            f"<duration>{dur_s}</duration>\n"
            "<scene>a "
        )
        
        return prefix + seeded

    return None


def _seed_first_bad_transcript_segment(segment_stream: str) -> str | None:
    """Return retry seed at first malformed transcript block, else None.

    Rule: a transcript payload must stay inline after <transcript> and should
    not begin on a new line. When violated, truncate from that segment and
    reseed with '<transcript>[SPEAKER_' so continuation regenerates transcript
    content in the expected format.
    """
    if not segment_stream:
        return None

    seg_re = re.compile(r"<\s*segment\s*>([\s\S]*?)</\s*segment\s*>", flags=re.IGNORECASE)
    for m in seg_re.finditer(segment_stream):
        body = m.group(1)

        open_count = len(re.findall(r"<\s*transcript\s*>", body, flags=re.IGNORECASE))
        close_count = len(re.findall(r"</\s*transcript\s*>", body, flags=re.IGNORECASE))
        starts_with_newline = bool(
            re.search(r"<\s*transcript\s*>\s*(?:\r?\n)", body, flags=re.IGNORECASE)
        )

        is_bad = starts_with_newline or open_count != close_count
        if not is_bad:
            continue

        def _tag_value(tag: str) -> str:
            mm = re.search(
                rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not mm:
                return ""
            return (mm.group(1) or "").strip()

        start_s = _tag_value("start")
        end_s = _tag_value("end")
        dur_s = _tag_value("duration")

        scene_m = re.search(
            r"<\s*scene\s*>([\s\S]*?)</\s*scene\s*>",
            body,
            flags=re.IGNORECASE,
        )
        scene_text = (scene_m.group(1).strip() if scene_m is not None else "")

        prefix = segment_stream[: m.start()].rstrip()
        if prefix:
            prefix += "\n"

        scene_block = f"<scene>{scene_text}</scene>\n" if scene_text else "<scene>a </scene>\n"
        seeded = (
            "<segment>\n"
            f"<start>{start_s}</start>\n"
            f"<end>{end_s}</end>\n"
            f"<duration>{dur_s}</duration>\n"
            f"{scene_block}"
            "<transcript>[SPEAKER_"
        )
        return prefix + seeded

    return None


def _seed_first_missing_transcript_and_audio_segment(segment_stream: str) -> str | None:
    """Return retry seed at first segment missing both transcript and audio.

    Rule: each segment must include either transcript content or audio content.
    When both are absent, truncate from that segment and reseed with an audio
    tag scaffold so continuation fills non-empty audio metadata.
    """
    if not segment_stream:
        return None

    seg_re = re.compile(r"<\s*segment\s*>([\s\S]*?)</\s*segment\s*>", flags=re.IGNORECASE)
    for m in seg_re.finditer(segment_stream):
        body = m.group(1)

        transcript_values = [
            (mm.group(1) or "").strip()
            for mm in re.finditer(
                r"<\s*transcript\s*>([\s\S]*?)</\s*transcript\s*>",
                body,
                flags=re.IGNORECASE,
            )
        ]
        audio_values = [
            (mm.group(1) or "").strip()
            for mm in re.finditer(
                r"<\s*audio\s*>([\s\S]*?)</\s*audio\s*>",
                body,
                flags=re.IGNORECASE,
            )
        ]

        transcript_text = " ".join(v for v in transcript_values if v).strip()
        audio_text = " ".join(v for v in audio_values if v).strip()
        if transcript_text or audio_text:
            continue

        def _tag_value(tag: str) -> str:
            mm = re.search(
                rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not mm:
                return ""
            return (mm.group(1) or "").strip()

        start_s = _tag_value("start")
        end_s = _tag_value("end")
        dur_s = _tag_value("duration")

        scene_m = re.search(
            r"<\s*scene\s*>([\s\S]*?)</\s*scene\s*>",
            body,
            flags=re.IGNORECASE,
        )
        scene_text = (scene_m.group(1).strip() if scene_m is not None else "")

        prefix = segment_stream[: m.start()].rstrip()
        if prefix:
            prefix += "\n"

        scene_block = f"<scene>{scene_text}</scene>\n" if scene_text else "<scene>a </scene>\n"
        seeded = (
            "<segment>\n"
            f"<start>{start_s}</start>\n"
            f"<end>{end_s}</end>\n"
            f"<duration>{dur_s}</duration>\n"
            f"{scene_block}"
            "<audio>a "
        )
        return prefix + seeded

    return None


def _seed_first_missing_or_empty_transcript_segment(segment_stream: str) -> str | None:
    """Return retry seed at first segment missing/empty transcript, else None.

    Used by --force-speech. Once a scene is present, continuation is reseeded
    immediately after that scene with a transcript tag on a new line, starting
    with a random letter so the model is less likely to leave it empty.
    """
    if not segment_stream:
        return None

    seg_re = re.compile(r"<\s*segment\s*>([\s\S]*?)</\s*segment\s*>", flags=re.IGNORECASE)
    for m in seg_re.finditer(segment_stream):
        body = m.group(1)
        scene_m = re.search(r"<\s*scene\s*>([\s\S]*?)</\s*scene\s*>", body, flags=re.IGNORECASE)
        scene_text = (scene_m.group(1).strip() if scene_m is not None else "")
        if not scene_text:
            continue

        transcript_values = [
            (mm.group(1) or "").strip()
            for mm in re.finditer(
                r"<\s*transcript\s*>([\s\S]*?)</\s*transcript\s*>",
                body,
                flags=re.IGNORECASE,
            )
        ]
        has_transcript_text = any(v for v in transcript_values)
        if has_transcript_text:
            continue

        def _tag_value(tag: str) -> str:
            mm = re.search(
                rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not mm:
                return ""
            return (mm.group(1) or "").strip()

        start_s = _tag_value("start")
        end_s = _tag_value("end")
        dur_s = _tag_value("duration")

        prefix = segment_stream[: m.start()].rstrip()
        if prefix:
            prefix += "\n"

        # transcript_seed = random.choice(["T", "L", "S", "W"])
        seeded = (
            "<segment>\n"
            f"<start>{start_s}</start>\n"
            f"<end>{end_s}</end>\n"
            f"<duration>{dur_s}</duration>\n"
            f"<scene>{scene_text}</scene>\n"
            f"<transcript>[SPEAKER_"
        )
        return prefix + seeded

    return None


def _seed_last_empty_scene_for_retry(text: str) -> str:
    """Truncate at the last empty <scene></scene> and seed continuation with '<scene>a'."""
    if not text:
        return text
    pattern = re.compile(r"<\s*scene\s*>\s*</\s*scene\s*>", flags=re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    last = matches[-1]
    return text[: last.start()] + "<scene>a"


def _enforce_xmlish_rules(text: str) -> str:
    """Apply strict structural rules for allowed XML-ish schema tags."""
    if not text:
        return text

    min_seg_dur = 0.2
    max_seg_dur = 15.0

    # Keep only allowed tags (plus ui_description).
    text = _drop_disallowed_xmlish_tags(text)

    # Ensure `</input>` is followed by `<output>` on next line.
    text = re.sub(r"</\s*input\s*>\s*<\s*output\s*>", "</input>\n<output>", text, flags=re.IGNORECASE)

    # After first </input>, remove repeated metadata/input wrappers.
    input_close = re.search(r"</\s*input\s*>", text, flags=re.IGNORECASE)
    if input_close:
        head = text[: input_close.end()]
        tail = text[input_close.end() :]
        tail = re.sub(r"<\s*input\s*>.*?</\s*input\s*>", "", tail, flags=re.IGNORECASE | re.DOTALL)
        tail = re.sub(r"<\s*(topic|title|creator)\s*>.*?</\s*\1\s*>", "", tail, flags=re.IGNORECASE | re.DOTALL)
        text = head + tail

    def _target_total_duration(src: str) -> float | None:
        m = re.search(r"<\s*input\s*>.*?<\s*duration\s*>(.*?)</\s*duration\s*>", src, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        v = _extract_first_number(m.group(1))
        if v is None or v <= 0:
            return None
        return float(v)

    target_total = _target_total_duration(text)

    # Normalize each segment so timings are sequential and bounded.
    seg_infos: list[dict[str, Any]] = []
    seg_re = re.compile(r"<\s*segment\s*>(.*?)</\s*segment\s*>", flags=re.IGNORECASE | re.DOTALL)
    for m in seg_re.finditer(text):
        body = m.group(1)

        # Repair orphan text nodes where closing tags exist but opening tags were dropped.
        if re.search(r"</\s*scene\s*>", body, flags=re.IGNORECASE) and not re.search(r"<\s*scene\s*>", body, flags=re.IGNORECASE):
            body = re.sub(
                r"^\s*([\s\S]*?)</\s*scene\s*>",
                lambda mm: f"<scene>{mm.group(1).strip()}</scene>",
                body,
                count=1,
                flags=re.IGNORECASE,
            )
        if re.search(r"</\s*transcript\s*>", body, flags=re.IGNORECASE) and not re.search(r"<\s*transcript\s*>", body, flags=re.IGNORECASE):
            body = re.sub(
                r"^\s*([\s\S]*?)</\s*transcript\s*>",
                lambda mm: f"<transcript>{mm.group(1).strip()}</transcript>",
                body,
                count=1,
                flags=re.IGNORECASE,
            )

        def _get(tag: str) -> float | None:
            mm = re.search(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", body, flags=re.IGNORECASE | re.DOTALL)
            if not mm:
                return None
            return _extract_first_number(mm.group(1))

        start_v = _get("start")
        end_v = _get("end")
        dur_v = _get("duration")

        timing_issue = False
        if start_v is None or end_v is None or dur_v is None:
            timing_issue = True
        elif end_v <= start_v:
            timing_issue = True
        else:
            implied = end_v - start_v
            if abs(implied - dur_v) > 0.25:
                timing_issue = True

        raw_dur: float | None = None
        if dur_v is not None:
            raw_dur = dur_v
        elif start_v is not None and end_v is not None:
            raw_dur = end_v - start_v
        if raw_dur is None or raw_dur <= 0:
            raw_dur = 1.0
        raw_dur = max(min_seg_dur, min(max_seg_dur, float(raw_dur)))

        # Remove any existing timing tags before re-inserting canonical ones.
        body_wo_timing = re.sub(
            r"<\s*(start|end|duration)\s*>.*?</\s*\1\s*>",
            "",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        # Remove stray bare ">" tokens left by the model (e.g. as list/bullet artifacts).
        body_wo_timing = re.sub(r"(?m)^\s*>\s*$", "", body_wo_timing).strip()

        seg_infos.append({"duration": raw_dur, "body": body_wo_timing, "timing_issue": timing_issue})

    # If there are no segments yet but we have a target duration, create one starter segment.
    if not seg_infos and target_total is not None and target_total > 0:
        seg_infos.append({"duration": max(min_seg_dur, min(max_seg_dur, target_total)), "body": ""})

    # Trim overflow so we do not exceed target duration while respecting min segment length.
    if seg_infos and target_total is not None:
        total = sum(float(s["duration"]) for s in seg_infos)
        overflow = max(0.0, total - target_total)
        if overflow > 1e-9:
            for s in reversed(seg_infos):
                reducible = max(0.0, float(s["duration"]) - min_seg_dur)
                take = min(reducible, overflow)
                s["duration"] = float(s["duration"]) - take
                overflow -= take
                if overflow <= 1e-9:
                    break
            # If still overflowing due to min-duration constraint, drop trailing tiny segments where possible.
            while overflow > 1e-9 and len(seg_infos) > 1:
                last = seg_infos[-1]
                if float(last["duration"]) <= min_seg_dur + 1e-9:
                    seg_infos.pop()
                    overflow = max(0.0, sum(float(s["duration"]) for s in seg_infos) - target_total)
                else:
                    break

    # Extend with additional segments (instead of over-extending a single segment)
    # so the last end time reaches input duration.
    if target_total is not None and target_total > 0:
        total = sum(float(s["duration"]) for s in seg_infos)
        remain = target_total - total
        while remain > 1e-9:
            chunk = min(max_seg_dur, remain)
            if chunk < min_seg_dur:
                if seg_infos:
                    spare = max_seg_dur - float(seg_infos[-1]["duration"])
                    take = min(spare, remain)
                    if take > 1e-9:
                        seg_infos[-1]["duration"] = float(seg_infos[-1]["duration"]) + take
                        remain -= take
                        break
                chunk = min_seg_dur
            seg_infos.append({"duration": chunk, "body": ""})
            remain -= chunk

    # Force strict sequential timings (start_i == end_{i-1}), starting at 0.
    rebuilt: list[str] = []
    first_seg = re.search(r"<\s*segment\b", text, flags=re.IGNORECASE)
    if first_seg is not None:
        rebuilt.append(text[: first_seg.start()])
    t = 0.0
    for s in seg_infos:
        dur_v = max(min_seg_dur, min(max_seg_dur, float(s["duration"])))
        start_v = t
        end_v = start_v + dur_v
        t = end_v
        body_wo_timing = str(s["body"]).strip()
        if bool(s.get("timing_issue", False)):
            seg_text = (
                "<segment>\n"
                f"<start>{_format_short_number(start_v)}</start>\n"
                f"<end>{_format_short_number(end_v)}</end>\n"
                f"<duration>{_format_short_number(dur_v)}</duration>\n"
                "</segment>\n"
            )
        elif body_wo_timing:
            seg_text = (
                "<segment>\n"
                f"<start>{_format_short_number(start_v)}</start>\n"
                f"<end>{_format_short_number(end_v)}</end>\n"
                f"<duration>{_format_short_number(dur_v)}</duration>\n"
                f"{body_wo_timing}\n"
                "</segment>\n"
            )
        else:
            seg_text = (
                "<segment>\n"
                f"<start>{_format_short_number(start_v)}</start>\n"
                f"<end>{_format_short_number(end_v)}</end>\n"
                f"<duration>{_format_short_number(dur_v)}</duration>\n"
                "</segment>\n"
            )
        rebuilt.append(seg_text)

    # Keep tail after the last original segment wrapper (e.g., </segments>...</example>).
    last_seg = None
    for m in seg_re.finditer(text):
        last_seg = m
    if last_seg is not None:
        rebuilt.append(text[last_seg.end() :])
    text = "".join(rebuilt)

    return text


def _is_common_word(word: str) -> bool:
    w = word.lower()
    if not w:
        return False
    if w in {
        "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "from",
        "is", "are", "was", "were", "be", "being", "has", "have", "had", "it", "its", "this", "that",
        "video", "scene", "camera", "background", "close", "up", "shot", "music", "mood", "happy",
        "elf", "elves", "christmas", "deluxe", "face", "white", "red", "green", "wall", "smile",
        "stands", "standing", "looks", "looking", "holds", "holding", "joyful", "playful", "cozy",
        "indoors", "space", "lighting", "static", "expression", "features", "transcript", "audio",
        "today", "topic", "market", "local", "shop", "sale", "breaking", "news",
    }:
        return True
    # Conservative fallback: keep short common function words only.
    if len(w) <= 2:
        return w in {"a", "i", "an", "to", "of", "in", "on", "at", "it", "we", "he", "is", "or", "be", "by", "up", "as"}
    # Treat alphabetic words of moderate length as plausible when no lexicon is available.
    return bool(re.fullmatch(r"[a-z']{3,20}", w))


def _repair_fragmented_words(text: str) -> str:
    """Repair word-boundary fragmentation like 'mus ic' -> 'music', 'int he' -> 'in the'."""
    if not text or len(text) < 4:
        return text

    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)

    changed = True
    passes = 0
    while changed and passes < 4:
        changed = False
        passes += 1
        i = 0
        out: list[str] = []
        while i < len(tokens):
            if (
                i + 2 < len(tokens)
                and re.fullmatch(r"[A-Za-z']+", tokens[i])
                and tokens[i + 1] == " "
                and re.fullmatch(r"[A-Za-z']+", tokens[i + 2])
            ):
                a = tokens[i]
                b = tokens[i + 2]
                merged = a + b

                # Case 1: direct merge when combined form is common.
                if _is_common_word(merged) and not (_is_common_word(a) and _is_common_word(b)):
                    out.append(merged)
                    i += 3
                    changed = True
                    continue

                # Case 2: split first token and merge tail with second token.
                # Example: 'int he' -> 'in the', 'acl ose' -> 'a close'
                best_split: tuple[str, str] | None = None
                best_score = -1.0
                for k in range(1, len(a)):
                    left = a[:k]
                    right = a[k:] + b
                    if _is_common_word(left) and _is_common_word(right):
                        # Prefer balanced splits and longer reconstructed words.
                        score = min(len(left), len(right)) + 0.1 * len(right)
                        if score > best_score:
                            best_score = score
                            best_split = (left, right)
                if best_split is not None:
                    out.append(best_split[0])
                    out.append(" ")
                    out.append(best_split[1])
                    i += 3
                    changed = True
                    continue

            out.append(tokens[i])
            i += 1
        tokens = out

    repaired = "".join(tokens)
    repaired = re.sub(r"\s+([,.!?;:])", r"\1", repaired)
    repaired = re.sub(r"\s{2,}", " ", repaired)
    return repaired.strip()


def _load_nlp_spacing_scorer(model_id: str = "distilgpt2") -> tuple[Any, Any] | tuple[None, None]:
    """Lazy-load a lightweight pretrained LM used for spacing candidate scoring."""
    global _NLP_SCORER_TOKENIZER, _NLP_SCORER_MODEL
    if _NLP_SCORER_TOKENIZER is not None and _NLP_SCORER_MODEL is not None:
        return _NLP_SCORER_TOKENIZER, _NLP_SCORER_MODEL

    _ensure_ml_imports()

    try:
        _NLP_SCORER_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _NLP_SCORER_MODEL = AutoModelForCausalLM.from_pretrained(model_id)
        _NLP_SCORER_MODEL.eval()
        return _NLP_SCORER_TOKENIZER, _NLP_SCORER_MODEL
    except Exception as exc:
        print(f"[INFO] NLP spacing scorer unavailable ({model_id}): {exc}")
        _NLP_SCORER_TOKENIZER = None
        _NLP_SCORER_MODEL = None
        return None, None


def _lm_avg_logprob(text: str, lm_tokenizer: Any, lm_model: Any) -> float:
    """Return average token log-prob proxy (negative LM loss). Higher is better."""
    if not text.strip():
        return float("-inf")

    encoded = lm_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    )
    with torch.no_grad():
        out = lm_model(**encoded, labels=encoded["input_ids"])
    return -float(out.loss.item())


def _window_text(tokens: list[str], center: int, radius: int = 8) -> str:
    start = max(0, center - radius)
    end = min(len(tokens), center + radius + 1)
    return " ".join(tokens[start:end])


def _nlp_repair_spacing_text(text: str, scorer_model_id: str = "distilgpt2") -> str:
    """Repair fragmented spacing by searching merge/split edits scored by a pretrained LM."""
    cleaned = _repair_fragmented_words(text)
    if not cleaned:
        return cleaned

    lm_tokenizer, lm_model = _load_nlp_spacing_scorer(scorer_model_id)
    if lm_tokenizer is None or lm_model is None:
        return cleaned

    tokens = cleaned.split()
    if len(tokens) < 2:
        return cleaned

    alpha_re = re.compile(r"[A-Za-z']+")

    # Pass 1: merge suspicious adjacent fragments when LM likelihood improves.
    i = 0
    merge_edits = 0
    while i < len(tokens) - 1 and merge_edits < 40:
        a = tokens[i]
        b = tokens[i + 1]
        if not (alpha_re.fullmatch(a) and alpha_re.fullmatch(b)):
            i += 1
            continue
        if not (len(a) <= 3 or len(b) <= 3):
            i += 1
            continue

        base = _window_text(tokens, i)
        merged_tokens = tokens[:i] + [a + b] + tokens[i + 2 :]
        cand = _window_text(merged_tokens, i)
        base_score = _lm_avg_logprob(base, lm_tokenizer, lm_model)
        cand_score = _lm_avg_logprob(cand, lm_tokenizer, lm_model)

        if cand_score > base_score + 0.10:
            tokens = merged_tokens
            merge_edits += 1
            i = max(0, i - 1)
            continue
        i += 1

    # Pass 2: split suspicious long compact tokens when LM likelihood improves.
    i = 0
    split_edits = 0
    while i < len(tokens) and split_edits < 30:
        tok = tokens[i]
        if not alpha_re.fullmatch(tok) or len(tok) < 7:
            i += 1
            continue

        base = _window_text(tokens, i)
        base_score = _lm_avg_logprob(base, lm_tokenizer, lm_model)
        best_tokens = None
        best_score = base_score

        for k in range(2, len(tok) - 1):
            left = tok[:k]
            right = tok[k:]
            if len(left) < 2 or len(right) < 2:
                continue
            cand_tokens = tokens[:i] + [left, right] + tokens[i + 1 :]
            cand = _window_text(cand_tokens, i)
            cand_score = _lm_avg_logprob(cand, lm_tokenizer, lm_model)
            if cand_score > best_score + 0.12:
                best_score = cand_score
                best_tokens = cand_tokens

        if best_tokens is not None:
            tokens = best_tokens
            split_edits += 1
            i = max(0, i - 1)
            continue
        i += 1

    repaired = " ".join(tokens)
    repaired = re.sub(r"\s+([,.!?;:])", r"\1", repaired)
    repaired = re.sub(r"\s{2,}", " ", repaired)
    return repaired.strip()


def _repair_xmlish_text_nodes(xml_text: str) -> str:
    """Repair fragmented spacing only inside textual content nodes, not tags/attributes."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text

    for elem in root.iter():
        if elem.tag in {"scene", "transcript", "audio", "title", "topic", "creator"} and elem.text:
            elem.text = _repair_fragmented_words(elem.text)

    return ET.tostring(root, encoding="unicode")


def _split_description_chunks(text: str) -> list[str]:
    """Split a long generated description into punctuation-delimited chunks."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_quoted_phrases(text: str) -> list[str]:
    """Extract quoted phrase candidates using both single and double quotes."""
    if not text:
        return []
    out: list[str] = []
    for m in re.finditer(r'"([^"\n]{2,200})"|\'([^\'\n]{2,200})\'', text):
        phrase = (m.group(1) or m.group(2) or "").strip()
        if phrase:
            out.append(phrase)
    return out


def _normalize_for_match(text: str) -> str:
    """Normalize text for loose quote/transcript matching."""
    s = text.lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9']+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _classify_description_chunk(chunk: str) -> str:
    """Classify a description chunk as scene/ui/audio_effect/speech."""
    low = chunk.lower()
    has_quote = bool(_extract_quoted_phrases(chunk))

    scene_terms = (
        "close-up", "shadows", "lighting", "camera", "gesture", "eye", "expression", "hand",
        "background", "shot", "frame", "indoors", "outdoors", "posture", "face",
    )
    ui_terms = ("text", "letters", "bold", "reads", "on-screen", "overlay", "caption", "word")
    audio_terms = ("audio", "voice", "music", "sound effect", "sound effects", "sfx", "tone")
    strong_effect_terms = (
        "music", "sound effect", "sound effects", "sfx", "onomatopoeia",
        "ambient sound", "ambient noise",
        "noise", "rumble", "thump", "squish", "flush", "buzz", "beep", "ringtone",
        "laughter", "giggle", "whoosh", "shatter", "crash", "bang",
    )
    speech_terms = (
        "says", "said", "speaks", "speaking", "asks", "adds", "continues", "concludes",
        "yells", "shouts", "whispers", "tells", "replies",
    )

    has_scene = any(t in low for t in scene_terms)
    has_ui = any(t in low for t in ui_terms)
    has_audio = any(t in low for t in audio_terms)
    has_speech = any(t in low for t in speech_terms)
    is_effect = any(t in low for t in strong_effect_terms)
    sound_effect_phrasing = bool(
        re.search(
            r"\b(?:high-pitched|low-pitched|loud|soft|squeaky|shrill|piercing|comical|cartoonish|metallic|whirring|buzzing|crackling|booming|thunderous)\b[\s\S]{0,40}\bsound\b",
            low,
            flags=re.IGNORECASE,
        )
    )

    if has_quote and has_ui and not has_audio:
        return "ui"
    if is_effect or sound_effect_phrasing:
        return "audio_effect"
    if has_quote and (has_speech or ("voice" in low) or ("tone" in low)):
        return "speech"
    # Mentions of "audio/voice/tone" are often narration context rather than sound effects.
    if has_audio and not has_quote and not has_scene and not has_ui:
        return "speech"
    if has_ui:
        return "ui"
    if has_scene:
        return "scene"
    return "scene"


def _split_audio_chunks(text: str) -> list[str]:
    """Split audio_description text into smaller chunks for reclassification."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|\n+|\s*,\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


def _description_without_quotes(chunk: str) -> str:
    """Strip quoted speech from chunk and return the descriptive remainder."""
    chunk = chunk.replace(r'\"', '"').replace(r"\'", "'")
    text = re.sub(r'"[^"\n]{2,200}"|\'[^\'\n]{2,200}\'', "", chunk)
    text = re.sub(
        r"\b(he|she|they|speaker)\s+(says?|said|speaks?|speaking|asks?|adds?|continues?|concludes?|yells?|shouts?|whispers?)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    return text


def _spoken_text_only(text: str) -> str:
    """Return transcript text with speaker/annotation brackets removed."""
    if not text:
        return ""
    text = text.replace(r'\"', '"').replace(r"\'", "'")
    text = re.sub(r"\[\s*SPEAKER[_\s]*[0-9]+\s*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]\n]{1,200}\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_tone_label(desc: str) -> str:
    """Condense a verbose speech description to a short tone label.

    Examples::

        "speaks in a calm, conversational tone: His voice..." -> "calm, conversational tone"
        "narrates with a firm, demanding voice"               -> "firm, demanding voice"
        "tone: solemn and measured"                           -> "solemn and measured"
    """
    cleaned = desc.replace(r'\"', '"').replace(r"\'", "'")
    cleaned = cleaned.replace("\\", " ")
    cleaned = re.sub(r'^[\s\[\]\(\)\{\}"\'`.,:;!?-]+', '', cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    def _clean_phrase(phrase: str) -> str:
        phrase = phrase.strip(" ,;:-")
        phrase = re.sub(r"\b(full of|filled with|driven by|because of|throughout|until|while|when|that|which)\b[\s\S]*$", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\s+", " ", phrase).strip(" ,;:-")
        return phrase

    # "in a(n) <adjectives> tone/voice/manner/style/delivery"
    m = re.search(
        r"\bin\s+an?\s+([A-Za-z][A-Za-z,\s]{1,60}?)\s*\b(tone|voice|manner|style|delivery)\b",
        cleaned, re.IGNORECASE,
    )
    if m:
        phrase = _clean_phrase(m.group(1))
        return f"{phrase} {m.group(2).lower()}".strip() if phrase else ""
    # "in <adjectives> tone/voice" without an article.
    m = re.search(
        r"\bin\s+([A-Za-z][A-Za-z,\s-]{1,60}?)\s*\b(tone|voice|manner|style|delivery)\b",
        cleaned, re.IGNORECASE,
    )
    if m:
        phrase = _clean_phrase(m.group(1))
        return f"{phrase} {m.group(2).lower()}".strip() if phrase else ""
    # "with a(n) <adjectives> tone/voice"
    m = re.search(
        r"\bwith\s+an?\s+([A-Za-z][A-Za-z,\s]{1,60}?)\s*\b(tone|voice)\b",
        cleaned, re.IGNORECASE,
    )
    if m:
        phrase = _clean_phrase(m.group(1))
        return f"{phrase} {m.group(2).lower()}".strip() if phrase else ""
    # "tone: <label>"
    m = re.search(r"\btone\s*:\s*([A-Za-z][^.;!\n]{3,50})", cleaned, re.IGNORECASE)
    if m:
        return _clean_phrase(m.group(1))
    # Short bare labels like "cheerful, encouraging tone" are already usable,
    # but only accept them when the whole string is already descriptor-like.
    m = re.fullmatch(r"([A-Za-z][A-Za-z,\s-]{1,50}?)\s+(tone|voice|manner|style|delivery)", cleaned, re.IGNORECASE)
    if m:
        phrase = _clean_phrase(m.group(1))
        return f"{phrase} {m.group(2).lower()}".strip() if phrase else ""
    return ""


def _normalize_transcript_annotations(transcript: str) -> str:
    """Normalize leading transcript bracket tags and drop malformed junk."""
    if not transcript:
        return ""
    transcript = transcript.replace(r'\"', '"').replace(r"\'", "'").strip()
    match = re.match(r"^\s*((?:\[[^\]\n]{1,200}\]\s*)+)", transcript)
    if not match:
        return transcript

    speaker_tags: list[str] = []
    tone_tags: list[str] = []
    seen_tones: set[str] = set()
    for raw_tag in re.findall(r"\[([^\]\n]{1,200})\]", match.group(1)):
        raw_tag = raw_tag.strip()
        speaker_match = re.fullmatch(r"SPEAKER[_\s]*([0-9]+)", raw_tag, flags=re.IGNORECASE)
        if speaker_match:
            speaker_tags.append(f"[SPEAKER_{speaker_match.group(1)}]")
            continue
        label = _extract_tone_label(raw_tag)
        norm_label = _normalize_for_match(label)
        if label and norm_label and norm_label not in seen_tones:
            seen_tones.add(norm_label)
            tone_tags.append(f"[{label}]")

    body = transcript[match.end() :].strip()
    prefix = "".join(speaker_tags + tone_tags)
    if prefix and body:
        return f"{prefix} {body}".strip()
    if prefix:
        return prefix
    return body


def _reclassify_segment_payload(seg: dict[str, Any]) -> bool:
    """Move mixed scene payload into dedicated fields after generation.

    Returns True if any field changed.
    """
    scene_text = str(seg.get("scene_description", "") or "").strip()
    audio_text = str(seg.get("audio_description", "") or "").strip()

    chunks = _split_description_chunks(scene_text)
    audio_chunks_in = _split_audio_chunks(audio_text)
    if not chunks and not audio_chunks_in:
        return False

    visual_chunks: list[str] = []
    ui_chunks: list[str] = []
    audio_chunks: list[str] = []
    speech_items: list[tuple[str, str]] = []

    for chunk in chunks:
        kind = _classify_description_chunk(chunk)
        if kind == "ui":
            quotes = _extract_quoted_phrases(chunk)
            if quotes:
                ui_chunks.extend(quotes)
            else:
                ui_chunks.append(chunk)
        elif kind == "audio_effect":
            audio_chunks.append(chunk)
        elif kind == "speech":
            quotes = _extract_quoted_phrases(chunk)
            quote = quotes[0].strip() if quotes else ""
            desc = _description_without_quotes(chunk)
            if desc:
                speech_items.append((desc, quote))
        else:
            visual_chunks.append(chunk)

    # Reclassify pre-existing audio_description content as well; this fixes
    # prior over-extraction where visual narration was moved into audio.
    for chunk in audio_chunks_in:
        kind = _classify_description_chunk(chunk)
        if kind == "audio_effect":
            audio_chunks.append(chunk)
        elif kind == "ui":
            quotes = _extract_quoted_phrases(chunk)
            if quotes:
                ui_chunks.extend(quotes)
            else:
                ui_chunks.append(chunk)
        elif kind == "speech":
            quotes = _extract_quoted_phrases(chunk)
            quote = quotes[0].strip() if quotes else ""
            desc = _description_without_quotes(chunk)
            if desc:
                speech_items.append((desc, quote))
        else:
            visual_chunks.append(chunk)

    changed = False

    # Keep scene_description focused on visible elements.
    scene_joined = " ".join(visual_chunks).strip()
    if scene_joined != scene_text:
        seg["scene_description"] = scene_joined
        changed = True

    # Merge UI captions into ui_elements_html with dedupe against existing caption bodies.
    if ui_chunks:
        existing_ui_html = str(seg.get("ui_elements_html", "") or "")
        existing_norm = {
            _normalize_for_match(m.group(1))
            for m in re.finditer(r">([^<]{1,300})<", existing_ui_html)
            if m.group(1).strip()
        }
        new_lines: list[str] = []
        for txt in ui_chunks:
            norm = _normalize_for_match(txt)
            if not norm or norm in existing_norm:
                continue
            existing_norm.add(norm)
            new_lines.append(f"<div class=\"overlay caption\">{html.escape(txt, quote=False)}</div>")
        if new_lines:
            seg["ui_elements_html"] = (existing_ui_html + ("\n" if existing_ui_html else "") + "\n".join(new_lines)).strip()
            changed = True

    # Merge audio effects text into audio_description.
    # Overwrite audio_description with filtered effect-only chunks.
    filtered_audio = " ".join(audio_chunks).strip()
    if filtered_audio != audio_text:
        seg["audio_description"] = filtered_audio
        changed = True

    # Annotate transcript with speech descriptors when quote matches are found.
    transcript = _normalize_transcript_annotations(str(seg.get("transcript", "") or ""))

    if speech_items:
        transcript_norm = _normalize_for_match(transcript)
        added_descs: list[str] = []
        seen_descs: set[str] = set()
        missing_quotes: list[str] = []
        for desc, quote in speech_items:
            label = _extract_tone_label(desc)
            quote_norm = _normalize_for_match(quote)
            if label:
                label_norm = _normalize_for_match(label)
                if transcript and quote_norm and quote_norm in transcript_norm and label_norm not in seen_descs:
                    seen_descs.add(label_norm)
                    added_descs.append(label)
                elif transcript and not quote_norm and label_norm not in seen_descs:
                    seen_descs.add(label_norm)
                    added_descs.append(label)
                elif not transcript and label_norm not in seen_descs:
                    seen_descs.add(label_norm)
                    added_descs.append(label)
                elif transcript and quote_norm and quote_norm not in transcript_norm and label_norm not in seen_descs:
                    seen_descs.add(label_norm)
                    added_descs.append(label)

            if quote and (not quote_norm or quote_norm not in transcript_norm):
                missing_quotes.append(quote)

        if added_descs:
            desc_block = " ".join(f"[{d}]" for d in added_descs)

            # Insert description tags immediately after any leading speaker tokens.
            speaker_prefix_match = re.match(r"^\s*((?:\[[^\]\n]{1,50}\]\s*)+)", transcript)
            if speaker_prefix_match:
                prefix = speaker_prefix_match.group(1).strip()
                body = transcript[speaker_prefix_match.end() :].strip()
                updated = f"{prefix}{desc_block if desc_block.startswith('[') else ' ' + desc_block}".strip()
                if body:
                    updated = f"{updated} {body}".strip()
            else:
                updated = (desc_block + (" " + transcript if transcript else "")).strip()

            # If transcript has no actual spoken text, append recovered quote(s).
            bare_text = _spoken_text_only(updated)
            has_spoken_text = len(re.sub(r"[^A-Za-z0-9']+", "", bare_text)) >= 8
            if (not has_spoken_text) and missing_quotes:
                dedup_quotes: list[str] = []
                seen_q: set[str] = set()
                for q in missing_quotes:
                    nq = _normalize_for_match(q)
                    if nq and nq not in seen_q:
                        seen_q.add(nq)
                        dedup_quotes.append(q)
                if dedup_quotes:
                    updated = (updated + " " + " ".join(dedup_quotes)).strip()

            if updated != transcript:
                seg["transcript"] = updated
                changed = True

    final_transcript = _normalize_transcript_annotations(str(seg.get("transcript", "") or ""))
    bare_final = _spoken_text_only(final_transcript)
    has_final_spoken_text = len(re.sub(r"[^A-Za-z0-9']+", "", bare_final)) >= 8
    if not has_final_spoken_text:
        final_transcript = ""
    if final_transcript != str(seg.get("transcript", "") or ""):
        seg["transcript"] = final_transcript
        changed = True

    return changed


def _prepare_xmlish_candidate(prefix_prompt: str, completion_text: str) -> str:
    """Prepare a normalized XML-ish candidate from raw model output."""
    raw = _sanitize_segments_markers_in_completion(_strip_markdown_fence(completion_text))
    raw = _normalize_known_xmlish_tags(raw)

    # If model restarts metadata/output wrappers, keep only the segment stream.
    first_segment = re.search(r"<\s*segment\b", raw, flags=re.IGNORECASE)
    if first_segment is not None and first_segment.start() > 0:
        raw = raw[first_segment.start() :]

    if _looks_repetition_loop(raw):
        raise ValueError("Generated output appears to be in a repetition loop")

    if "<example" in raw:
        candidate = raw[raw.find("<example") :]
    else:
        candidate = prefix_prompt + raw

    if "</example>" in candidate:
        # Keep only the first full example to avoid repetition loops like
        # </example><example ...><example ...> in long generations.
        candidate = candidate[: candidate.find("</example>") + len("</example>")]
    else:
        if "</segments>" not in candidate:
            candidate += "\n    </segments>"
        if "</output>" not in candidate:
            candidate += "\n  </output>"
        if "</example>" not in candidate:
            candidate += "\n</example>"

    # Remove nested structural wrappers accidentally emitted inside segment bodies.
    first_segment_in_candidate = re.search(r"<\s*segment\b", candidate, flags=re.IGNORECASE)
    if first_segment_in_candidate is not None:
        head = candidate[: first_segment_in_candidate.start()]
        body = candidate[first_segment_in_candidate.start() :]
        # Drop nested opening wrappers accidentally emitted inside segment streams,
        # but keep closing wrappers required to finish a valid example.
        body = re.sub(r"<\s*input\s*>", "", body, flags=re.IGNORECASE)
        body = re.sub(r"</\s*input\s*>", "", body, flags=re.IGNORECASE)
        body = re.sub(r"<\s*output\s*>", "", body, flags=re.IGNORECASE)
        body = re.sub(r"<\s*segments\s*>", "", body, flags=re.IGNORECASE)
        candidate = head + body

    # Enforce schema-level formatting rules and segment timing requirements.
    candidate = _enforce_xmlish_rules(candidate)

    # Inject missing structural wrappers (</input>, <output>, <segments>) before segment stream
    # when the model re-emitted <example> but skipped the transition tags.
    _first_seg = re.search(r"<\s*segment\b", candidate, flags=re.IGNORECASE)
    if _first_seg is not None:
        _before = candidate[:_first_seg.start()]
        _after = candidate[_first_seg.start():]
        _inject = ""
        if not re.search(r"</\s*input\s*>", _before, re.IGNORECASE):
            _inject += "\n</input>"
        if not re.search(r"<\s*output\b", _before, re.IGNORECASE):
            _inject += "\n<output>"
        if not re.search(r"<\s*segments\b", _before, re.IGNORECASE):
            _inject += "\n<segments>"
        if _inject:
            candidate = _before.rstrip() + _inject + "\n" + _after

    # If generation ends with unclosed segments, close them so XML remains parseable.
    opens = len(re.findall(r"<\s*segment\b", candidate, flags=re.IGNORECASE))
    closes = len(re.findall(r"</\s*segment\s*>", candidate, flags=re.IGNORECASE))
    if opens > closes:
        candidate = candidate.rstrip() + ("\n</segment>" * (opens - closes))

    # Canonicalize closing order by dropping stray closers and appending one final sequence.
    candidate = re.sub(r"</\s*segments\s*>", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"</\s*output\s*>", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"</\s*example\s*>", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.rstrip() + "\n</segments>\n</output>\n</example>"

    candidate = _normalize_speaker_tags(candidate)
    candidate = _escape_bare_ampersands(candidate)

    return candidate


def finalize_xmlish_record(prefix_prompt: str, completion_text: str, *, force_speech: bool = False) -> str:
    """Finalize model output into a valid XML-ish record and validate parseability."""
    candidate = _prepare_xmlish_candidate(prefix_prompt, completion_text)

    def _validate_xmlish(text: str) -> ET.Element:
        try:
            parsed = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError(f"Generated XML-ish output is invalid: {exc}") from exc

        segments_parent = parsed.find(".//segments")
        segments = segments_parent.findall("segment") if segments_parent is not None else []
        if segments_parent is None or not segments:
            raise ValueError("Generated XML-ish output has no <segment> elements")

        for idx, seg in enumerate(segments, start=1):
            scene_elem = seg.find("scene")
            scene_text = (scene_elem.text or "").strip() if scene_elem is not None else ""
            if not scene_text:
                raise ValueError(f"Generated XML-ish output has empty <scene> in segment {idx}")

            transcript_elems = seg.findall("transcript")
            audio_elems = seg.findall("audio")
            transcript_text = " ".join((e.text or "").strip() for e in transcript_elems if (e.text or "").strip())
            audio_text = " ".join((e.text or "").strip() for e in audio_elems if (e.text or "").strip())
            if force_speech and not transcript_text:
                raise ValueError(
                    f"Generated XML-ish output has empty/missing <transcript> in segment {idx}"
                )
            if not transcript_text and not audio_text:
                raise ValueError(
                    f"Generated XML-ish output missing transcript/audio in segment {idx}"
                )
        return parsed

    # Validate the normalized raw candidate directly.
    _validate_xmlish(candidate)
    return candidate.strip() + "\n"


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    """Generate model text for a prompt using shared decoding code."""
    if _CLOUD_ENDPOINT_CFG is not None:
        endpoint = str(_CLOUD_ENDPOINT_CFG.get("url") or "").strip()
        if not endpoint:
            raise RuntimeError("Cloud endpoint URL is not configured")

        token = str(_CLOUD_ENDPOINT_CFG.get("token") or "").strip() or None
        timeout_s = int(_CLOUD_ENDPOINT_CFG.get("timeout_s") or 240)
        max_retries = int(_CLOUD_ENDPOINT_CFG.get("max_retries") or 2)

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        payload: dict[str, Any] = {
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "repetition_penalty": float(repetition_penalty),
                "return_full_text": False,
            },
            "options": {
                "wait_for_model": True,
            },
        }
        if do_sample:
            payload["parameters"]["temperature"] = float(temperature)
            payload["parameters"]["top_p"] = float(top_p)

        print(
            "[Cloud] Dispatching generation request "
            f"to {endpoint} (prompt_chars={len(prompt_text)}, max_new_tokens={max_new_tokens}, retries={max_retries})",
            flush=True,
        )

        last_exc: Exception | None = None
        for attempt_idx in range(max_retries + 1):
            try:
                print(f"[Cloud] Attempt {attempt_idx + 1}/{max_retries + 1}", flush=True)
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout_s,
                )
                print(f"[Cloud] Response status: {resp.status_code}", flush=True)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Endpoint {endpoint} returned HTTP {resp.status_code}: {resp.text[:800]}")

                data = resp.json()
                completion = ""
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        completion = str(first.get("generated_text") or first.get("text") or "")
                    else:
                        completion = str(first)
                elif isinstance(data, dict):
                    if isinstance(data.get("choices"), list) and data["choices"]:
                        choice0 = data["choices"][0]
                        if isinstance(choice0, dict):
                            completion = str(
                                choice0.get("text")
                                or ((choice0.get("message") or {}).get("content") if isinstance(choice0.get("message"), dict) else "")
                                or ""
                            )
                    if not completion:
                        completion = str(data.get("generated_text") or data.get("text") or data.get("output_text") or "")

                completion = completion.strip()
                if completion.startswith(prompt_text):
                    completion = completion[len(prompt_text) :].lstrip()

                text = _normalize_generated_text(completion)
                text = _normalize_known_xmlish_tags(text)
                text = _normalize_speaker_tags(text)
                text = _drop_disallowed_xmlish_tags(text)
                print("[Cloud] Request succeeded", flush=True)
                return text
            except Exception as exc:
                last_exc = exc
                print(f"[Cloud] Request failed: {exc}", flush=True)
                if attempt_idx >= max_retries:
                    break
                time.sleep(1.0 + attempt_idx)

        raise RuntimeError("Cloud generation failed after retries") from last_exc

    _ensure_ml_imports()
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    generate_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)

    generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]

    decoded_default = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    # Do not use convert_tokens_to_string path here. On some tokenizer/backends it can
    # surface byte-fallback tokens like <0x0A> verbatim and amplify repetition artifacts.
    text = _normalize_generated_text(decoded_default)
    text = _normalize_known_xmlish_tags(text)
    text = _normalize_speaker_tags(text)
    text = _drop_disallowed_xmlish_tags(text)
    return text


def _patch_hf_symlinks() -> None:
    """
    On Windows without Developer Mode, os.symlink can fail (WinError 1314).
    Patch HF Hub internals to fall back to plain copies.
    """
    try:
        import huggingface_hub.file_download as _fd

        _orig = _fd._create_symlink

        def _safe_symlink(src: str, dst: str, new_blob: bool = False) -> None:
            try:
                _orig(src, dst, new_blob=new_blob)
            except OSError:
                if not os.path.exists(dst):
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)

        _fd._create_symlink = _safe_symlink
    except Exception:
        pass


def _patch_yarn_rope() -> None:
    """
    Monkey-patch transformers._compute_yarn_parameters to handle models where
    head_dim is missing from rope_scaling (e.g. SuperAGI/mistral-7B-PoSE-32k).
    The newer transformers versions try to read rope_scaling["head_dim"] and
    crash with TypeError when that key is absent.
    """
    try:
        import transformers.modeling_rope_utils as _ru

        _orig_yarn = _ru._compute_yarn_parameters

        def _patched_yarn(config, device):
            rope_cfg = getattr(config, "rope_scaling", None) or {}
            if isinstance(rope_cfg, dict) and rope_cfg.get("head_dim") is None:
                hidden = getattr(config, "hidden_size", None)
                heads = getattr(config, "num_attention_heads", None)
                if hidden and heads:
                    computed = hidden // heads
                    try:
                        rope_cfg["head_dim"] = computed
                    except TypeError:
                        pass
                    try:
                        object.__setattr__(config, "rope_scaling", dict(rope_cfg, head_dim=computed))
                    except Exception:
                        pass
                    try:
                        object.__setattr__(config, "head_dim", computed)
                    except Exception:
                        pass
            return _orig_yarn(config, device)

        _ru._compute_yarn_parameters = _patched_yarn
    except Exception:
        pass


def _safe_snapshot_download(repo_id: str, ignore_patterns: list[str], local_dir: Path) -> str:
    """Download snapshot to local_dir, avoiding symlink-based cache paths when possible."""
    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "ignore_patterns": ignore_patterns,
        "local_dir": str(local_dir),
    }
    sig = inspect.signature(snapshot_download)
    if "local_dir_use_symlinks" in sig.parameters:
        kwargs["local_dir_use_symlinks"] = False
    return snapshot_download(**kwargs)


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Path | None,
    use_4bit: bool,
    force_no_hub_adapter: bool,
    base_cache_dir: Path,
) -> tuple[Any, Any]:
    """Load either a merged model directly, or base model + LoRA adapter."""
    _ensure_ml_imports()
    using_adapter = adapter_path is not None
    if using_adapter and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    # Disable HF symlink warning (common on Windows without Developer Mode).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _patch_hf_symlinks()
    _patch_yarn_rope()

    # For LoRA adapters, always prefer the original base tokenizer. Also prefer
    # the standard slow tokenizer implementation first so decode goes through the
    # model's native SentencePiece path instead of a custom fast backend.
    tokenizer_source = base_model_name

    def _load_tokenizer_robust(source: str) -> Any:
        """Load tokenizer with fallback for additional_chat_templates 404 errors."""
        for trust in (False, True):
            for fast in (False, True):
                try:
                    return AutoTokenizer.from_pretrained(
                        source,
                        trust_remote_code=trust,
                        use_fast=fast,
                    )
                except Exception as exc:
                    last_exc = exc
                    if "additional_chat_templates" not in str(exc):
                        continue
                    # Snap a local copy and load from disk to bypass the 404.
                    from huggingface_hub import snapshot_download
                    try:
                        local_dir = snapshot_download(
                            repo_id=source,
                            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.ot"],
                        )
                        return AutoTokenizer.from_pretrained(
                            local_dir,
                            trust_remote_code=trust,
                            use_fast=fast,
                            local_files_only=True,
                        )
                    except Exception:
                        pass
        raise RuntimeError(f"Could not load tokenizer for {source!r}") from last_exc

    tokenizer = _load_tokenizer_robust(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _load_config_patched(path_or_id: str) -> Any:
        """Load config and fix head_dim=None in YaRN rope configs in-memory."""
        cfg = AutoConfig.from_pretrained(path_or_id, trust_remote_code=True)
        rope_cfg = getattr(cfg, "rope_scaling", None)
        if isinstance(rope_cfg, dict) and rope_cfg.get("type") == "yarn":
            if rope_cfg.get("head_dim") is None:
                hidden = getattr(cfg, "hidden_size", None)
                heads = getattr(cfg, "num_attention_heads", None)
                if hidden and heads:
                    computed = hidden // heads
                    # Patch in-place and reassign (bypasses most validators)
                    try:
                        rope_cfg["head_dim"] = computed
                    except TypeError:
                        pass
                    try:
                        object.__setattr__(cfg, "rope_scaling", dict(rope_cfg, head_dim=computed))
                    except Exception:
                        cfg.rope_scaling = dict(rope_cfg, head_dim=computed)
                    # Also set as a direct config attribute (fallback path in _compute_yarn_parameters)
                    try:
                        object.__setattr__(cfg, "head_dim", computed)
                    except Exception:
                        pass
        return cfg

    def _load_from(path_or_id: str) -> Any:
        cfg = _load_config_patched(path_or_id)
        return AutoModelForCausalLM.from_pretrained(
            path_or_id,
            config=cfg,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16 if not use_4bit else None,
        )

    model_path = Path(base_model_name)
    is_local_model_path = model_path.exists()
    local_dir = base_cache_dir / base_model_name.replace("/", "--")

    if force_no_hub_adapter and not is_local_model_path:
        print("Loading base model from snapshot without hub adapter artifacts...")
        local_path = _safe_snapshot_download(
            repo_id=base_model_name,
            ignore_patterns=["adapter_model.safetensors", "adapter_config.json"],
            local_dir=local_dir,
        )
        base_model = _load_from(local_path)
    else:
        try:
            base_model = _load_from(base_model_name)
        except AttributeError as exc:
            if "is not an nn.Module" not in str(exc):
                raise
            print("Detected adapter-loading crash. Retrying with cleaned local snapshot...")
            local_path = _safe_snapshot_download(
                repo_id=base_model_name,
                ignore_patterns=["adapter_model.safetensors", "adapter_config.json"],
                local_dir=local_dir,
            )
            base_model = _load_from(local_path)

    if using_adapter:
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def resolve_adapter_path(requested: Path | None) -> Path | None:
    """Resolve adapter path with sane defaults: final > latest checkpoint."""
    if requested is not None:
        return requested

    base = Path("models/carmind2_lora")
    final_path = base / "final"
    if final_path.exists():
        return final_path

    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        return checkpoints[-1]

    return None


def is_lora_adapter_dir(path: Path) -> bool:
    """Return True if a folder looks like a PEFT adapter checkpoint/directory."""
    if not path.exists() or not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def resolve_base_from_adapter_config(adapter_path: Path) -> str | None:
    """Read adapter_config.json and return base_model_name_or_path when present."""
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    base = payload.get("base_model_name_or_path")
    return str(base).strip() if base else None


def resolve_latest_longcontext_merged_model() -> Path | None:
    """Prefer the most recent local longcontext merged model when available."""
    candidates: list[Path] = []
    for path in Path(".").glob("longcontext-*"):
        merged = path / "merged_model"
        if merged.exists() and merged.is_dir():
            candidates.append(merged)

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.parent.name)[-1]


def resolve_merged_model_for_adapter(adapter_path: Path | None) -> Path | None:
    """If adapter is in a run folder, return that run's merged_model when present."""
    if adapter_path is None:
        return None

    adapter_path = adapter_path.resolve()
    candidates = [
        adapter_path / "merged_model",
        adapter_path.parent / "merged_model",
        adapter_path.parent.parent / "merged_model",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def build_fallback_xmlish_record(
    prefix_prompt: str,
    duration: int,
    creator: str,
    topic: str,
    title: str,
    force_speech: bool = False,
) -> str:
    """Build a minimal valid XML-ish record when generation is unusable."""
    safe_dur = round(float(max(1, duration)), 2)
    segment = (
        "      <segment>\n"
        f"        <start>0.0</start>\n"
        f"        <end>{_xml_text(safe_dur)}</end>\n"
        f"        <duration>{_xml_text(safe_dur)}</duration>\n"
        f"        <scene>{_xml_text(f'A short-form segment in {creator} style about {topic}. Primary angle: {title}.')}</scene>\n"
        f"        <transcript>{'T' if force_speech else ''}</transcript>\n"
        "      </segment>"
    )
    return finalize_xmlish_record(prefix_prompt, segment, force_speech=force_speech)


def main() -> None:
    print("[Generator] Startup", flush=True)
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Generate an XML-ish training example using local fine-tuned model",
    )
    parser.add_argument("--sources-file", type=Path, default=None, help="JSON file containing a list of sources")
    parser.add_argument("--source", action="append", default=[], help="Inline source URL (repeatable)")
    parser.add_argument("--duration", type=int, default=None, help="Target short duration in seconds (default: 45)")
    parser.add_argument("--creator", type=str, default=None, help="Creator style name (default: Noner)")
    parser.add_argument("--topic", type=str, default=None, help="Topic/category (default: General)")
    parser.add_argument("--title", type=str, default=None, help="Optional title for input payload (default: generated by LLM)")
    parser.add_argument("--force-speech", action="store_true", help="Force every segment to have non-empty transcript text")

    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model id/path. Ignored when local longcontext merged model is auto-detected.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to LoRA adapter directory (default: auto, prefers models/carmind2_lora/final)",
    )
    parser.add_argument(
        "--no-auto-adapter",
        action="store_true",
        help="Disable automatic adapter fallback when --adapter is not provided",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training_dataset/generated_example.xmlish.txt"),
        help="Output XML-ish example path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--no-sample", action="store_true", help="Use deterministic decoding")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument(
        "--force-no-hub-adapter",
        action="store_true",
        help="Always load base model from local snapshot that excludes hub adapter artifacts",
    )
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=Path("models/base_models"),
        help="Local cache directory for base model snapshots",
    )
    parser.add_argument(
        "--repair-spacing",
        action="store_true",
        help="Apply aggressive text-node spacing repair on generated XML (can over-merge words)",
    )
    parser.add_argument(
        "--no-prefer-merged",
        action="store_true",
        help="When using --adapter, do not auto-switch to a sibling merged_model if found",
    )
    parser.add_argument(
        "--nlp-spacing-clean",
        action="store_true",
        help="Apply pretrained-LM spacing cleanup to JSON fields (scene/transcript/audio)",
    )
    parser.add_argument(
        "--nlp-scorer-model",
        type=str,
        default="distilgpt2",
        help="Pretrained LM id used for spacing scoring when --nlp-spacing-clean is enabled",
    )
    parser.add_argument(
        "--max-continuations",
        type=int,
        default=12,
        help="If XML is incomplete, request additional continuation passes before failing",
    )
    parser.add_argument(
        "--cloud-endpoint-url",
        type=str,
        default=None,
        help="Optional Hugging Face Inference Endpoint URL. When set, generation uses remote endpoint instead of local model loading.",
    )
    parser.add_argument(
        "--cloud-endpoint-token",
        type=str,
        default=None,
        help="Optional token for cloud endpoint auth. Defaults to HF_TOKEN/HUGGINGFACE_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--cloud-timeout-s",
        type=int,
        default=240,
        help="HTTP timeout in seconds for cloud endpoint requests",
    )
    parser.add_argument(
        "--cloud-max-retries",
        type=int,
        default=2,
        help="Retry count for failed cloud endpoint requests",
    )

    args = parser.parse_args()
    adapter_was_explicit = args.adapter is not None
    if not args.no_auto_adapter:
        args.adapter = resolve_adapter_path(args.adapter)

    base_model_name = args.base_model
    detected_adapter_from_base: Path | None = None

    # Allow passing an unmerged adapter folder via --base-model for convenience.
    base_model_path = Path(args.base_model)
    if args.adapter is None and is_lora_adapter_dir(base_model_path):
        detected_adapter_from_base = base_model_path
        args.adapter = base_model_path
        base_model_name = DEFAULT_BASE_MODEL

    # If an adapter is selected, prefer its recorded base model to avoid
    # tokenizer/model mismatch when users pass a different --base-model.
    if args.adapter is not None:
        adapter_base = resolve_base_from_adapter_config(args.adapter)
        if adapter_base and adapter_base != base_model_name:
            print(
                "Adapter base model differs from requested base; "
                f"using adapter's base model: {adapter_base}"
            )
            base_model_name = adapter_base

    auto_merged_model = None
    # If user explicitly passed --adapter, respect it and do not auto-switch
    # to a sibling merged_model unless they intentionally omit adapter.
    if args.adapter is not None and not args.no_prefer_merged and not adapter_was_explicit:
        auto_merged_model = resolve_merged_model_for_adapter(args.adapter)
        if auto_merged_model is not None:
            print(
                "Found merged model for adapter; using merged weights for inference: "
                f"{auto_merged_model}"
            )
            args.adapter = None

    if args.adapter is None and auto_merged_model is None:
        auto_merged_model = resolve_latest_longcontext_merged_model()

    effective_model = str(auto_merged_model) if auto_merged_model else base_model_name

    use_cloud_endpoint = bool((args.cloud_endpoint_url or "").strip())
    if use_cloud_endpoint:
        token = (
            (args.cloud_endpoint_token or "").strip()
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or ""
        )
        global _CLOUD_ENDPOINT_CFG
        _CLOUD_ENDPOINT_CFG = {
            "url": args.cloud_endpoint_url.strip(),
            "token": token,
            "timeout_s": int(args.cloud_timeout_s),
            "max_retries": max(0, int(args.cloud_max_retries)),
        }
        print(f"Using cloud endpoint for generation: {_CLOUD_ENDPOINT_CFG['url']}")

    if not use_cloud_endpoint and auto_merged_model:
        print(f"Using local merged model: {effective_model}")
    elif not use_cloud_endpoint and detected_adapter_from_base is not None:
        print(
            "Detected LoRA adapter folder passed via --base-model; "
            f"using base model + adapter: {base_model_name} + {detected_adapter_from_base}"
        )
    elif not use_cloud_endpoint and args.adapter:
        print(f"Using base model + adapter: {base_model_name} + {args.adapter}")
    elif not use_cloud_endpoint:
        print(f"Using base model directly: {effective_model}")

    sources = load_sources(args.sources_file, args.source)

    if use_cloud_endpoint:
        model = None
        tokenizer = None
    else:
        model, tokenizer = load_model_and_tokenizer(
            base_model_name=effective_model,
            adapter_path=args.adapter,
            use_4bit=not args.no_4bit,
            force_no_hub_adapter=args.force_no_hub_adapter,
            base_cache_dir=args.base_cache_dir,
        )

    title_raw = (args.title or "").strip()
    title = title_raw if title_raw else None
    if not title:
        print("Generating title with LLM...")
        try:
            title = generate_title_with_llm(
                model,
                tokenizer,
                creator=args.creator,
                topic=args.topic,
                duration=args.duration,
                sources=sources,
            )
            print(f"Generated title: {title}")
        except Exception as exc:
            title = f"{(args.topic or 'General').strip() or 'General'} Short"
            print(f"[INFO] Title generation failed; using fallback title: {title} ({exc})")

    user_prompt = build_xmlish_prefix_prompt(
        creator=args.creator,
        topic=args.topic,
        duration=args.duration,
        sources=sources,
        title=title,
    )

    # Raw continuation: no chat wrapper, direct prefix -> next token prediction.
    prompt_text = user_prompt

    generated_text = ""
    final_xml: str | None = None
    first_error: ValueError | None = None
    fallback_used = False

    for attempt_idx in range(max(0, args.max_continuations) + 1):
        prompt_for_pass = prompt_text if attempt_idx == 0 else (user_prompt + generated_text)
        generated_chunk = generate_text(
            model,
            tokenizer,
            prompt_for_pass,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        generated_text = generated_text + generated_chunk
        generated_text = _normalize_known_xmlish_tags(generated_text)
        generated_text = _normalize_speaker_tags(generated_text)
        generated_text = _drop_disallowed_xmlish_tags(generated_text)

        try:
            final_xml = finalize_xmlish_record(
                prefix_prompt=user_prompt,
                completion_text=generated_text,
                force_speech=args.force_speech,
            )
            break
        except ValueError as err:
            first_error = err
            prepared_candidate = None
            try:
                prepared_candidate = _prepare_xmlish_candidate(user_prompt, generated_text)
            except Exception:
                prepared_candidate = None

            if prepared_candidate is not None:
                generated_text = _extract_segment_stream(prepared_candidate)

            # Detect malformed transcript blocks (e.g. newline immediately
            # after <transcript>) and reseed with a speaker scaffold.
            seeded_transcript_retry = _seed_first_bad_transcript_segment(generated_text)
            if seeded_transcript_retry is not None and attempt_idx < max(0, args.max_continuations):
                generated_text = seeded_transcript_retry
                print(
                    "[INFO] Detected malformed transcript block; reseeding continuation "
                    f"with '<transcript>[SPEAKER_' ({attempt_idx + 1}/{max(0, args.max_continuations)})."
                )
                continue

            # Detect first segment with missing/empty scene and reseed continuation
            # directly from that point.
            seeded_retry = _seed_first_missing_or_empty_scene_segment(generated_text)
            if seeded_retry is not None and attempt_idx < max(0, args.max_continuations):
                generated_text = seeded_retry
                print(
                    "[INFO] Detected missing/empty scene segment; reseeding continuation "
                    f"from first bad segment ({attempt_idx + 1}/{max(0, args.max_continuations)})."
                )
                continue

            if args.force_speech:
                seeded_force_speech_retry = _seed_first_missing_or_empty_transcript_segment(generated_text)
                if seeded_force_speech_retry is not None and attempt_idx < max(0, args.max_continuations):
                    generated_text = seeded_force_speech_retry
                    print(
                        "[INFO] Detected empty/missing transcript under --force-speech; reseeding continuation "
                        f"with '<transcript>[SPEAKER_' ({attempt_idx + 1}/{max(0, args.max_continuations)})."
                    )
                    continue

            # Ensure each segment has transcript or audio; if both are missing,
            # reseed continuation with an audio scaffold.
            seeded_audio_retry = _seed_first_missing_transcript_and_audio_segment(generated_text)
            if seeded_audio_retry is not None and attempt_idx < max(0, args.max_continuations):
                generated_text = seeded_audio_retry
                print(
                    "[INFO] Detected segment missing transcript/audio; reseeding continuation "
                    f"with '<audio>a ' ({attempt_idx + 1}/{max(0, args.max_continuations)})."
                )
                continue

            has_empty_scene = False
            if prepared_candidate is not None and _has_empty_scene_tags(prepared_candidate):
                has_empty_scene = True
            elif _has_empty_scene_tags(user_prompt + generated_text):
                has_empty_scene = True

            maybe_incomplete = "</output>" not in (user_prompt + generated_text).lower()

            if has_empty_scene and attempt_idx < max(0, args.max_continuations):
                generated_text = _seed_last_empty_scene_for_retry(generated_text)
                print(
                    "[INFO] Detected empty <scene> tag; requesting continuation "
                    f"with seeded scene text ({attempt_idx + 1}/{max(0, args.max_continuations)})."
                )
                continue

            if maybe_incomplete and attempt_idx < max(0, args.max_continuations):
                print(
                    "[INFO] Generation appears incomplete; requesting continuation "
                    f"({attempt_idx + 1}/{max(0, args.max_continuations)})."
                )
                continue
            break

    if final_xml is not None:
        if args.repair_spacing:
            final_xml = _repair_xmlish_text_nodes(final_xml)
    else:
        # Save first-pass raw output to help debugging when model drifts from XML.
        raw_path = args.output.with_suffix(args.output.suffix + ".raw.txt")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(generated_text)

        full_raw_path = args.output.with_suffix(args.output.suffix + ".full.raw.txt")
        with open(full_raw_path, "w", encoding="utf-8") as f:
            f.write(user_prompt + generated_text)

        normalized_path = args.output.with_suffix(args.output.suffix + ".normalized.raw.txt")
        try:
            normalized_candidate = _prepare_xmlish_candidate(user_prompt, generated_text)
            with open(normalized_path, "w", encoding="utf-8") as f:
                f.write(normalized_candidate)
        except Exception:
            normalized_path = None

        details = [
            "Generation failed.",
            f"Raw output saved to: {raw_path}.",
            f"Full prefixed output saved to: {full_raw_path}.",
            f"Validation error: {first_error if first_error is not None else 'unknown'}",
        ]
        if normalized_path is not None:
            details.append(f"Normalized candidate saved to: {normalized_path}.")

        raise SystemExit(" ".join(details)) from first_error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final_xml)

    meta_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "generator": "generate_example_with_carmind2_lora.py",
                "base_model": base_model_name,
                "model_loaded_from": effective_model,
                "adapter": str(args.adapter) if args.adapter else None,
                "format": "xmlish",
                "fallback_used": fallback_used,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Generated XML-ish example saved to: {args.output}")
    print(f"     Metadata: {meta_path}")

    # Extract structured JSON from the normalized XML-ish output
    if _XMLISH_JSON_AVAILABLE:
        json_path = args.output.with_suffix(".json")
        try:
            # Read back from disk and normalize known tag mistakes before extraction
            with open(args.output, "r", encoding="utf-8") as f:
                xml_for_json = f.read()
            xml_for_json = _normalize_known_xmlish_tags(xml_for_json)
            
            normalized = _xe_normalize_tags(xml_for_json)
            blocks = _XE_EXAMPLE_BLOCK_RE.findall(normalized)
            if blocks:
                example_elem = ET.fromstring(blocks[0])
                _example_id, example_obj = _xe_parse_example(
                    example_elem, fallback_id=args.output.stem
                )
            else:
                _example_id, example_obj = _xe_parse_fragment(
                    normalized, fallback_id=args.output.stem
                )

            # Always apply punctuation spacing fix to JSON fields
            segs = example_obj.get("output", {}).get("segments", [])
            fields = ("scene_description", "transcript", "audio_description")
            punct_fixed = 0
            payload_reclassified = 0
            for seg in segs if isinstance(segs, list) else []:
                if not isinstance(seg, dict):
                    continue
                for field in fields:
                    if field not in seg:
                        continue
                    raw_val = str(seg.get(field, ""))
                    if not raw_val.strip():
                        continue
                    fixed_val = fix_punctuation_spacing(_normalize_speaker_tags(raw_val))
                    if fixed_val != raw_val:
                        seg[field] = fixed_val
                        punct_fixed += 1

                if _reclassify_segment_payload(seg):
                    payload_reclassified += 1
            
            if punct_fixed > 0:
                print(f"[INFO] Punctuation spacing fixed: {punct_fixed} fields")
            if payload_reclassified > 0:
                print(f"[INFO] Scene payload reclassified: {payload_reclassified} segments")
            
            # Optional: Apply NLP-based spacing cleanup if requested
            if args.nlp_spacing_clean:
                segs = example_obj.get("output", {}).get("segments", [])
                changed = 0
                seen = 0
                for seg in segs if isinstance(segs, list) else []:
                    if not isinstance(seg, dict):
                        continue
                    for field in fields:
                        if field not in seg:
                            continue
                        raw_val = str(seg.get(field, ""))
                        if not raw_val.strip():
                            continue
                        seen += 1
                        fixed_val = _nlp_repair_spacing_text(raw_val, scorer_model_id=args.nlp_scorer_model)
                        if fixed_val and fixed_val != raw_val:
                            seg[field] = fixed_val
                            changed += 1
                print(
                    "[INFO] NLP spacing clean: "
                    f"segments={len(segs) if isinstance(segs, list) else 0}, "
                    f"fields_seen={seen}, fields_changed={changed}, "
                    f"scorer={args.nlp_scorer_model}"
                )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(example_obj, f, indent=2, ensure_ascii=False)
            print(f"[OK] Structured JSON saved to:          {json_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] JSON extraction failed: {exc}")


if __name__ == "__main__":
    main()
