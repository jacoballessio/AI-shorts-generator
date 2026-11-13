"""
Convert XML-ish training data back into *_example.json files.

Supports:
- Full corpus files containing one or more <example>...</example> records
- Segment-only fragments (fallback), producing one generated example

Usage:
  python src/training/convert_xmlish_to_examples.py \
    --input-file data/training_dataset_xmlish.txt \
    --output-dir data/training_dataset_from_xmlish
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


EXAMPLE_BLOCK_RE = re.compile(r"<example\b[\s\S]*?</example>", re.IGNORECASE)
AUDIO_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+",
)
UI_CONTEXT_RE = re.compile(
    r"\b(reads|read|text|caption|captions|letters|phrase|phrases|overlay|on-screen)\b",
    re.IGNORECASE,
)
AUDIO_CONTEXT_RE = re.compile(
    r"\b(audio|voice|music|sound|sfx|speaks|saying|says|exclaims|tone|ringtone|beep|whoosh|bang|laugh)\b",
    re.IGNORECASE,
)
QUOTED_TEXT_RE = re.compile(r'"([^"]+)"')
HASHTAG_RE = re.compile(r"(?<!\w)(#[A-Za-z0-9_]+)")
SEGMENT_BLOCK_RE = re.compile(r"<segment\b[\s\S]*?</segment>", re.IGNORECASE)


def _normalize_spaced_xml_tags(text: str) -> str:
    """Normalize tokenized XML-ish tags like '< segment >' -> '<segment>'."""
    text = text.replace("▁", " ")

    def _fix_tag(match: re.Match[str]) -> str:
        slash = match.group(1)
        name = re.sub(r"\s+", "", match.group(2))
        return f"<{slash}{name}>"

    text = re.sub(r"<\s*(/?)\s*([A-Za-z_][A-Za-z0-9_\s]*)\s*>", _fix_tag, text)
    return text


def _text(parent: ET.Element | None, tag: str, default: str = "") -> str:
    if parent is None:
        return default
    node = parent.find(tag)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    normalized = value.strip()
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    try:
        return round(float(normalized), 2)
    except (TypeError, ValueError):
        return default


def _to_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    normalized = value.strip()
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    try:
        return int(float(normalized))
    except (TypeError, ValueError):
        return default


def _safe_id(raw: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", (raw or "").strip()).strip("_")
    return value or fallback


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in AUDIO_SENTENCE_RE.split(text.strip()) if part.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _extract_ui_texts(scene_text: str) -> list[str]:
    ui_texts: list[str] = []
    for sentence in _split_sentences(scene_text):
        if not UI_CONTEXT_RE.search(sentence):
            continue
        ui_texts.extend(QUOTED_TEXT_RE.findall(sentence))
        ui_texts.extend(HASHTAG_RE.findall(sentence))
    return _dedupe_keep_order(ui_texts)


def _extract_audio_sentences(scene_text: str) -> list[str]:
    audio_sentences: list[str] = []
    for sentence in _split_sentences(scene_text):
        if AUDIO_CONTEXT_RE.search(sentence):
            audio_sentences.append(sentence)
    return _dedupe_keep_order(audio_sentences)


def _build_ui_elements_html(ui_texts: list[str]) -> str:
    if not ui_texts:
        return ""
    html_lines: list[str] = []
    for text in ui_texts:
        css_class = "overlay hashtag" if text.startswith("#") else "overlay caption"
        html_lines.append(f'<div class="{css_class}">{text}</div>')
    return "\n".join(html_lines)


def _normalize_audio_description(audio_sentences: list[str], transcript: str) -> str:
    if not audio_sentences:
        return ""
    transcript_norm = (transcript or "").strip().lower().strip('"')
    cleaned: list[str] = []
    for sentence in audio_sentences:
        candidate = sentence.strip()
        if transcript_norm:
            quoted = [q.strip().lower() for q in QUOTED_TEXT_RE.findall(candidate)]
            if quoted and all(q == transcript_norm for q in quoted):
                candidate = QUOTED_TEXT_RE.sub("", candidate)
                candidate = re.sub(r"\s{2,}", " ", candidate).strip(" ,.-")
        candidate = re.sub(r",?\s*\b(saying|says)\b", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s{2,}", " ", candidate).strip(" ,.-")
        if candidate:
            cleaned.append(candidate)
    return " ".join(_dedupe_keep_order(cleaned))


def _remove_inferred_details_from_scene(scene_text: str, audio_sentences: list[str], ui_texts: list[str]) -> str:
    stripped = scene_text
    for sentence in audio_sentences:
        stripped = stripped.replace(sentence, " ")
    for text in ui_texts:
        stripped = stripped.replace(f'"{text}"', " ")
        stripped = stripped.replace(text, " ")
    stripped = re.sub(r",?\s*and below it\s+[^.?!]*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r",?\s*below it\s+[^.?!]*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(
        r"\bLarge\s+[^.?!]*\b(?:read|reads|appear|appears|change|changes|form)\b[^.?!]*",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.!?;:])", r"\1", stripped)
    stripped = re.sub(r"([.!?]){2,}", r"\1", stripped)
    return stripped.strip()


def _parse_sources(input_elem: ET.Element | None) -> list[dict[str, str]]:
    if input_elem is None:
        return []

    sources_elem = input_elem.find("sources")
    if sources_elem is None:
        return []

    out: list[dict[str, str]] = []
    for src in sources_elem.findall("source"):
        out.append(
            {
                "title": (src.get("title") or "").strip(),
                "description": (src.get("description") or "").strip(),
                "text": (src.text or "").strip(),
                "url": (src.get("url") or "").strip(),
                "type": (src.get("type") or "article").strip(),
                "source_name": (src.get("source_name") or "").strip(),
                "published_at": (src.get("published_at") or "").strip(),
            }
        )
    return out


def _parse_segment_elem(seg: ET.Element) -> dict[str, Any]:
    start = _to_float(_text(seg, "start", "0"), 0.0)
    end = _to_float(_text(seg, "end", "0"), 0.0)
    duration = _to_float(_text(seg, "duration", ""), 0.0)
    if duration <= 0 and end > start:
        duration = round(end - start, 2)

    scene_text = _text(seg, "scene", "")
    transcript = _text(seg, "transcript", "")
    explicit_ui = _text(seg, "ui", "")
    explicit_audio = _text(seg, "audio", "")

    inferred_ui_texts = _extract_ui_texts(scene_text) if not explicit_ui else []
    inferred_audio_sentences = _extract_audio_sentences(scene_text) if not explicit_audio else []
    normalized_audio = explicit_audio or _normalize_audio_description(inferred_audio_sentences, transcript)
    normalized_ui = explicit_ui or _build_ui_elements_html(inferred_ui_texts)
    cleaned_scene = _remove_inferred_details_from_scene(
        scene_text,
        inferred_audio_sentences,
        inferred_ui_texts,
    ) if (inferred_ui_texts or inferred_audio_sentences) else scene_text

    item: dict[str, Any] = {
        "start_time": start,
        "end_time": end,
        "scene_description": cleaned_scene,
        "ui_elements_html": normalized_ui,
        "transcript": transcript,
        "duration": duration,
    }

    if normalized_audio:
        item["audio_description"] = normalized_audio

    return item


def _parse_segments(segments_parent: ET.Element | None) -> list[dict[str, Any]]:
    if segments_parent is None:
        return []

    out: list[dict[str, Any]] = []
    for seg in segments_parent.findall("segment"):
        out.append(_parse_segment_elem(seg))

    return out


def _find_segment_container(example_elem: ET.Element) -> ET.Element | None:
    """Find where segment nodes live across common XML-ish variants."""
    output_elem = example_elem.find("output")
    if output_elem is not None:
        nested = output_elem.find("segments")
        if nested is not None:
            return nested
        # Some generations place <segment> directly under <output>.
        if output_elem.findall("segment"):
            return output_elem

    # Last-resort fallback for malformed-but-usable examples.
    if example_elem.findall("segment"):
        return example_elem

    return None


def _parse_example(example_elem: ET.Element, fallback_id: str) -> tuple[str, dict[str, Any]]:
    example_id = _safe_id(example_elem.attrib.get("id", ""), fallback_id)

    input_elem = example_elem.find("input")
    segments_elem = _find_segment_container(example_elem)

    obj = {
        "input": {
            "sources": _parse_sources(input_elem),
            "duration": _to_int(_text(input_elem, "duration", "0"), 0),
            "creator": _text(input_elem, "creator", "Unknown Creator"),
            "topic": _text(input_elem, "topic", "General"),
            "title": _text(input_elem, "title", ""),
        },
        "output": {
            "segments": _parse_segments(segments_elem),
        },
    }

    return example_id, obj


def _parse_segment_fragment(text: str, fallback_id: str) -> tuple[str, dict[str, Any]]:
    # Accept common output-only shapes when no <example> wrapper exists:
    # 1) <output><segments>...</segments></output>
    # 2) <segments>...</segments>
    # 3) raw repeated <segment>...</segment>
    normalized = _normalize_spaced_xml_tags(text)

    segments: list[dict[str, Any]] = []
    try:
        wrapped = f"<root>{normalized}</root>"
        root = ET.fromstring(wrapped)

        container: ET.Element | None = None
        output_elem = root.find("output")
        if output_elem is not None:
            container = output_elem.find("segments") or output_elem
        if container is None:
            container = root.find("segments")
        if container is None and root.findall("segment"):
            container = root

        segments = _parse_segments(container)
    except ET.ParseError:
        segment_blocks = SEGMENT_BLOCK_RE.findall(normalized)
        for block in segment_blocks:
            try:
                seg_elem = ET.fromstring(block)
            except ET.ParseError:
                continue
            segments.append(_parse_segment_elem(seg_elem))

    if not segments:
        raise ET.ParseError("No parseable <segment> nodes found in output/segments fragment")

    max_end = max((float(s.get("end_time", 0.0)) for s in segments), default=0.0)

    obj = {
        "input": {
            "sources": [],
            "duration": int(round(max_end)),
            "creator": "Unknown Creator",
            "topic": "General",
            "title": "Generated from XML-ish fragment",
        },
        "output": {
            "segments": segments,
        },
    }
    return _safe_id(fallback_id, "generated"), obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XML-ish records to *_example.json files")
    parser.add_argument("--input-file", type=Path, required=True, help="Path to XML-ish input file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/training_dataset_from_xmlish"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.input_file.exists():
        raise SystemExit(f"Input file not found: {args.input_file}")

    raw = args.input_file.read_text(encoding="utf-8")
    raw = _normalize_spaced_xml_tags(raw)
    blocks = EXAMPLE_BLOCK_RE.findall(raw)

    parsed: list[tuple[str, dict[str, Any]]] = []

    if blocks:
        for idx, block in enumerate(blocks, start=1):
            try:
                elem = ET.fromstring(block)
                parsed.append(_parse_example(elem, fallback_id=f"example_{idx:04d}"))
            except ET.ParseError as exc:
                print(f"[WARN] Skipping malformed <example> block #{idx}: {exc}")
    else:
        try:
            parsed.append(_parse_segment_fragment(raw, fallback_id=args.input_file.stem))
            print("[INFO] No <example> blocks found; parsed as output/segments fragment")
        except ET.ParseError as exc:
            raise SystemExit(f"Input is neither valid <example> blocks nor segment fragment XML-ish: {exc}")

    if not parsed:
        raise SystemExit("No valid examples were parsed")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for example_id, obj in parsed:
        out_path = args.output_dir / f"{example_id}_example.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1

    print(f"[OK] Parsed examples: {len(parsed)}")
    print(f"[OK] Written: {written}")
    print(f"[OK] Skipped existing: {skipped}")
    print(f"[OK] Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
