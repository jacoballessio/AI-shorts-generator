"""
Convert *_example.json files into an XMLish training corpus.

This is useful when you want a tag-structured text format rather than raw JSON
objects for next-token CLM training.

Cleaning defaults:
- Drop `metadata` entirely
- Keep only key input fields: sources, duration, creator, topic, title
- Keep only key segment fields: start_time, end_time, scene_description,
  ui_elements_html, transcript, duration, audio_description (if present)
- Remove XML/chat-log artifacts such as <message>, <datetime>, <sender>
- Collapse repeated whitespace and trim long text fields

Usage:
  python src/training/convert_examples_to_xmlish.py \
    --input-dir data/training_dataset_stage2 \
    --output-file data/training_dataset_xmlish.txt
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHAT_ARTIFACT_PATTERNS = [
    re.compile(r"<\s*/?\s*message\s*>", re.IGNORECASE),
    re.compile(r"<\s*datetime\s*>.*?<\s*/\s*datetime\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\s*sender\s*>.*?<\s*/\s*sender\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"!roast\s+[^\s<]+", re.IGNORECASE),
]


@dataclass
class CleanStats:
    files_seen: int = 0
    files_written: int = 0
    chars_trimmed: int = 0
    artifact_hits: int = 0
    bad_files: int = 0


def clean_text(value: str, max_len: int, stats: CleanStats) -> str:
    """Normalize text and remove common contamination artifacts."""
    if not value:
        return ""

    text = str(value)

    for pat in CHAT_ARTIFACT_PATTERNS:
        new_text, n = pat.subn(" ", text)
        if n > 0:
            stats.artifact_hits += n
        text = new_text

    # Remove low/control chars except newline and tab.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    # Collapse excessive whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > max_len:
        stats.chars_trimmed += len(text) - max_len
        text = text[:max_len].rstrip()

    return text


def clean_source(source: dict[str, Any], stats: CleanStats, max_source_text: int) -> dict[str, str]:
    """Keep source schema compact and clean."""
    return {
        "title": clean_text(str(source.get("title", "")), 160, stats),
        "description": clean_text(str(source.get("description", "")), 500, stats),
        "text": clean_text(str(source.get("text", "")), max_source_text, stats),
        "url": clean_text(str(source.get("url", "")), 500, stats),
        "type": clean_text(str(source.get("type", "article")), 60, stats),
        "source_name": clean_text(str(source.get("source_name", "")), 120, stats),
        "published_at": clean_text(str(source.get("published_at", "")), 64, stats),
    }


def _round2(value: Any) -> float:
    """Round a numeric value to 2 decimal places."""
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def clean_segment(segment: dict[str, Any], stats: CleanStats, include_ui_html: bool) -> dict[str, Any]:
    """Keep key output segment fields and sanitize text content."""
    out: dict[str, Any] = {
        "start": _round2(segment.get("start_time", 0.0)),
        "end": _round2(segment.get("end_time", 0.0)),
        "duration": _round2(segment.get("duration", 0.0)),
        "scene": clean_text(str(segment.get("scene_description", "")), 2000, stats),
        "transcript": clean_text(str(segment.get("transcript", "")), 600, stats),
    }

    if include_ui_html:
        ui = clean_text(str(segment.get("ui_elements_html", "")), 1200, stats)
        if ui:
            out["ui"] = ui

    if "audio_description" in segment and str(segment.get("audio_description", "")).strip():
        out["audio"] = clean_text(str(segment.get("audio_description", "")), 240, stats)

    return out


def clean_example(
    obj: dict[str, Any],
    stats: CleanStats,
    max_source_text: int,
    include_ui_html: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return cleaned input payload and cleaned output payload."""
    inp = obj.get("input", {}) if isinstance(obj, dict) else {}
    out = obj.get("output", {}) if isinstance(obj, dict) else {}

    sources = inp.get("sources", []) if isinstance(inp, dict) else []
    if not isinstance(sources, list):
        sources = []

    cleaned_input = {
        "sources": [clean_source(s, stats, max_source_text) for s in sources if isinstance(s, dict)],
        "duration": int(inp.get("duration", 0) or 0),
        "creator": clean_text(str(inp.get("creator", "Unknown Creator")), 120, stats),
        "topic": clean_text(str(inp.get("topic", "General")), 120, stats),
        "title": clean_text(str(inp.get("title", "")), 240, stats),
    }

    segments = out.get("segments", []) if isinstance(out, dict) else []
    if not isinstance(segments, list):
        segments = []

    cleaned_output = {
        "segments": [clean_segment(seg, stats, include_ui_html) for seg in segments if isinstance(seg, dict)]
    }

    return cleaned_input, cleaned_output


def _xml_text(value: Any) -> str:
    """Escape element text for XML-like output."""
    return html.escape(str(value), quote=False)


def _xml_attr(value: Any) -> str:
    """Escape attribute value for XML-like output."""
    return html.escape(str(value), quote=True)


def to_xmlish_record(example_id: str, cleaned_input: dict[str, Any], cleaned_output: dict[str, Any]) -> str:
    """Build XMLish record using nested input/output tags and segment time attrs."""
    lines: list[str] = [f"<example id=\"{_xml_attr(example_id)}\">", "  <input>", "    <sources>"]

    for src in cleaned_input.get("sources", []):
        src_title = _xml_attr(src.get("title", ""))
        src_desc = _xml_attr(src.get("description", ""))
        src_url = _xml_attr(src.get("url", ""))
        src_type = _xml_attr(src.get("type", ""))
        src_name = _xml_attr(src.get("source_name", ""))
        src_pub = _xml_attr(src.get("published_at", ""))
        src_text = _xml_text(src.get("text", ""))
        lines.append(
            "      "
            f"<source title=\"{src_title}\" description=\"{src_desc}\" "
            f"url=\"{src_url}\" type=\"{src_type}\" source_name=\"{src_name}\" "
            f"published_at=\"{src_pub}\">{src_text}</source>"
        )

    lines.extend(
        [
            "    </sources>",
            f"    <duration>{_xml_text(cleaned_input.get('duration', 0))}</duration>",
            f"    <creator>{_xml_text(cleaned_input.get('creator', ''))}</creator>",
            f"    <topic>{_xml_text(cleaned_input.get('topic', ''))}</topic>",
            f"    <title>{_xml_text(cleaned_input.get('title', ''))}</title>",
            "  </input>",
            "  <output>",
            "    <segments>",
        ]
    )

    for seg in cleaned_output.get("segments", []):
        scene = _xml_text(seg.get("scene", ""))
        transcript = _xml_text(seg.get("transcript", ""))
        lines.extend(
            [
                "      <segment>",
                f"        <start>{_xml_text(seg.get('start', 0.0))}</start>",
                f"        <end>{_xml_text(seg.get('end', 0.0))}</end>",
                f"        <duration>{_xml_text(seg.get('duration', 0.0))}</duration>",
                f"        <scene>{scene}</scene>",
                f"        <transcript>{transcript}</transcript>",
            ]
        )
        if seg.get("ui"):
            lines.append(f"        <ui>{_xml_text(seg['ui'])}</ui>")
        if seg.get("audio"):
            lines.append(f"        <audio>{_xml_text(seg['audio'])}</audio>")
        lines.append("      </segment>")

    lines.extend(["    </segments>", "  </output>", "</example>"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert *_example.json to XMLish training corpus")
    parser.add_argument("--input-dir", type=Path, default=Path("data/training_dataset"))
    parser.add_argument("--output-file", type=Path, default=Path("data/training_dataset_xmlish.txt"))
    parser.add_argument("--max-source-text", type=int, default=600, help="Max chars for source.text")
    parser.add_argument("--include-ui-html", action="store_true", help="Keep non-empty ui_elements_html")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle input file order")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_example.json"))
    if not files:
        raise RuntimeError(f"No *_example.json files found in {args.input_dir}")

    if args.shuffle:
        import random

        random.shuffle(files)

    stats = CleanStats()
    records: list[str] = []

    for path in files:
        stats.files_seen += 1
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            cleaned_input, cleaned_output = clean_example(
                obj,
                stats,
                max_source_text=args.max_source_text,
                include_ui_html=args.include_ui_html,
            )

            # Skip completely empty targets.
            if not cleaned_output.get("segments"):
                continue

            example_id = path.stem.replace("_example", "")
            records.append(to_xmlish_record(example_id, cleaned_input, cleaned_output))
            stats.files_written += 1
        except Exception:
            stats.bad_files += 1

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(records))
        f.write("\n")

    print(f"[OK] Wrote XMLish corpus: {args.output_file}")
    print(f"     Files seen: {stats.files_seen}")
    print(f"     Records written: {stats.files_written}")
    print(f"     Bad files skipped: {stats.bad_files}")
    print(f"     Artifact removals: {stats.artifact_hits}")
    print(f"     Characters trimmed: {stats.chars_trimmed}")
    print("\nCleaning policy:")
    print("- metadata dropped")
    print("- sources kept but text/description trimmed")
    print("- chat/log artifact tags removed (<message>, <datetime>, <sender>, !roast)")
    print("- long fields truncated to stabilize training")


if __name__ == "__main__":
    main()
