#!/usr/bin/env python3
"""
Simplify training examples for long-context model training.

What this does:
- Keeps transcripts unchanged.
- Summarizes source text/content using transcript+scene keywords as anchors.
- Compresses scene descriptions to key nouns/verbs (max chars configurable).
- Replaces HTML-heavy UI fields with compact plain-text UI summaries.

Usage:
  python simplify_training_examples.py
  python simplify_training_examples.py --input-dir data/training_dataset --in-place
  python simplify_training_examples.py --input-dir data/training_dataset --output-dir data/training_dataset_simplified
    python simplify_training_examples.py --xmlish-file data/training_dataset_xmlish.txt --xmlish-in-place
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "their", "they", "them", "you", "your", "our", "we",
    "or", "if", "but", "so", "than", "then", "there", "here", "into", "about", "after",
    "before", "while", "through", "during", "also", "just", "very", "more", "most", "such",
    "what", "when", "where", "who", "whom", "why", "how", "can", "could", "should", "would",
    "do", "does", "did", "done", "have", "had", "having", "been", "being", "up", "down",
    "over", "under", "out", "off", "again", "further", "once", "all", "any", "both", "each",
    "few", "other", "some", "no", "nor", "not", "only", "own", "same", "too", "s", "t",
    # Simplifier artifact tokens to avoid recursive degradation on re-runs.
    "nouns", "verbs", "text", "ui", "cues",
}

VERB_HINTS = {
    "say", "says", "said", "talk", "talks", "talking", "speak", "speaks", "speaking",
    "show", "shows", "showing", "open", "opens", "opening", "close", "closes", "closing",
    "look", "looks", "looking", "move", "moves", "moving", "stand", "stands", "standing",
    "sit", "sits", "sitting", "walk", "walks", "walking", "read", "reads", "reading",
    "write", "writes", "writing", "type", "types", "typing", "scroll", "scrolls", "scrolling",
    "point", "points", "pointing", "smile", "smiles", "smiling", "laugh", "laughs", "laughing",
    "ask", "asks", "asking", "answer", "answers", "answering", "play", "plays", "playing",
}


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    # Remove prior simplifier labels when re-processing already simplified data.
    text = re.sub(r"\b(?:nouns|verbs|text|ui\s+cues)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']{1,}", text.lower())


def extract_keywords_from_example(example: dict[str, Any], max_keywords: int = 16) -> list[str]:
    # Prefer transcript keywords (what we most want preserved), then scene fallback.
    transcript_bag: list[str] = []
    scene_bag: list[str] = []
    output = example.get("output", {})
    segments = output.get("segments", []) if isinstance(output, dict) else []

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        transcript_bag.extend(tokenize(str(seg.get("transcript", ""))))
        scene_bag.extend(tokenize(str(seg.get("scene_description", ""))))

    t_counts = Counter(w for w in transcript_bag if w not in STOPWORDS and len(w) >= 4)
    s_counts = Counter(w for w in scene_bag if w not in STOPWORDS and len(w) >= 4)

    ordered: list[str] = []
    for w, _ in t_counts.most_common(max_keywords):
        if w not in ordered:
            ordered.append(w)
    for w, _ in s_counts.most_common(max_keywords):
        if w not in ordered:
            ordered.append(w)
    return ordered[:max_keywords]


def split_sentences(text: str) -> list[str]:
    rough = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [clean_text(s) for s in rough if clean_text(s)]


def summarize_with_keywords(text: str, keywords: list[str], max_chars: int = 320) -> str:
    text = clean_text(text)
    if not text:
        return ""

    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars]

    scored: list[tuple[int, int, str]] = []
    for i, s in enumerate(sentences):
        words = set(tokenize(s))
        overlap = sum(1 for k in keywords if k in words)
        scored.append((overlap, -i, s))

    scored.sort(reverse=True)

    picked: list[str] = []
    used = 0
    for overlap, _, sentence in scored:
        if used >= max_chars:
            break
        # Favor keyword-containing sentences; allow first sentence fallback.
        if overlap <= 0 and picked:
            continue
        add_len = len(sentence) + (1 if picked else 0)
        if used + add_len > max_chars:
            continue
        picked.append(sentence)
        used += add_len

    if not picked:
        picked = [sentences[0][:max_chars]]

    summary = " ".join(picked).strip()

    # Ensure key terms appear when available.
    present = set(tokenize(summary))
    missing = [k for k in keywords if k not in present and k in set(tokenize(text))][:4]
    if missing:
        tail = " Keywords: " + ", ".join(missing)
        if len(summary) + len(tail) <= max_chars:
            summary += tail

    return summary[:max_chars].strip()


def extract_nouns_verbs(text: str) -> tuple[list[str], list[str]]:
    text = clean_text(text)
    words = [w for w in tokenize(text) if w not in STOPWORDS]
    nouns: list[str] = []
    verbs: list[str] = []

    for w in words:
        is_verb = (
            w in VERB_HINTS
            or w.endswith("ing")
            or w.endswith("ed")
            or w.endswith("ize")
            or w.endswith("ise")
        )
        if is_verb:
            if w not in verbs:
                verbs.append(w)
        else:
            if w not in nouns:
                nouns.append(w)

    return nouns[:12], verbs[:12]


def compact_scene(scene: str, max_chars: int = 200) -> str:
    scene = clean_text(scene)
    if not scene:
        return ""

    nouns, verbs = extract_nouns_verbs(scene)
    parts: list[str] = []
    if nouns:
        parts.append(", ".join(nouns[:8]))
    if verbs:
        parts.append(", ".join(verbs[:8]))

    compact = "; ".join(parts).strip()
    if not compact:
        compact = scene[:max_chars]

    return compact[:max_chars].strip(" ,;")


def extract_ui_text_from_html(ui_html: str) -> list[str]:
    raw = html.unescape(ui_html or "")
    # Capture aria labels and visible text, then strip tags.
    attrs = re.findall(r"aria-label=[\"']([^\"']+)[\"']", raw, flags=re.IGNORECASE)
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    chunks = split_sentences(no_tags)

    bits = attrs + chunks
    dedup: list[str] = []
    seen = set()
    for b in bits:
        c = clean_text(b)
        if not c:
            continue
        # Skip CSS/HTML debris and very long fragments.
        if any(x in c for x in ("{", "}", "<", ">", "DOCTYPE", "viewport", "charset")):
            continue
        if len(c) > 60:
            continue
        if len(c.split()) > 8:
            continue
        low = c.lower()
        if low in seen:
            continue
        seen.add(low)
        dedup.append(c)
    return dedup


def extract_on_screen_text_from_scene(scene: str) -> list[str]:
    # Text often appears in quotes in scene descriptions.
    quotes = re.findall(r"[\"'“”]([^\"'“”]{2,80})[\"'“”]", scene or "")
    cleaned: list[str] = []
    for q in quotes:
        c = clean_text(q)
        if not c:
            continue
        if len(c) > 60:
            continue
        if len(c.split()) > 8:
            continue
        cleaned.append(c)
    return cleaned


def compact_ui(scene: str, ui_html: str, max_chars: int = 200) -> str:
    scene_text = extract_on_screen_text_from_scene(scene)
    html_text = extract_ui_text_from_html(ui_html)
    nouns, verbs = extract_nouns_verbs(scene)

    parts: list[str] = []
    if scene_text or html_text:
        visible = (scene_text + html_text)[:8]
        parts.append(" | ".join(visible))

    nv = []
    if nouns:
        nv.extend(nouns[:5])
    if verbs:
        nv.extend(verbs[:5])
    if nv:
        parts.append(", ".join(nv))

    out = "; ".join(parts).strip()
    return out[:max_chars].strip(" ,;")


def simplify_example(
    example: dict[str, Any],
    source_max_chars: int,
    scene_max_chars: int,
    ui_max_chars: int,
) -> dict[str, Any]:
    keywords = extract_keywords_from_example(example)

    # Simplify source text while preserving metadata.
    input_block = example.get("input", {})
    sources = input_block.get("sources", []) if isinstance(input_block, dict) else []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            candidate = "\n".join(
                str(source.get(k, "")) for k in ("text", "content", "description") if source.get(k)
            )
            summary = summarize_with_keywords(candidate, keywords, max_chars=source_max_chars)
            if summary:
                source["text"] = summary
                # Keep content lean and consistent.
                source["content"] = summary

    # Simplify segment scene/UI fields, keep transcript as-is.
    output_block = example.get("output", {})
    segments = output_block.get("segments", []) if isinstance(output_block, dict) else []
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            scene = str(seg.get("scene_description", ""))
            ui_html = str(seg.get("ui_elements_html", ""))

            seg["scene_description"] = compact_scene(scene, max_chars=scene_max_chars)
            seg["ui_elements"] = compact_ui(scene, ui_html, max_chars=ui_max_chars)
            # Remove/blank heavy HTML field as requested.
            seg["ui_elements_html"] = ""

    return example


def process_file(
    src_path: Path,
    dst_path: Path,
    source_max_chars: int,
    scene_max_chars: int,
    ui_max_chars: int,
) -> tuple[bool, str]:
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False, "not a JSON object"

        data = simplify_example(
            data,
            source_max_chars=source_max_chars,
            scene_max_chars=scene_max_chars,
            ui_max_chars=ui_max_chars,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def extract_keywords_from_xml_root(root: ET.Element, max_keywords: int = 16) -> list[str]:
    transcript_bag: list[str] = []
    scene_bag: list[str] = []

    for seg in root.findall("./output/segments/segment"):
        transcript_bag.extend(tokenize(clean_text(seg.findtext("transcript", default=""))))
        scene_bag.extend(tokenize(clean_text(seg.findtext("scene", default=""))))

    t_counts = Counter(w for w in transcript_bag if w not in STOPWORDS and len(w) >= 4)
    s_counts = Counter(w for w in scene_bag if w not in STOPWORDS and len(w) >= 4)

    ordered: list[str] = []
    for w, _ in t_counts.most_common(max_keywords):
        if w not in ordered:
            ordered.append(w)
    for w, _ in s_counts.most_common(max_keywords):
        if w not in ordered:
            ordered.append(w)
    return ordered[:max_keywords]


def simplify_xmlish_file(
    src_path: Path,
    dst_path: Path,
    source_max_chars: int,
    scene_max_chars: int,
    ui_max_chars: int,
) -> tuple[bool, str]:
    try:
        text = src_path.read_text(encoding="utf-8")
        blocks = re.findall(r"<example\b[\s\S]*?</example>", text)
        if not blocks:
            return False, "no <example> blocks found"

        out_blocks: list[str] = []
        parse_failures = 0

        for block in blocks:
            try:
                root = ET.fromstring(block)
            except Exception:
                parse_failures += 1
                out_blocks.append(block.strip())
                continue

            keywords = extract_keywords_from_xml_root(root)

            # Simplify source body text
            for source in root.findall("./input/sources/source"):
                candidate = "\n".join(
                    x for x in [
                        source.get("description", ""),
                        source.get("text", ""),
                        source.text or "",
                    ] if x
                )
                summary = summarize_with_keywords(candidate, keywords, max_chars=source_max_chars)
                if summary:
                    source.text = summary

            # Simplify segment scene + UI fields while preserving transcript
            for seg in root.findall("./output/segments/segment"):
                scene_el = seg.find("scene")
                ui_html_el = seg.find("ui_elements_html")

                scene_text = clean_text(scene_el.text if scene_el is not None else "")
                ui_html_text = clean_text(ui_html_el.text if ui_html_el is not None else "")

                if scene_el is not None and scene_text:
                    scene_el.text = compact_scene(scene_text, max_chars=scene_max_chars)

                ui_summary = compact_ui(scene_text, ui_html_text, max_chars=ui_max_chars)
                if ui_summary:
                    ui_el = seg.find("ui_elements")
                    if ui_el is None:
                        ui_el = ET.SubElement(seg, "ui_elements")
                    ui_el.text = ui_summary

                if ui_html_el is not None:
                    ui_html_el.text = ""

            out_blocks.append(ET.tostring(root, encoding="unicode"))

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text("\n\n".join(out_blocks) + "\n", encoding="utf-8")
        return True, f"ok (examples={len(blocks)}, parse_failures={parse_failures})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplify training examples (JSON folder or XML-ish txt file)")
    parser.add_argument("--input-dir", type=Path, default=Path("data/training_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training_dataset_simplified"))
    parser.add_argument("--in-place", action="store_true", help="Overwrite files in input dir")
    parser.add_argument("--glob", type=str, default="*_example.json")
    parser.add_argument("--xmlish-file", type=Path, default=None, help="Path to XML-ish training txt file")
    parser.add_argument("--xmlish-output", type=Path, default=Path("data/training_dataset_xmlish_simplified.txt"), help="Output path for simplified XML-ish txt")
    parser.add_argument("--xmlish-in-place", action="store_true", help="Overwrite --xmlish-file")
    parser.add_argument("--source-max-chars", type=int, default=320)
    parser.add_argument("--scene-max-chars", type=int, default=200)
    parser.add_argument("--ui-max-chars", type=int, default=200)
    args = parser.parse_args()

    # XML-ish mode
    if args.xmlish_file is not None:
        if not args.xmlish_file.exists():
            raise SystemExit(f"XML-ish input file not found: {args.xmlish_file}")
        dst = args.xmlish_file if args.xmlish_in_place else args.xmlish_output
        ok, msg = simplify_xmlish_file(
            args.xmlish_file,
            dst,
            source_max_chars=args.source_max_chars,
            scene_max_chars=args.scene_max_chars,
            ui_max_chars=args.ui_max_chars,
        )
        if not ok:
            raise SystemExit(f"[FAIL] {msg}")
        print(f"Done. {msg}")
        print(f"Output: {dst}")
        return

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    paths = sorted(args.input_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched {args.glob} in {args.input_dir}")

    ok_count = 0
    fail_count = 0

    for src in paths:
        dst = src if args.in_place else (args.output_dir / src.name)
        ok, msg = process_file(
            src,
            dst,
            source_max_chars=args.source_max_chars,
            scene_max_chars=args.scene_max_chars,
            ui_max_chars=args.ui_max_chars,
        )
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            print(f"[FAIL] {src.name}: {msg}")

    print(f"Done. processed={len(paths)} ok={ok_count} failed={fail_count}")
    print(f"Output: {args.input_dir if args.in_place else args.output_dir}")


if __name__ == "__main__":
    main()
