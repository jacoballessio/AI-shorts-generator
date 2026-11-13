from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None

TARGET_TAGS = {"scene", "transcript", "audio", "ui", "title", "topic", "creator"}
TIMING_TAGS = {"start", "end", "duration"}
STRUCTURAL_TAGS = {"segment", "segments", "input", "output", "sources"}
NORMALIZABLE_TAGS = TARGET_TAGS | TIMING_TAGS | STRUCTURAL_TAGS

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

COMMON_WORDS = {
    "a", "about", "add", "again", "all", "alone", "along", "always", "an", "and", "any", "are",
    "around", "as", "at", "attention", "available", "away", "background", "back", "be", "because",
    "black", "blue", "both", "bright", "but", "by", "calm", "camera", "care", "cast", "clear",
    "close", "cozy", "concrete", "continue", "creator", "day", "directly", "dog", "don't", "doors",
    "draw", "each", "else", "end", "environment", "exercise", "expression", "face", "features", "final",
    "first", "focus", "follow", "for", "from", "gently", "get", "green", "gesturing", "go", "going",
    "good", "grass", "hand", "happy", "has", "have", "he", "head", "her", "here", "holds", "how",
    "i", "if", "in", "inside", "is", "it", "its", "keep", "keeping", "know", "likely", "lined",
    "long", "looks", "make", "market", "medium", "more", "mood", "name", "natural", "near", "new",
    "now", "of", "offering", "on", "one", "opens", "or", "our", "out", "outdoors", "over", "path",
    "pattern", "people", "phone", "plain", "positive", "quick", "remains", "remember", "responsibility",
    "right", "safe", "scene", "segment", "she", "shot", "show", "side", "sky", "small", "smile",
    "so", "soft", "some", "standing", "starts", "stay", "street", "style", "sure", "take", "talks",
    "text", "that", "the", "them", "there", "they", "this", "to", "today", "topic", "transcript",
    "treat", "treats", "under", "up", "used", "video", "view", "walk", "walking", "warm", "watch",
    "weather", "wearing", "with", "word", "wrapping", "you", "your", "zoom", "carmind",
}

CUSTOM_WORDS: set[str] = set()
PAIRING_HINT_WORDS: set[str] = set()

# Common correct multi-word phrases that should be strongly preferred
COMMON_PHRASES: set[str] = {
    "park or garden", "camera remains", "urban environment", "outdoor scene",
    "clear sky", "bright sunlight", "green outdoor", "park setting",
    "camera stays", "camera keeps", "natural daylight", "concrete sidewalk",
    "to a", "wearing a", "on a", "in a", "in a frame", "graphic tee", "intense emotions", "together on a", "mouth and eyes", "a man's", "clear skies", "blue sky", "checkered shirt", "checkered pattern", "holding a", "holding the", "looking at the camera", "i'm a different", "he has an", "on a sidewalk", "on the sidewalk", "over"
}

# Anti-phrases: fragmented patterns that should never appear.
# Maps fragmented form -> intended correct form for penalty application.
ANTI_PHRASES: dict[str, str] = {
    "deme a nor": "demeanor",
    "fig ures": "figures",
    "in tense": "intense",
    "sp read": "spread",
    "e mph astically": "emphatically",
    "emp has is": "emphasis",
    "emp has i zing": "emphasizing",
    "speak ing e": "speaking",
    "at mos phere": "atmosphere",
    "atmo sphere": "atmosphere",
    "seri ouse x": "serious",
    "ex press ive": "expressive",
    "exp ress ive": "expressive",
    "del ivery": "delivery",
    "cons istent": "consistent",
    "fram ke sh": "frame",
    "youth ink": "you think",
    "na tur ally": "naturally",
    "rem mains": "remains",
    "sh adows": "shadows",
    "sh aky": "shaky",
    "spe aking": "speaking",
    "speak in g": "speaking",
    "cartoon ish": "cartoonish",
    "car to on is h": "cartoonish",
    "pan ick edt one": "panicked tone",
    "p an ickedt one": "panicked tone",
    "post ure": "posture",
    "in ten see mot ions": "intense emotions",
    "sina frame": "in a frame",
    "ina frame": "in a frame",
    "gra phi cte e": "graphic tee",
    "suggest in gout": "suggesting out",
    "doors cen es": "door scenes",
    "s cen es": "scenes",
    "an dey es": "and eyes",
    "ups elie": "up selfie",
    "ups elfie": "up selfie",
    "aman's": "a man's",
    "ona": "on a",
    "ina": "in a",
    "concretes ide walk": "concrete sidewalk",
    "concrete ide walk": "concrete sidewalk",
    "i'mad if fer ent": "i'm a different",
    "i ' mad if fer ent": "i'm a different",
    "I'm ad ifferent": "i'm a different",
    "appear so ver": "appears over",
}

_LEXICON_CACHE: set[str] | None = None
WORD_FREQUENCIES: dict[str, int] = {}
_WORD_VALUE_CACHE: dict[str, float] = {}
_SEGMENT_CANDIDATE_CACHE: dict[str, bool] = {}

# Performance guardrails for badly fragmented long text.
MAX_REPAIR_WINDOWS = 1200
MAX_COMPACT_WINDOW_LEN = 60
MAX_RUN_REPAIRS = 600
MAX_RUN_TOKENS = 10
TOKEN_SPLIT_PENALTY = 0.6
UNKNOWN_ALPHA_PENALTY = 1.2

COMPOUND_HINTS = {
    "selfie", "sidewalk", "beside", "forward", "outdoor", "background",
    "concrete", "checkered", "graphic", "mouth", "eyes", "camera",
}
ALLOWED_SHORT_WORDS = {
    "a", "i", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
}


def _accept_dictionary_token(token: str) -> bool:
    if not re.fullmatch(r"[a-z']{2,40}", token):
        return False
    if len(token) <= 2 and token not in ALLOWED_SHORT_WORDS:
        return False
    return True


def _invalidate_lexicon_cache() -> None:
    global _LEXICON_CACHE
    _LEXICON_CACHE = None
    _WORD_VALUE_CACHE.clear()
    _SEGMENT_CANDIDATE_CACHE.clear()


def _load_dictionary_words(path: Path) -> list[str]:
    """Load dictionary words from plain text (one per line) or CSV (first column)."""
    words: list[str] = []

    if path.suffix.lower() == ".csv":
        # Some dictionary CSV files contain very large definition fields.
        # Raise parser limit so we can still read the first column (word).
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                break
            except OverflowError:
                limit = limit // 10

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row:
                    continue
                token = row[0].strip().lower()
                if first:
                    first = False
                    if token in {"word", "token", "term"}:
                        continue
                if not token or token.startswith("#"):
                    continue
                if _accept_dictionary_token(token):
                    words.append(token)
        return words

    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if not token or token.startswith("#"):
            continue
        if _accept_dictionary_token(token):
            words.append(token)
    return words


def _load_word_frequencies(path: Path) -> dict[str, int]:
    """Load word->count map from a CSV like: Rank,Word,Frequency count,..."""
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 3:
                continue
            if first:
                first = False
                if row[0].strip().lower() == "rank" and row[1].strip().lower() == "word":
                    continue

            word = row[1].strip().lower()
            if not _accept_dictionary_token(word):
                continue

            count_raw = row[2].strip().replace(",", "")
            try:
                count = int(count_raw)
            except ValueError:
                continue

            if count > 0:
                out[word] = count
    return out


def _load_common_phrases(path: Path) -> set[str]:
    """Load common correct phrases from file (one per line)."""
    phrases: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        phrase = line.strip().lower()
        if phrase and not phrase.startswith("#"):
            phrases.add(phrase)
    return phrases


def _load_anti_phrases(path: Path) -> dict[str, str]:
    """Load anti-phrases from file: fragmented_form<tab or comma>correct_form.
    
    Format can be:
    - Tab-separated: "deme a nor	demeanor"
    - Comma-separated: "deme a nor,demeanor"
    - Format auto-detected from first non-header line
    """
    anti_phrases: dict[str, str] = {}
    delimiter = None
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Auto-detect delimiter
            if delimiter is None:
                if "\t" in line:
                    delimiter = "\t"
                elif "," in line:
                    delimiter = ","
                else:
                    continue  # Skip lines without delimiters
            
            parts = line.split(delimiter, 1)
            if len(parts) == 2:
                fragmented = parts[0].strip().lower()
                correct = parts[1].strip().lower()
                if fragmented and correct:
                    anti_phrases[fragmented] = correct
    
    return anti_phrases


def _rare_score_threshold() -> float:
    # Frequency-table scores are log10(count+1)+1, so ~4.5 means "not common".
    if WORD_FREQUENCIES:
        return 4.5
    # Zipf-only fallback threshold.
    return 3.5

TOKEN_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")
TAG_TEXT_RE = re.compile(
    r"(<(?P<tag>scene|transcript|audio|ui|title|topic|creator)\b[^>]*>)(?P<body>.*?)(</(?P=tag)\s*>)",
    re.IGNORECASE | re.DOTALL,
)
ALLOWED_XMLISH_TAGS = NORMALIZABLE_TAGS | {"example", "input", "output", "segments", "source", "sources"}


def _strip_unknown_xmlish_tags(text: str) -> str:
    """Drop unrecognized tags while preserving enclosed text content.

    XML parsers cannot ignore malformed or unknown tag names once they are part
    of the token stream. Removing those tags pre-parse is the permissive mode.
    """
    def _repl(match: re.Match[str]) -> str:
        raw_name = match.group("name") or ""
        normalized = re.sub(r"\s+", "", raw_name).lower()
        if normalized in ALLOWED_XMLISH_TAGS:
            slash = "/" if match.group("slash") else ""
            return f"<{slash}{normalized}>"
        return ""

    return re.sub(
        r"<\s*(?P<slash>/?)\s*(?P<name>[A-Za-z_][A-Za-z0-9_\s-]*)\s*[^>]*>",
        _repl,
        text,
        flags=re.IGNORECASE,
    )


def _normalize_broken_target_tags(text: str) -> str:
    """Normalize malformed target tags like '< trans cript }' -> '<transcript>'."""
    normalized = text

    # Fast direct fixes for observed artifacts.
    normalized = normalized.replace("< trans cript }", "<transcript>")
    normalized = normalized.replace("< trans cript ]", "<transcript>")
    normalized = re.sub(r"<\s*/?\s*section\s*>", lambda m: "</segment>" if "/" in m.group(0) else "<segment>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*starts\s*>", "<start>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*starts\s*>", "</start>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*ends\s*>", "<end>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*ends\s*>", "</end>", normalized, flags=re.IGNORECASE)
    # Generated outputs often drift into near-transcript tag names such as
    # <transcription>, <transscript>, <transition>, etc. Normalize any
    # trans* tag (except the canonical transcript tag handled later) to
    # transcript so mismatched close tags do not break parsing.
    normalized = re.sub(
        r"<\s*(/?)\s*trans(?!\s*c\s*r\s*i\s*p\s*t\b)([A-Za-z_]*)\s*>",
        lambda m: f"<{ '/' if m.group(1) else '' }transcript>".replace("< /", "</"),
        normalized,
        flags=re.IGNORECASE,
    )

    # Remove stray quote characters that leak into timing bodies right before a
    # closing tag, e.g. <duration> 5.0 "</duration>.
    normalized = re.sub(
        r"([0-9.])\s*[\"']\s*(</(?:start|end|duration)>)",
        r"\1\2",
        normalized,
        flags=re.IGNORECASE,
    )

    for tag in NORMALIZABLE_TAGS:
        # Convert self-closing malformed tags to explicit empty elements.
        normalized = re.sub(
            rf"<\s*{tag}\s*/\s*>",
            f"<{tag}></{tag}>",
            normalized,
            flags=re.IGNORECASE,
        )

    for tag in NORMALIZABLE_TAGS:
        spaced = r"\s*".join(re.escape(ch) for ch in tag)

        # Normalize malformed opening tags, including missing '>' cases.
        open_pattern = re.compile(
            r"<\s*" + spaced + r"(?![A-Za-z0-9_])\s*[\}\]\)\"']?\s*(?:>|(?=\s|$))",
            flags=re.IGNORECASE,
        )
        normalized = open_pattern.sub(f"<{tag}>", normalized)

        # Normalize malformed closing tags.
        close_pattern = re.compile(
            r"<\s*/\s*" + spaced + r"(?![A-Za-z0-9_])\s*[\}\]\)\"']?\s*(?:>|(?=\s|$))",
            flags=re.IGNORECASE,
        )
        normalized = close_pattern.sub(f"</{tag}>", normalized)

        # Cleanup accidental residues from malformed self-closing variants.
        normalized = re.sub(rf"<{tag}>\s*/\s*>", f"<{tag}></{tag}>", normalized, flags=re.IGNORECASE)

    # `example` can carry attributes, so only normalize malformed closing tags.
    normalized = re.sub(
        r"<\s*/\s*e\s*x\s*a\s*m\s*p\s*l\s*e\s*[\}\]\)\"']?\s*(?:>|(?=\s|$))",
        "</example>",
        normalized,
        flags=re.IGNORECASE,
    )

    # Common malformed nesting from generations: close the child tag first,
    # then close the segment. Also collapse duplicate premature </segment> tags.
    for tag in TARGET_TAGS | TIMING_TAGS:
        normalized = re.sub(
            rf"(?:</segment>\s*)+</{tag}>",
            f"</{tag}></segment>",
            normalized,
            flags=re.IGNORECASE,
        )

    # Preserve empty transcript tags when generations emit a malformed
    # self-closing/opening transcript directly before a sibling tag.
    normalized = re.sub(
        r"<transcript>\s*(?=(?:<audio>|</segment>|</output>|</example>))",
        "<transcript></transcript>",
        normalized,
        flags=re.IGNORECASE,
    )

    # If segments are placed directly under <output>, restore the expected
    # container wrapper so closing </segments> remains structurally valid.
    if re.search(r"<\s*output\s*>\s*<\s*segment\b", normalized, flags=re.IGNORECASE):
        normalized = re.sub(
            r"(<\s*output\s*>\s*)(<\s*segment\b)",
            r"\1<segments>\n\2",
            normalized,
            count=1,
            flags=re.IGNORECASE,
        )

    normalized = _strip_unknown_xmlish_tags(normalized)

    return normalized


def _flatten_segments_wrappers(text: str) -> str:
    """Keep only one </segments> wrapper when generations emit nested duplicates, remove all <segments>"""
    # Remove all <segments> tags.
    flattened = re.sub(r"<\s*segments\s*>", "", text, flags=re.IGNORECASE)
    # Collapse multiple </segments> into one.
    flattened = re.sub(r"(?:</segments>\s*)+</segments>", "</segments>", flattened, flags=re.IGNORECASE)
    return flattened

def _ensure_target_tags_closed(text: str) -> str:
    """Ensure target tags are balanced by appending missing closing tags."""
    balanced = text
    trailer_match = re.search(r"(\s*</(?:segments|output|example)>\s*)+$", balanced, flags=re.IGNORECASE)
    trailer = ""
    if trailer_match:
        trailer = trailer_match.group(0)
        balanced = balanced[: trailer_match.start()]

    for tag in TARGET_TAGS | {"segment"}:
        open_count = len(re.findall(rf"<\s*{tag}(?:\s+[^>]*)?>", balanced, flags=re.IGNORECASE))
        close_count = len(re.findall(rf"<\s*/\s*{tag}\s*>", balanced, flags=re.IGNORECASE))
        if open_count > close_count:
            balanced += "".join(f"</{tag}>" for _ in range(open_count - close_count))
    return balanced + trailer


def _strip_invalid_xml_chars(text: str) -> str:
    """Remove control chars that are illegal in XML 1.0."""
    return "".join(
        ch for ch in text
        if ch in "\t\n\r" or ord(ch) >= 0x20
    )


def _format_xmlish_output(xmlish_text: str) -> str:
    """Pretty-format XML-ish text with tabs and newlines, tolerant of fragments."""
    text = re.sub(r">\s*<", ">\n<", xmlish_text.strip())
    tokens = re.findall(r"<[^>]+>|[^<]+", text)

    lines: list[str] = []
    indent = 0

    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("<") and tok.endswith(">"):
            tag = tok.strip()
            is_close = bool(re.match(r"</", tag))
            is_self = tag.endswith("/>") or tag.startswith("<?") or tag.startswith("<!")

            if is_close:
                indent = max(0, indent - 1)
            lines.append("\t" * indent + tag)
            if not is_close and not is_self:
                indent += 1
            continue

        txt = " ".join(tok.split())
        if txt:
            lines.append("\t" * indent + txt)

    return "\n".join(lines).strip() + "\n"


def _normalize_number_words(text: str) -> str:
    return re.sub(
        r"\b(" + "|".join(NUMBER_WORDS.keys()) + r")\b",
        lambda match: NUMBER_WORDS[match.group(1).lower()],
        text,
        flags=re.IGNORECASE,
    )


def _extract_numeric_tag_value(raw_text: str, tag_name: str) -> str | None:
    normalized = _normalize_number_words(raw_text.strip())
    normalized = normalized.replace(",", " ")

    raw_candidates = re.findall(r"\d+(?:\.\d+)?", normalized)
    has_spaced_numbers = bool(re.search(r"\d+(?:\.\d+)?\s+\d+(?:\.\d+)?", normalized))
    if tag_name in {"start", "end"} and raw_candidates and has_spaced_numbers:
        return raw_candidates[0]

    compact = re.sub(r"(?<=\d)\s+(?=[\d.])", "", normalized)
    compact = re.sub(r"(?<=\.)\s+(?=\d)", "", compact)
    compact_candidates = re.findall(r"\d+(?:\.\d+)?", compact)
    if compact_candidates:
        return compact_candidates[0]

    return raw_candidates[0] if raw_candidates else None


def _format_numeric_value(value: float) -> str:
    formatted = f"{value:.12f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted += ".0"
    return formatted


def _replace_tag_body(block: str, tag_name: str, new_value: str) -> str:
    pattern = re.compile(rf"(<{tag_name}>\s*)(.*?)(\s*</{tag_name}>)", re.IGNORECASE | re.DOTALL)
    return pattern.sub(lambda match: f"{match.group(1)}{new_value}{match.group(3)}", block, count=1)


def _collect_tag_bodies(block: str, tag_name: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(
            rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]


def _collapse_segment_tag(block: str, tag_name: str, replacement_body: str | None) -> str:
    pattern = re.compile(
        rf"\s*<{tag_name}>\s*.*?\s*</{tag_name}>\s*",
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = pattern.sub("", block)
    if replacement_body is None:
        return stripped

    segment_close = re.search(r"\s*</segment>\s*$", stripped, flags=re.IGNORECASE)
    insertion = f"\n		<{tag_name}>\n			{replacement_body}\n		</{tag_name}>"
    if segment_close:
        return stripped[: segment_close.start()] + insertion + stripped[segment_close.start():]
    return stripped + insertion


def _normalize_segment_metadata(block: str) -> str:
    values: dict[str, str | None] = {}
    for tag_name in TIMING_TAGS:
        match = re.search(rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>", block, flags=re.IGNORECASE | re.DOTALL)
        values[tag_name] = _extract_numeric_tag_value(match.group(1), tag_name) if match else None

    start_value: float | None = None
    duration_value: float | None = None
    end_value: float | None = None

    try:
        start_value = float(values["start"]) if values["start"] is not None else None
    except ValueError:
        start_value = None

    try:
        duration_value = float(values["duration"]) if values["duration"] is not None else None
    except ValueError:
        duration_value = None

    try:
        end_value = float(values["end"]) if values["end"] is not None else None
    except ValueError:
        end_value = None

    if end_value is None and start_value is not None and duration_value is not None:
        values["end"] = _format_numeric_value(start_value + duration_value)

    out = block
    for tag_name in ("start", "end", "duration"):
        if values[tag_name] is not None:
            out = _collapse_segment_tag(out, tag_name, values[tag_name])

    # Ensure at most one content tag per segment. Preserve information by
    # concatenating duplicate bodies instead of dropping them outright.
    for tag_name in ("scene", "transcript", "audio", "ui"):
        bodies = [body for body in _collect_tag_bodies(out, tag_name) if body]
        if not bodies:
            out = _collapse_segment_tag(out, tag_name, None)
            continue

        if tag_name == "ui":
            merged_body = "\n".join(bodies)
        else:
            merged_body = " ".join(body.strip() for body in bodies if body.strip())
        out = _collapse_segment_tag(out, tag_name, merged_body)
    return out


def _normalize_timing_metadata(text: str) -> str:
    normalized = _normalize_broken_target_tags(text)
    return re.sub(
        r"<segment>.*?</segment>",
        lambda match: _normalize_segment_metadata(match.group(0)),
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _dictionary_lexicon() -> set[str]:
    global _LEXICON_CACHE
    if _LEXICON_CACHE is None:
        _LEXICON_CACHE = (
            set(COMMON_WORDS)
            | set(CUSTOM_WORDS)
            | set(WORD_FREQUENCIES.keys())
            | {"a", "i"}
        )
    return _LEXICON_CACHE


def _has_pairing_hint(word: str) -> bool:
    return word.lower() in PAIRING_HINT_WORDS


def _pairing_hint_bonus(word: str) -> float:
    low = word.lower()
    if low not in PAIRING_HINT_WORDS:
        return 0.0
    if len(low) >= 8:
        return 1.15
    if len(low) >= 5:
        return 0.8
    if len(low) >= 4:
        return 0.45
    return 0.0


def _is_anti_phrase(tokens: list[str]) -> bool:
    """Check if token sequence matches a known anti-phrase (fragmented pattern)."""
    phrase = " ".join(tok.lower() for tok in tokens)
    return phrase in ANTI_PHRASES


def _has_common_phrase(tokens: list[str]) -> bool:
    """Check if token sequence is a known common correct phrase."""
    phrase = " ".join(tok.lower() for tok in tokens)
    return phrase in COMMON_PHRASES


def _anti_phrase_penalty() -> float:
    """Penalty magnitude for detected anti-phrase in output."""
    return 50.0  # Very strong penalty to prevent anti-phrases in output


def _common_phrase_bonus() -> float:
    """Bonus magnitude for detected common phrase in output."""
    return 5.0  # Significant bonus to prefer common phrases


def _is_alpha_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z']+", token))


def _is_punct_token(token: str) -> bool:
    return bool(token) and all((not ch.isalnum()) and ch != "'" for ch in token)


def _is_valid_contraction(token: str) -> bool:
    low = token.lower()
    return bool(
        re.fullmatch(r"[a-z]+'(m|s|d|t|re|ve|ll)", low)
        or re.fullmatch(r"[a-z]+n't", low)
    )


def _word_value(word: str) -> float:
    """DP score for a dictionary match: length + frequency bonus.

    Common/frequent words receive a bonus so they beat rare same-length words
    in the DP.  E.g. 'with'(common,len=4) → value≈6 beats 'witha'(rare,len=5) → value≈5.
    """
    cached = _WORD_VALUE_CACHE.get(word)
    if cached is not None:
        return cached

    # _word_score is defined later in the file but resolved at call time.
    n = len(word)
    bonus = max(0.0, (_word_score(word) - _rare_score_threshold()) * 0.7)  # type: ignore[name-defined]
    value = n + bonus + _pairing_hint_bonus(word)
    # Discourage short non-common fragments (e.g. "cen", "arsa", "ict").
    if word not in COMMON_WORDS:
        if n <= 3:
            value -= 1.2
        elif n == 4:
            value -= 0.8
    _WORD_VALUE_CACHE[word] = value
    return value


def _is_segment_candidate(word: str, lex: set[str]) -> bool:
    """Whether a word should be allowed as a segmentation atom.

    The lexicon can contain many rare/OCR artifacts. Restricting segmentation
    to high-confidence words prevents outputs like "we arsa" and
    "in gout do or sce n es".
    """
    cached = _SEGMENT_CANDIDATE_CACHE.get(word)
    if cached is not None:
        return cached

    if word in {"a", "i"}:
        _SEGMENT_CANDIDATE_CACHE[word] = True
        return True

    # Never allow standalone single-letter fragments beyond a/i.
    if len(word) == 1:
        _SEGMENT_CANDIDATE_CACHE[word] = False
        return False

    in_primary_lex = word in lex
    in_pairing_hints = _has_pairing_hint(word)
    if not in_primary_lex and not in_pairing_hints:
        _SEGMENT_CANDIDATE_CACHE[word] = False
        return False

    if word in COMMON_WORDS:
        _SEGMENT_CANDIDATE_CACHE[word] = True
        return True

    score = _word_score(word)
    if WORD_FREQUENCIES:
        # With explicit frequency table available, be stricter.
        ok = score >= 4.2
        if not ok and in_pairing_hints and len(word) >= 4:
            ok = score >= 3.3
    else:
        ok = score >= 3.0
        if not ok and in_pairing_hints and len(word) >= 4:
            ok = score >= 2.5

    _SEGMENT_CANDIDATE_CACHE[word] = ok
    return ok


def _is_relaxed_merge_candidate(word: str, lex: set[str]) -> bool:
    """Relaxed candidate gate used only for direct merge cleanup passes."""
    if word in COMMON_WORDS:
        return True
    if word in COMPOUND_HINTS:
        return True
    if _has_pairing_hint(word) and len(word) >= 4:
        return True
    if word not in lex:
        return False
    score = _word_score(word)
    if WORD_FREQUENCIES:
        return score >= 2.4
    return score >= 2.2


def _segment_string_by_dictionary(
    compact: str,
    lex: set[str] | None = None,
    max_len: int | None = None,
) -> tuple[list[str], int]:
    """Segment compact string to maximise covered chars, preferring common/frequent words."""
    s_raw = compact
    s = compact.lower()
    n = len(s)
    if n == 0:
        return [], 0

    if lex is None:
        lex = _dictionary_lexicon()
    if max_len is None:
        max_len = max((len(w) for w in lex), default=1)

    # dp[i] = (covered_chars, token_count, score, next_index, token_or_none, is_covered, span)
    # Prefer full coverage first, then fewer words, then higher lexical score.
    dp: list[tuple[int, int, float, int, str | None, bool, int]] = [
        (0, 0, 0.0, i + 1, None, False, 1) for i in range(n + 1)
    ]
    dp[n] = (0, 0, 0.0, n, None, True, 0)

    for i in range(n - 1, -1, -1):
        ch = s_raw[i]
        alpha_penalty = UNKNOWN_ALPHA_PENALTY if ch.isalpha() else 0.0
        best_covered = dp[i + 1][0]
        best_token_count = dp[i + 1][1]
        best_score = dp[i + 1][2] - alpha_penalty
        best_next, best_token, best_is_covered, best_span = i + 1, None, False, 1

        def _is_better(candidate_covered: int, candidate_token_count: int, candidate_score: float, candidate_span: int) -> bool:
            if candidate_covered != best_covered:
                return candidate_covered > best_covered
            if candidate_token_count != best_token_count:
                return candidate_token_count < best_token_count
            if candidate_score != best_score:
                return candidate_score > best_score
            return candidate_span > best_span

        # Punctuation chars are considered fully covered and do not add a word.
        if (not ch.isalnum()) and ch != "'":
            covered = 1 + dp[i + 1][0]
            token_count = dp[i + 1][1]
            score = 1.0 + dp[i + 1][2]
            if _is_better(covered, token_count, score, 1):
                best_covered, best_token_count, best_score = covered, token_count, score
                best_next, best_token, best_is_covered, best_span = i + 1, ch, True, 1

        # Dictionary word matches.
        end_limit = min(n, i + max_len)
        for j in range(i + 1, end_limit + 1):
            cand = s[i:j]
            if _is_segment_candidate(cand, lex):
                if len(cand) == 1 and cand in {"a", "i"}:
                    prev_alpha = i > 0 and s[i - 1].isalpha()
                    next_alpha = j < n and s[j].isalpha()
                    if prev_alpha or next_alpha:
                        continue
                span = j - i
                covered = span + dp[j][0]
                token_count = 1 + dp[j][1]
                score = _word_value(cand) - TOKEN_SPLIT_PENALTY + dp[j][2]
                if _is_better(covered, token_count, score, span):
                    best_covered, best_token_count, best_score = covered, token_count, score
                    best_next, best_token, best_is_covered, best_span = j, cand, True, span

        dp[i] = (best_covered, best_token_count, best_score, best_next, best_token, best_is_covered, best_span)

    out: list[str] = []
    unknown_buf: list[str] = []
    i = 0
    while i < n:
        _, _, _, nxt, token, is_covered, _ = dp[i]
        if is_covered and token is not None:
            if unknown_buf:
                out.append("".join(unknown_buf))
                unknown_buf = []
            out.append(token)
            i = nxt
        else:
            unknown_buf.append(s_raw[i])
            i += 1
    if unknown_buf:
        out.append("".join(unknown_buf))

    return out, dp[0][0]


def _segment_with_single_char_deletion(
    compact: str,
    lex: set[str],
    max_len: int,
) -> tuple[list[str], int] | None:
    """Try re-segmentation after deleting one stray alpha character.

    This recovers OCR insertion errors inside compact runs, such as an extra
    character near a word boundary.
    """
    if len(compact) < 8:
        return None

    best: tuple[int, float, list[str], int] | None = None
    for idx, ch in enumerate(compact):
        if not ch.isalpha():
            continue
        edited = compact[:idx] + compact[idx + 1 :]
        segmented, covered = _segment_string_by_dictionary(edited, lex, max_len)
        if covered != len(edited) or not segmented:
            continue
        token_count = len(segmented)
        score = _sequence_value(segmented, lex) - token_count * TOKEN_SPLIT_PENALTY - 0.35
        candidate = (token_count, -score, segmented, covered)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return None
    return best[2], best[3]


def _repair_nonword_windows(text: str) -> str:
    """Repair non-words by segmenting compact ±2 token windows against dictionary."""
    tokens = text.split()
    if len(tokens) < 3:
        return text

    lex = _dictionary_lexicon()
    max_len = max((len(w) for w in lex), default=1)
    repaired_windows = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()
        # A token is a non-word if it's absent from the lexicon, OR if it's a
        # short token (≤5 chars) that is in the dict but very rarely used — a
        # sign it's a broken merged fragment rather than a real word.
        _in_lex = low in lex or _is_valid_contraction(tok)
        _is_rare = (
            _in_lex
            and low not in COMMON_WORDS
            and len(low) <= 5
            and _word_score(low) < _rare_score_threshold()
        )
        is_nonword = _is_alpha_token(tok) and (not _in_lex or _is_rare)
        if not is_nonword:
            i += 1
            continue

        # Use exactly two surrounding space-separated tokens on each side.
        start = max(0, i - 2)
        end = min(len(tokens) - 1, i + 2)
        window = tokens[start : end + 1]
        compact = "".join(window)
        if len(compact) > MAX_COMPACT_WINDOW_LEN:
            i += 1
            continue
        segmented, covered = _segment_string_by_dictionary(compact, lex, max_len)
        deletion_result = _segment_with_single_char_deletion(compact, lex, max_len)
        if deletion_result is not None:
            deleted_segmented, deleted_covered = deletion_result
            prefer_deleted = (
                not segmented
                or deleted_covered > covered
                or (
                    deleted_covered == covered
                    and len(deleted_segmented) + 1 < len(segmented)
                )
                or (
                    deleted_covered == covered - 1
                    and len(deleted_segmented) + 1 < len(segmented)
                )
            )
            if prefer_deleted:
                segmented, covered = deleted_segmented, deleted_covered
        if not segmented:
            i += 1
            continue

        coverage = covered / max(1, len(compact))
        if coverage < 0.70:
            i += 1
            continue

        existing_covered = 0
        for part in window:
            pl = part.lower()
            # Don't count rare/flagged tokens as existing coverage — they're
            # the very tokens we're trying to fix, so treating them as "covered"
            # would prevent any improvement from being detected.
            _part_in_lex = pl in lex or _is_valid_contraction(part)
            _part_is_rare = (
                _part_in_lex
                and pl not in COMMON_WORDS
                and len(pl) <= 5
                and _word_score(pl) < _rare_score_threshold()
            )
            _part_is_nonword = _is_alpha_token(part) and (not _part_in_lex or _part_is_rare)
            if not _part_is_nonword and _is_alpha_token(part):
                existing_covered += len(part)
            elif _is_punct_token(part):
                existing_covered += len(part)

        if covered <= existing_covered:
            i += 1
            continue

        # Reject segmentation if it contains anti-phrases (fragmented patterns to avoid)
        if _is_anti_phrase(segmented):
            i += 1
            continue

        tokens[start : end + 1] = segmented
        repaired_windows += 1
        if repaired_windows >= MAX_REPAIR_WINDOWS:
            break
        i = max(0, start - 1)

    repaired = " ".join(tokens)
    repaired = _repair_pair_splits(repaired, lex, max_len)
    repaired = _repair_trigram_splits(repaired, lex, max_len)
    repaired = _repair_fragment_runs(repaired, lex, max_len)
    repaired = _merge_split_suffixes(repaired, lex)
    repaired = _merge_letter_token_letter(repaired, lex)
    repaired = _merge_high_confidence_triples(repaired, lex)
    repaired = _merge_high_confidence_pairs(repaired, lex)
    return repaired


def _is_rare_alpha_token(token: str, lex: set[str], threshold: float) -> bool:
    low = token.lower()
    in_lex = low in lex
    return (
        _is_alpha_token(token)
        and (low not in COMMON_WORDS)
        and ((not in_lex) or (_word_score(low) < threshold))
    )


def _sequence_value(parts: list[str], lex: set[str]) -> float:
    """Score a sequence of pieces so we can compare old vs new segmentations."""
    total = 0.0
    for part in parts:
        pl = part.lower()
        if _is_alpha_token(part):
            if len(pl) == 1 and pl not in {"a", "i"}:
                total -= 3.0
                continue
            if pl in lex or _has_pairing_hint(pl):
                total += _word_value(pl)
            else:
                total -= len(part) * 0.5
        elif _is_punct_token(part):
            total += len(part) * 0.2
    return total


def _repair_pair_splits(text: str, lex: set[str], max_len: int) -> str:
    """Repair 2-token split artifacts when a better dictionary segmentation exists.

    Examples this catches: "int othe" -> "in to the", "again sta" -> "against a".
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    i = 0
    fixes = 0
    threshold = _rare_score_threshold()
    while i < len(tokens) - 1:
        a = tokens[i]
        b = tokens[i + 1]
        if not (_is_alpha_token(a) and _is_alpha_token(b)):
            i += 1
            continue

        al = a.lower()
        bl = b.lower()

        a_rare = _is_rare_alpha_token(a, lex, threshold)
        b_rare = _is_rare_alpha_token(b, lex, threshold)
        if not (a_rare or b_rare or len(a) <= 4 or len(b) <= 4):
            i += 1
            continue

        compact = al + bl
        segmented, _ = _segment_string_by_dictionary(compact, lex, max_len)
        if len(segmented) < 2:
            i += 1
            continue
        if [s.lower() for s in segmented] == [al, bl]:
            i += 1
            continue

        old_val = _sequence_value([al, bl], lex)
        new_val = _sequence_value(segmented, lex)
        if new_val <= old_val + 0.6:
            i += 1
            continue

        tokens[i : i + 2] = segmented
        fixes += 1
        if fixes >= MAX_REPAIR_WINDOWS:
            break
        i = max(0, i - 1)

    return " ".join(tokens)


def _repair_trigram_splits(text: str, lex: set[str], max_len: int) -> str:
    """Repair stubborn 3-token artifacts using wider context scoring.

    This captures patterns like:
    - "int othe camera" -> "in to the camera"
    - "again sta clear" -> "against a clear"
    """
    tokens = text.split()
    if len(tokens) < 3:
        return text

    i = 0
    fixes = 0
    threshold = _rare_score_threshold()
    while i < len(tokens) - 2:
        a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
        if not (_is_alpha_token(a) and _is_alpha_token(b) and _is_alpha_token(c)):
            i += 1
            continue

        al, bl, cl = a.lower(), b.lower(), c.lower()
        # Trigger only when at least one token looks rare/non-common.
        any_rare = any(
            _is_rare_alpha_token(t, lex, threshold)
            for t in (a, b, c)
        )
        if not any_rare:
            i += 1
            continue

        compact = al + bl + cl
        if len(compact) > MAX_COMPACT_WINDOW_LEN:
            i += 1
            continue

        segmented, _ = _segment_string_by_dictionary(compact, lex, max_len)
        if len(segmented) < 3:
            i += 1
            continue
        if [s.lower() for s in segmented] == [al, bl, cl]:
            i += 1
            continue

        old_val = _sequence_value([al, bl, cl], lex)
        new_val = _sequence_value(segmented, lex)
        if new_val <= old_val + 0.8:
            i += 1
            continue

        tokens[i : i + 3] = segmented
        fixes += 1
        if fixes >= MAX_REPAIR_WINDOWS:
            break
        i = max(0, i - 1)

    return " ".join(tokens)


def _repair_fragment_runs(text: str, lex: set[str], max_len: int) -> str:
    """General repair for heavily fragmented alpha runs.

    This pass catches patterns where every piece is technically word-like but
    the whole run is implausible, e.g. "in gout do or sce n es" or
    "mad if fer ent".
    """
    tokens = text.split()
    if len(tokens) < 3:
        return text

    threshold = _rare_score_threshold()
    i = 0
    fixes = 0
    while i < len(tokens):
        if not _is_alpha_token(tokens[i]):
            i += 1
            continue

        j = i
        run_chars = 0
        while j < len(tokens) and _is_alpha_token(tokens[j]) and (j - i + 1) <= MAX_RUN_TOKENS:
            run_chars += len(tokens[j])
            if run_chars > MAX_COMPACT_WINDOW_LEN:
                break
            j += 1

        run = tokens[i:j]
        if len(run) < 3:
            i += 1
            continue

        short_count = sum(1 for t in run if len(t) <= 3)
        rare_count = sum(1 for t in run if _is_rare_alpha_token(t, lex, threshold))
        # Only attempt expensive re-segmentation when run looks suspicious.
        if short_count < 2 and rare_count == 0:
            i = j
            continue

        compact = "".join(t.lower() for t in run)
        segmented, covered = _segment_string_by_dictionary(compact, lex, max_len)
        deletion_result = _segment_with_single_char_deletion(compact, lex, max_len)
        if deletion_result is not None:
            deleted_segmented, deleted_covered = deletion_result
            prefer_deleted = (
                not segmented
                or deleted_covered > covered
                or (
                    deleted_covered == covered
                    and len(deleted_segmented) + 1 < len(segmented)
                )
                or (
                    deleted_covered == covered - 1
                    and len(deleted_segmented) + 1 < len(segmented)
                )
            )
            if prefer_deleted:
                segmented, covered = deleted_segmented, deleted_covered
        if not segmented:
            i = j
            continue

        coverage = covered / max(1, len(compact))
        if coverage < 0.75:
            i = j
            continue

        old_val = _sequence_value([t.lower() for t in run], lex) - len(run) * TOKEN_SPLIT_PENALTY
        new_val = _sequence_value(segmented, lex) - len(segmented) * TOKEN_SPLIT_PENALTY
        if deletion_result is not None and segmented == deletion_result[0]:
            new_val += 0.8
        if new_val <= old_val + 1.0:
            i = j
            continue

        # Reject segmentation if it contains anti-phrases (fragmented patterns to avoid)
        if _is_anti_phrase(segmented):
            i = j
            continue

        tokens[i:j] = segmented
        fixes += 1
        if fixes >= MAX_RUN_REPAIRS:
            break
        i = max(0, i - 1)

    return " ".join(tokens)


def _merge_split_suffixes(text: str, lex: set[str]) -> str:
    """Merge common split suffix patterns in a generic way.

    Examples: "suggest in g" -> "suggesting", "scene s" -> "scenes".
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    i = 0
    out: list[str] = []
    while i < len(tokens):
        if i + 3 < len(tokens):
            a, b, c, d = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
            # Generalized OCR split like: c has in g -> chasing
            if (
                _is_alpha_token(a)
                and _is_alpha_token(b)
                and c.lower() == "in"
                and d.lower() == "g"
                and len(a) == 1
            ):
                merged = (a + b + "ing").lower()
                if _is_segment_candidate(merged, lex):
                    out.append(merged)
                    i += 4
                    continue

        if i + 2 < len(tokens):
            a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
            if _is_alpha_token(a) and b.lower() == "in" and c.lower() == "g":
                merged = (a + "ing").lower()
                if _is_segment_candidate(merged, lex):
                    out.append(merged)
                    i += 3
                    continue

        if i + 1 < len(tokens):
            a, b = tokens[i], tokens[i + 1]
            if _is_alpha_token(a) and _is_alpha_token(b):
                bl = b.lower()
                if bl in {"s", "ed", "er", "ly"}:
                    merged = (a + bl).lower()
                    if _is_segment_candidate(merged, lex):
                        out.append(merged)
                        i += 2
                        continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def _merge_letter_token_letter(text: str, lex: set[str]) -> str:
    """Merge patterns like 'm out h' -> 'mouth' when strongly plausible."""
    tokens = text.split()
    if len(tokens) < 3:
        return text

    i = 0
    out: list[str] = []
    while i < len(tokens):
        if i + 2 < len(tokens):
            a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
            if (
                _is_alpha_token(a)
                and _is_alpha_token(b)
                and _is_alpha_token(c)
                and len(a) == 1
                and len(c) == 1
            ):
                merged = (a + b + c).lower()
                if _is_segment_candidate(merged, lex):
                    split_val = _sequence_value([a.lower(), b.lower(), c.lower()], lex) - 3 * TOKEN_SPLIT_PENALTY
                    merged_val = _sequence_value([merged], lex) - TOKEN_SPLIT_PENALTY + 0.8
                    if merged_val > split_val + 0.2:
                        out.append(merged)
                        i += 3
                        continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def _merge_high_confidence_pairs(text: str, lex: set[str]) -> str:
    """Merge adjacent alpha tokens when concatenation is clearly better.

    This handles broad OCR splits like:
    - self ie -> selfie
    - side walk -> sidewalk
    - be side -> beside
    - c an -> can
    - for ward -> forward
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    i = 0
    merges = 0
    while i < len(tokens) - 1:
        a = tokens[i]
        b = tokens[i + 1]
        if not (_is_alpha_token(a) and _is_alpha_token(b)):
            i += 1
            continue

        merged = (a + b).lower()
        candidate = merged
        if not _is_relaxed_merge_candidate(candidate, lex):
            # OCR often injects a trailing 't' in '...ic t' patterns.
            if b.lower().startswith("ict"):
                alt = (a + "ic").lower()
                if _is_relaxed_merge_candidate(alt, lex):
                    candidate = alt
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue

        if candidate in COMPOUND_HINTS and (len(a) <= 3 or len(b) <= 3):
            tokens[i : i + 2] = [candidate]
            merges += 1
            if merges >= MAX_RUN_REPAIRS:
                break
            i = max(0, i - 1)
            continue

        if candidate != merged:
            merged = candidate

        if not _is_relaxed_merge_candidate(merged, lex):
            i += 1
            continue

        split_val = _sequence_value([a.lower(), b.lower()], lex) - 2 * TOKEN_SPLIT_PENALTY
        merged_val = _sequence_value([merged], lex) - TOKEN_SPLIT_PENALTY
        # Encourage merges when one side is very short (common OCR split).
        if len(a) <= 2 or len(b) <= 2:
            merged_val += 0.7

        if merged_val <= split_val + 0.15:
            i += 1
            continue

        tokens[i : i + 2] = [merged]
        merges += 1
        if merges >= MAX_RUN_REPAIRS:
            break
        i = max(0, i - 1)

    return " ".join(tokens)


def _merge_high_confidence_triples(text: str, lex: set[str]) -> str:
    """Merge three-token alpha fragments when concatenation is clearly better.

    Examples: check e red -> checkered, c on crete -> concrete.
    """
    tokens = text.split()
    if len(tokens) < 3:
        return text

    i = 0
    merges = 0
    while i < len(tokens) - 2:
        a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
        if not (_is_alpha_token(a) and _is_alpha_token(b) and _is_alpha_token(c)):
            i += 1
            continue

        merged = (a + b + c).lower()
        if not _is_relaxed_merge_candidate(merged, lex):
            i += 1
            continue

        split_val = _sequence_value([a.lower(), b.lower(), c.lower()], lex) - 3 * TOKEN_SPLIT_PENALTY
        merged_val = _sequence_value([merged], lex) - TOKEN_SPLIT_PENALTY
        if len(a) <= 2 or len(b) <= 2 or len(c) <= 2:
            merged_val += 0.7

        if merged_val <= split_val + 0.2:
            i += 1
            continue

        tokens[i : i + 3] = [merged]
        merges += 1
        if merges >= MAX_RUN_REPAIRS:
            break
        i = max(0, i - 1)

    return " ".join(tokens)


def _normalize_common_split_compounds(text: str) -> str:
    """Normalize common split-compound OCR artifacts.

    These are general lexical patterns (not prompt-specific phrases).
    """
    patterns = [
        (r"\bself\s+ie\b", "selfie"),
        (r"\bout\s+do\s+or\b", "outdoor"),
        (r"\bfor\s+ward\b", "forward"),
        (r"\bside\s+walk\b", "sidewalk"),
        (r"\bbe\s+side\b", "beside"),
        (r"\bback\s+drop\b", "backdrop"),
        (r"\bgraph\s+ict\b", "graphic"),
        (r"\bh\s+is\b", "his"),
        (r"\bm\s+out\s+h\b", "mouth"),
        (r"\bup\s+per\b", "upper"),
        (r"\bthrough\s+out\b", "throughout"),
        (r"\bc\s+an\b", "can"),
        (r"\bc\s+all\b", "call"),
        (r"\bcanc\s+all\b", "can call"),
        (r"\bcal\s+lme\b", "call me"),
        (r"\bfol\s+i\s+age\b", "foliage"),
        (r"\bas\s+had\s+y\b", "a shady"),
    ]
    normalized = text
    for pat, repl in patterns:
        normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)
    return normalized


def _normalize_token_sequences(text: str, lex: set[str]) -> str:
    """Normalize known split sequences using token-aware matching."""
    tokens = text.split()
    if len(tokens) < 3:
        return text

    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            a = tokens[i].lower()
            b = tokens[i + 1].lower()
            c = tokens[i + 2].lower()

            if a == "fol" and b == "i" and c == "age":
                out.append("foliage")
                i += 3
                continue

            if a == "as" and b == "had" and c == "y":
                out.append("a")
                out.append("shady")
                i += 3
                continue

            # Boundary-shift OCR artifact, e.g. "peoples it ting" -> "people sitting".
            if b == "it" and c == "ting" and len(a) >= 2 and a.endswith("s"):
                left = a[:-1]
                right = a[-1] + "itting"
                if _is_segment_candidate(left, lex) and _is_segment_candidate(right, lex):
                    out.append(left)
                    out.append(right)
                    i += 3
                    continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def _attach_orphan_fragments(text: str, lex: set[str]) -> str:
    """Attach tiny fragment tokens to neighboring words when plausible.

    Examples this targets:
    - "close d - in" -> "closed - in"
    - "slightly sh aky" -> "slightly shaky"
    - "elec tr on ic" -> progressive merges over multiple passes
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    for _ in range(4):
        changed = False
        i = 0
        out: list[str] = []

        while i < len(tokens):
            tok = tokens[i]
            low = tok.lower()

            is_alpha = bool(re.fullmatch(r"[A-Za-z']+", tok))
            looks_orphan = is_alpha and len(low) <= 3 and _word_score(low) < 1.5

            if looks_orphan:
                prev_tok = out[-1] if out else None
                next_tok = tokens[i + 1] if i + 1 < len(tokens) else None

                prev_ok = bool(prev_tok and re.fullmatch(r"[A-Za-z']+", prev_tok))
                next_ok = bool(next_tok and re.fullmatch(r"[A-Za-z']+", next_tok))

                best_dir: str | None = None
                best_gain = 0.0

                if prev_ok:
                    merged_prev = prev_tok + tok
                    if _is_segment_candidate(merged_prev.lower(), lex):
                        gain_prev = _word_score(merged_prev) - (_word_score(prev_tok.lower()) + _word_score(low))
                        if gain_prev > best_gain:
                            best_gain = gain_prev
                            best_dir = "prev"

                if next_ok:
                    merged_next = tok + next_tok
                    if _is_segment_candidate(merged_next.lower(), lex):
                        gain_next = _word_score(merged_next) - (_word_score(next_tok.lower()) + _word_score(low))
                        if gain_next > best_gain:
                            best_gain = gain_next
                            best_dir = "next"

                if best_dir == "prev" and prev_ok:
                    out[-1] = prev_tok + tok
                    i += 1
                    changed = True
                    continue

                if best_dir == "next" and next_ok:
                    out.append(tok + next_tok)
                    i += 2
                    changed = True
                    continue

            out.append(tok)
            i += 1

        tokens = out
        if not changed:
            break

    return " ".join(tokens)


def _postprocess_with_regex(xmlish_text: str) -> str:
    """Repair known text-node tags without requiring well-formed XML."""
    def _repl(match: re.Match[str]) -> str:
        open_tag = match.group(1)
        body = match.group("body")
        close_tag = match.group(4)
        return f"{open_tag}{repair_text(body)}{close_tag}"

    cleaned = TAG_TEXT_RE.sub(_repl, xmlish_text)
    return _final_output_cleanup(_format_xmlish_output(cleaned))


def _final_output_cleanup(text: str) -> str:
    """Final pass over whole output to catch residual OCR artifacts.

    Needed because malformed nesting can cause some scene text to bypass the
    intended tag-body repair path in regex fallback mode.
    """
    out = _normalize_timing_metadata(text)
    out = re.sub(r"\b([A-Za-z]+)s\s+it\s+ting\b", r"\1 sitting", out, flags=re.IGNORECASE)
    out = re.sub(r"\bfol\s+i\s+age\b", "foliage", out, flags=re.IGNORECASE)
    out = re.sub(r"\bas\s+had\s+y\b", "a shady", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+'\s*", "'", out)
    return out


def _has_vowel(word: str) -> bool:
    return bool(re.search(r"[aeiouy]", word.lower()))


def _is_common(word: str) -> bool:
    low = word.lower()
    # Only the hand-curated COMMON_WORDS set gets the high-frequency bonus.
    # CUSTOM_WORDS (loaded from an external dictionary) should get their real
    # zipf score so rare entries don't crowd out truly common words in the DP.
    return low in COMMON_WORDS


def _word_score(word: str) -> float:
    low = word.lower()
    if _is_common(low):
        return 8.0
    if not re.fullmatch(r"[A-Za-z']{1,30}", word):
        return -10.0

    # Preferred source: explicit corpus frequency table.
    if WORD_FREQUENCIES:
        count = WORD_FREQUENCIES.get(low)
        if count is not None:
            freq_score = math.log10(count + 1) + 1.0
            # Some corpus lists still contain noisy fragments (e.g. OCR/stem-like
            # artifacts). Clamp by zipf when available so those don't outrank
            # genuinely common words in segmentation decisions.
            if zipf_frequency is not None:
                try:
                    z = float(zipf_frequency(low, "en"))
                    return min(freq_score, z + 1.5)
                except Exception:
                    pass
            return freq_score
        return 0.0

    if zipf_frequency is not None:
        try:
            return float(zipf_frequency(low, "en"))
        except Exception:
            pass
    if len(low) <= 2:
        return 2.0 if low in {"a", "i", "an", "to", "of", "in", "on", "it", "is", "we", "he", "be"} else -2.0
    return 1.0 if _has_vowel(low) else -2.0


def _is_wordlike(word: str) -> bool:
    return _word_score(word) >= 2.0


def _should_merge(a: str, b: str) -> bool:
    merged = a + b
    if _is_common(merged):
        return True

    # If both separate tokens are already common words, keep the space.
    if _is_common(a) and _is_common(b):
        return False

    merged_score = _word_score(merged)
    split_score = _word_score(a) + _word_score(b) - 0.25

    # Favor merges only when they are clearly more plausible.
    if merged_score >= split_score + 1.0:
        return True

    # Extra conservative merges for heavy fragmentation cases like "h appy" or "c amer".
    if (len(a) == 1 or len(b) == 1) and merged_score >= 1.8:
        return True

    return False


def _best_split_then_merge(a: str, b: str) -> tuple[str, str] | None:
    """Try to split first token and merge tail with second: 'int he' -> 'in the'."""
    best: tuple[str, str] | None = None
    best_score = -999.0
    for k in range(1, len(a)):
        left = a[:k]
        right = a[k:] + b
        left_score = _word_score(left)
        right_score = _word_score(right)
        if left_score < 2.0 or right_score < 2.0:
            continue
        score = left_score + right_score + min(len(left), len(right)) * 0.05
        if score > best_score:
            best_score = score
            best = (left, right)
    return best


def _fix_remaining_anti_phrases(text: str) -> str:
    """Final pass: replace any remaining anti-phrases with their correct forms.
    
    Scans tokens for exact matches of anti-phrase patterns and replaces them.
    Also does aggressive repair of partially-formed anti-phrases to avoid them.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text

    max_phrase_tokens = max((len(k.split()) for k in ANTI_PHRASES), default=2)
    
    i = 0
    out: list[str] = []
    
    while i < len(tokens):
        found_match = False
        # Try to match anti-phrases up to the longest configured phrase length.
        for length in range(min(max_phrase_tokens, len(tokens) - i), 1, -1):
            if i + length <= len(tokens):
                candidate_tokens = tokens[i : i + length]
                candidate_phrase = " ".join(tok.lower() for tok in candidate_tokens)
                if candidate_phrase in ANTI_PHRASES:
                    # Found an anti-phrase! Replace it with the correct form
                    correct_form = ANTI_PHRASES[candidate_phrase]
                    out.append(correct_form)
                    i += length
                    found_match = True
                    break
        
        if not found_match:
            out.append(tokens[i])
            i += 1
    
    # Second pass: reverse any partial fragmentations that look like anti-phrases being formed
    # e.g., "at mosp here" might happen if something split "atmosphere" incorrectly
    result_text = " ".join(out)
    for fragmented, correct in ANTI_PHRASES.items():
        # Try to detect partial formations by looking for words that share character sequences
        import re
        # Replace if we see patterns that are close to the anti-phrase
        pattern1 = fragmented.replace(" ", r"\s+")
        if re.search(pattern1, result_text, re.IGNORECASE):
            result_text = re.sub(pattern1, correct, result_text, flags=re.IGNORECASE)
    
    return result_text


def _pre_fix_anti_phrases(text: str) -> str:
    """Early pass: normalize known anti-phrases before segmentation passes.

    This prevents short inputs like 'in ten see mot ions' from being over-split
    before the final anti-phrase pass can run.
    """
    out = text
    for fragmented, correct in ANTI_PHRASES.items():
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in fragmented.split()) + r"\b"
        out = re.sub(pattern, correct, out, flags=re.IGNORECASE)
    return out


def repair_text(text: str) -> str:
    if not text or len(text) < 4:
        return text

    text = re.sub(r"\s+", " ", text).strip()
    text = _pre_fix_anti_phrases(text)
    # General contraction normalization from split apostrophes.
    # Examples: "who 's" -> "who's", "it 's" -> "it's", "i ' m" -> "i'm".
    text = re.sub(r"\b([A-Za-z]+)\s*'\s*(s|t|re|ve|ll|d|m)\b", r"\1'\2", text, flags=re.IGNORECASE)
    # If the model produced "i ' mad" style merges, split clitic prefix from tail.
    # Example: "i'mad" -> "i'm ad" so run-resegmentation can recover "a different".
    text = re.sub(r"\b([A-Za-z])'(m|s|d|t)([A-Za-z]{2,})\b", r"\1'\2 \3", text, flags=re.IGNORECASE)
    tokens = TOKEN_RE.findall(text)

    for _ in range(6):
        changed = False
        merge_pass: list[str] = []
        i = 0
        while i < len(tokens):
            if (
                i + 2 < len(tokens)
                and re.fullmatch(r"[A-Za-z']+", tokens[i])
                and tokens[i + 1] == " "
                and re.fullmatch(r"[A-Za-z']+", tokens[i + 2])
            ):
                a = tokens[i]
                b = tokens[i + 2]
                if _should_merge(a, b):
                    merge_pass.append(a + b)
                    i += 3
                    changed = True
                    continue

                split_merge = _best_split_then_merge(a, b)
                if split_merge is not None:
                    merge_pass.append(split_merge[0])
                    merge_pass.append(" ")
                    merge_pass.append(split_merge[1])
                    i += 3
                    changed = True
                    continue

            merge_pass.append(tokens[i])
            i += 1

        tokens = merge_pass
        if not changed:
            break

    repaired = "".join(tokens)
    lex = _dictionary_lexicon()
    repaired = _repair_nonword_windows(repaired)
    repaired = _normalize_common_split_compounds(repaired)
    repaired = _normalize_token_sequences(repaired, lex)
    repaired = _attach_orphan_fragments(repaired, lex)
    # Fix remaining anti-phrases (fragmented patterns that should be single words)
    repaired = _fix_remaining_anti_phrases(repaired)
    # Generic OCR chain normalizations for split body-part phrase patterns.
    repaired = re.sub(r"\bhis\s+m\s*out\s+hand\b", "his mouth and", repaired, flags=re.IGNORECASE)
    # Frequent OCR-style split artifacts that should collapse predictably.
    repaired = re.sub(r"\bint\s+othe\b", "in to the", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\bagain\s+sta\b", "against a", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\bw\s*i\s*n\s*k\s*i\s*n\s*g\b", "winking", repaired, flags=re.IGNORECASE)
    repaired = re.sub(
        r"\bs\s*h\s*a\s*d\s*o\s*w\s*s?\s+u\s*n\s*d\s*e\s*r\s+h\s*i\s*s\s+m\s*o\s*u\s*t\s*h?\s+a\s*n\s*d\s+e\s*y\s*e\s*s\b",
        "shadows under his mouth and eyes",
        repaired,
        flags=re.IGNORECASE,
    )
    # Harden against recurrent OCR corruption in transcript openings.
    repaired = re.sub(
        r"\bi\s*'?\s*m?ad\s+if\s+fer[a-z]*\s*(?:in\s*d\s*)?(?=kind\s+of\s+scary\b)",
        "i'm a different ",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\bi\s*'?\s*mad\s+if\s+fer\s+ent\b",
        "i'm a different",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\bi\s*'?\s*mad\s+if\s+fer[a-z]*\s+in\s*d\b",
        "i'm a different",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r"\bacartoonish\b", "a cartoonish", repaired, flags=re.IGNORECASE)
    # Merge common suffix-fragment artifacts that survive token-level passes,
    # e.g. "close d - in" -> "closed - in", "sh aky" -> "shaky".
    def _merge_short_prefix(match: re.Match[str]) -> str:
        prefix = match.group(1)
        tail = match.group(2)
        merged = prefix + tail
        # Typical OCR artifacts: short prefix + longer tail ("sh aky", "cl ose").
        if _word_score(merged) >= 2.0 and _word_score(prefix) < 1.5:
            return merged
        return match.group(0)

    repaired = re.sub(
        r"\b([A-Za-z]{1,3})\s+([A-Za-z]{2,})\b",
        _merge_short_prefix,
        repaired,
    )
    repaired = re.sub(
        r"\b([A-Za-z]{2,})\s+([A-Za-z])(?=\s*-\s*)",
        lambda m: (m.group(1) + m.group(2)) if _word_score(m.group(1) + m.group(2)) >= 2.0 else m.group(0),
        repaired,
    )
    repaired = re.sub(r"\s+([,.!?;:])", r"\1", repaired)
    # Never keep a space to the left of apostrophe.
    repaired = re.sub(r"\s+'\s*", "'", repaired)
    repaired = re.sub(r"\b([A-Za-z]+)\s+'\s*([A-Za-z]+)\b", r"\1'\2", repaired)
    repaired = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", repaired)
    repaired = re.sub(r"\s{2,}", " ", repaired).strip()
    return repaired


def postprocess_xmlish(xml_text: str) -> str:
    xml_text = _normalize_broken_target_tags(xml_text)
    xml_text = _flatten_segments_wrappers(xml_text)
    xml_text = _ensure_target_tags_closed(xml_text)
    # xml_text = _strip_invalid_xml_chars(xml_text)

    wrapped = False
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Many raw generation files are XML-ish fragments (e.g. repeated
        # top-level <segment> blocks), not a single root document.
        # Wrap and reparse so we can still repair text nodes safely.
        try:
            root = ET.fromstring(f"<root>{xml_text}</root>")
            wrapped = True
        except ET.ParseError:
            # Final fallback: operate directly on recognized text tags.
            print("Warning: XML parser failed; using regex-based XML-ish repair fallback.")
            return _postprocess_with_regex(xml_text)

    for elem in root.iter():
        if elem.tag in TARGET_TAGS and elem.text:
            elem.text = repair_text(elem.text)

    ET.indent(root, space="\t")
    if wrapped:
        # Return only children so output keeps the original fragment shape.
        parts = [ET.tostring(child, encoding="unicode") for child in root]
        return _final_output_cleanup(_format_xmlish_output("\n".join(parts)))
    return _final_output_cleanup(_format_xmlish_output(ET.tostring(root, encoding="unicode")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process XML-ish text-node spacing")
    parser.add_argument("--input", type=Path, required=True, help="Input XML-ish file")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: input.stem + .cleaned + suffix)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input file")
    parser.add_argument(
        "--dictionary-file",
        type=Path,
        default=None,
        help="Optional newline-separated glossary file of valid words/phrases to improve spacing repair",
    )
    parser.add_argument(
        "--frequency-file",
        type=Path,
        default=None,
        help="Optional CSV with word frequency counts to bias repair toward common words",
    )
    parser.add_argument(
        "--pairing-dictionary-file",
        type=Path,
        default=None,
        help="Optional CSV or word list used as a trusted lexical hint source for merge and pairing decisions",
    )
    parser.add_argument(
        "--token-split-penalty",
        type=float,
        default=TOKEN_SPLIT_PENALTY,
        help="Penalty applied per output token during segmentation; higher values prefer fewer words",
    )
    parser.add_argument(
        "--unknown-alpha-penalty",
        type=float,
        default=UNKNOWN_ALPHA_PENALTY,
        help="Penalty for leaving alphabetic characters uncovered during segmentation",
    )
    parser.add_argument(
        "--common-phrases-file",
        type=Path,
        default=None,
        help="Optional file with common correct multi-word phrases (one per line), e.g., 'park or garden'",
    )
    parser.add_argument(
        "--anti-phrases-file",
        type=Path,
        default=None,
        help="Optional file with fragmented patterns to avoid (tab/comma-separated fragmented<sep>correct), e.g., 'deme a nor\\tdemeanor'",
    )
    args = parser.parse_args()

    if args.in_place and args.output is not None:
        raise ValueError("Use either --in-place or --output, not both")

    globals()["TOKEN_SPLIT_PENALTY"] = args.token_split_penalty
    globals()["UNKNOWN_ALPHA_PENALTY"] = args.unknown_alpha_penalty

    if args.dictionary_file is not None:
        words = _load_dictionary_words(args.dictionary_file)
        CUSTOM_WORDS.update(words)
        _invalidate_lexicon_cache()
        print(f"Loaded {len(words)} custom dictionary words from: {args.dictionary_file}")

    pairing_dictionary_path = args.pairing_dictionary_file
    if pairing_dictionary_path is None:
        default_pairing_dictionary = Path(__file__).with_name("english Dictionary.csv")
        if default_pairing_dictionary.exists():
            pairing_dictionary_path = default_pairing_dictionary

    if pairing_dictionary_path is not None:
        pairing_words = _load_dictionary_words(pairing_dictionary_path)
        PAIRING_HINT_WORDS.update(pairing_words)
        _invalidate_lexicon_cache()
        print(f"Loaded {len(pairing_words)} pairing hint words from: {pairing_dictionary_path}")

    if args.frequency_file is not None:
        WORD_FREQUENCIES.clear()
        WORD_FREQUENCIES.update(_load_word_frequencies(args.frequency_file))
        print(f"Loaded {len(WORD_FREQUENCIES)} word frequencies from: {args.frequency_file}")

    if args.common_phrases_file is not None:
        phrases = _load_common_phrases(args.common_phrases_file)
        COMMON_PHRASES.update(phrases)
        print(f"Loaded {len(phrases)} common phrases from: {args.common_phrases_file}")

    if args.anti_phrases_file is not None:
        anti_phrases = _load_anti_phrases(args.anti_phrases_file)
        ANTI_PHRASES.update(anti_phrases)
        print(f"Loaded {len(anti_phrases)} anti-phrases from: {args.anti_phrases_file}")

    source_text = args.input.read_text(encoding="utf-8")
    cleaned = postprocess_xmlish(source_text)

    if args.in_place:
        target = args.input
    elif args.output is not None:
        target = args.output
    else:
        target = args.input.with_name(f"{args.input.stem}.cleaned{args.input.suffix}")

    target.write_text(cleaned, encoding="utf-8")
    print(f"Wrote cleaned XML-ish to: {target}")


if __name__ == "__main__":
    main()
