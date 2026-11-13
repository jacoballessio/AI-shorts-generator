"""Quick test to verify 'example' fix in _ALLOWED_XMLISH_TAGS."""
import re
import difflib

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


def _drop_disallowed_xmlish_tags(text: str) -> str:
    if not text:
        return text
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
    for name in truly_unknown:
        text = re.sub(
            rf"<{re.escape(name)}\b[^>]*>.*?</{re.escape(name)}\s*>",
            "", text, flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(rf"</?{re.escape(name)}\b[^>]*>", "", text, flags=re.IGNORECASE)

    def _fix_tag(match: re.Match) -> str:
        slash = match.group(1)
        name = match.group(2).lower()
        if name in _ALLOWED_XMLISH_TAGS:
            return f"</{name}>" if slash else f"<{name}>"
        if name in normalize_map:
            canonical = normalize_map[name]
            return f"</{canonical}>" if slash else f"<{canonical}>"
        return ""

    return re.sub(r"<\s*(/?)\s*([A-Za-z_][A-Za-z0-9_]*)\b[^>]*>", _fix_tag, text)


with open("new_example.xmlish.txt.full.raw.txt", encoding="utf-8", errors="replace") as f:
    full_raw = f.read()

result = _drop_disallowed_xmlish_tags(full_raw)
print(f"Input  length: {len(full_raw)}")
print(f"Output length: {len(result)}")
print(f"Has <example>: {'<example>' in result}")
print(f"Has <segment>: {'<segment>' in result or '<segment ' in result}")
print(f"Output starts: {result[:120]!r}")
print(f"Output ends:   {result[-80:]!r}")
