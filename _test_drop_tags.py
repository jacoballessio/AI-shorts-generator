import sys, re, difflib
sys.stdout.reconfigure(encoding='utf-8')

_ALLOWED_XMLISH_TAGS = {
    "topic", "creator", "title", "input", "output",
    "segments", "segment", "start", "end", "duration",
    "scene", "transcript", "audio", "ui_description",
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
        text = re.sub(rf"<{re.escape(name)}\b[^>]*>.*?</{re.escape(name)}\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(rf"</?{re.escape(name)}\b[^>]*>", "", text, flags=re.IGNORECASE)
    def _fix_tag(match):
        slash = match.group(1)
        name = match.group(2).lower()
        if name in _ALLOWED_XMLISH_TAGS:
            return f"</{name}>" if slash else f"<{name}>"
        if name in normalize_map:
            canonical = normalize_map[name]
            return f"</{canonical}>" if slash else f"<{canonical}>"
        return ""
    return re.sub(r"<\s*(/?)\s*([A-Za-z_][A-Za-z0-9_]*)\b[^>]*>", _fix_tag, text)

tests = [
    ('<endorse>true</endorse>', ''),
    ('<translipt>[SPEAKER_1] hello</translipt>', '<transcript>[SPEAKER_1] hello</transcript>'),
    ('<transcript>keep me</transcript>', '<transcript>keep me</transcript>'),
    ('<endorse>yes</endorse><segment><start>1</start></segment>', '<segment><start>1</start></segment>'),
    ('<Rating>5/5</Rating>', ''),  # clearly unknown
]
for inp, expected in tests:
    result = _drop_disallowed_xmlish_tags(inp)
    status = 'OK' if result == expected else 'FAIL'
    print(f'{status}: {repr(inp)} -> {repr(result)}')
