"""Verify the three fixes on the current normalized candidate."""
import re, xml.etree.ElementTree as ET

# ── inline copies of the fixed helpers ─────────────────────────────────────

_ALLOWED_XMLISH_TAGS = {
    "example", "topic", "creator", "title", "input", "output",
    "segments", "segment", "start", "end", "duration",
    "scene", "transcript", "audio", "ui_description",
}

def _normalize_speaker_tags(text):
    if not text:
        return text

    def _fix(match):
        raw_name = match.group(1).upper()
        idx = match.group(2)
        compact = re.sub(r"[^A-Z]", "", raw_name)
        if (compact.startswith("SPEAK") or compact.startswith("SPAKER")
                or compact.startswith("SPEKER") or compact.startswith("SPEAKER")
                or (compact.startswith("SP") and "EAK" in compact)):
            return f"[SPEAKER_{idx}]"
        return match.group(0)

    # Handle wildcard/missing index: [SPEAKER_*], [SPEAKER_?]
    text = re.sub(r"\[\s*SPEAK(?:ER|HER|ERS|ARY|ERY)?[_\s]*[*?]\s*\]",
                  "[SPEAKER_1]", text, flags=re.IGNORECASE)
    return re.sub(r"\[\s*([A-Za-z][A-Za-z _-]*)\s*_?\s*(\d+)\s*\]", _fix, text)


def _inject_wrappers_if_missing(candidate):
    _first_seg = re.search(r"<\s*segment\b", candidate, flags=re.IGNORECASE)
    if _first_seg is None:
        return candidate
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
        return _before.rstrip() + _inject + "\n" + _after
    return candidate


# ── load the normalized file ────────────────────────────────────────────────
with open("new_example.xmlish.txt.normalized.raw.txt", encoding="utf-8", errors="replace") as f:
    text = f.read()

print("=== TEST 1: [SPEAKER_*] normalization ===")
fixed = _normalize_speaker_tags(text)
if "[SPEAKER_*]" in fixed:
    print("FAIL: [SPEAKER_*] still present")
else:
    print("OK: no [SPEAKER_*] in output")

print()
print("=== TEST 2: Stray > removal ===")
has_stray = bool(re.search(r"(?m)^\s*>\s*$", text))
print(f"Stray > present in input: {has_stray}")
# simulate the body_wo_timing cleanup
cleaned = re.sub(r"(?m)^\s*>\s*$", "", text).strip()
has_stray_after = bool(re.search(r"(?m)^\s*>\s*$", cleaned))
print(f"Stray > present after cleanup: {has_stray_after}")

print()
print("=== TEST 3: Missing wrapper injection ===")
injected = _inject_wrappers_if_missing(fixed)
has_output = bool(re.search(r"<\s*output\b", injected, re.IGNORECASE))
has_segments = bool(re.search(r"<\s*segments\b", injected, re.IGNORECASE))
has_input_close = bool(re.search(r"</\s*input\s*>", injected, re.IGNORECASE))
print(f"</input> present: {has_input_close}")
print(f"<output> present: {has_output}")
print(f"<segments> present: {has_segments}")

# Now simulate the canonical closing append
sim = re.sub(r"</\s*segments\s*>", "", injected, flags=re.IGNORECASE)
sim = re.sub(r"</\s*output\s*>", "", sim, flags=re.IGNORECASE)
sim = re.sub(r"</\s*example\s*>", "", sim, flags=re.IGNORECASE)
sim = sim.rstrip() + "\n</segments>\n</output>\n</example>"

print()
print("=== TEST 4: Parse simulated final candidate ===")
try:
    ET.fromstring(sim)
    print("OK: XML parses successfully")
except ET.ParseError as e:
    print(f"FAIL: {e}")
    # show context around error
    lines = sim.splitlines()
    lineno = e.position[0] - 1
    for i in range(max(0, lineno-2), min(len(lines), lineno+3)):
        marker = ">>>" if i == lineno else "   "
        print(f"  {marker} {i+1}: {lines[i]}")
