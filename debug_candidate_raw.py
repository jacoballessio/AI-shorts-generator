import re
import xml.etree.ElementTree as ET
from generate_example_with_carmind2_lora import (
    build_xmlish_prefix_prompt,
    _strip_markdown_fence,
    _sanitize_segments_markers_in_completion,
    _normalize_known_xmlish_tags,
    _normalize_speaker_tags,
    _escape_bare_ampersands,
)

with open("new_example.xmlish.txt.raw.txt", "r", encoding="utf-8") as f:
    completion_text = f.read()

prefix_prompt = build_xmlish_prefix_prompt(
    creator="Daniel Hentschel",
    topic="Comedy",
    duration=42,
    sources=[],
    title="The amazing reason he's red.",
)

raw = _sanitize_segments_markers_in_completion(_strip_markdown_fence(completion_text))
raw = _normalize_known_xmlish_tags(raw)

first_segment = re.search(r"<\s*segment\b", raw, flags=re.IGNORECASE)
if first_segment is not None and first_segment.start() > 0:
    raw = raw[first_segment.start() :]

if "<example" in raw:
    candidate = raw[raw.find("<example") :]
else:
    candidate = prefix_prompt + raw

if "</example>" in candidate:
    candidate = candidate[: candidate.find("</example>") + len("</example>")]
else:
    if "</segments>" not in candidate:
        candidate += "\n    </segments>"
    if "</output>" not in candidate:
        candidate += "\n  </output>"
    if "</example>" not in candidate:
        candidate += "\n</example>"

first_segment_in_candidate = re.search(r"<\s*segment\b", candidate, flags=re.IGNORECASE)
if first_segment_in_candidate is not None:
    head = candidate[: first_segment_in_candidate.start()]
    body = candidate[first_segment_in_candidate.start() :]
    body = re.sub(r"<\s*input\s*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"</\s*input\s*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*output\s*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*segments\s*>", "", body, flags=re.IGNORECASE)
    candidate = head + body

opens = len(re.findall(r"<\s*segment\b", candidate, flags=re.IGNORECASE))
closes = len(re.findall(r"</\s*segment\s*>", candidate, flags=re.IGNORECASE))
if opens > closes:
    candidate = candidate.rstrip() + ("\n</segment>" * (opens - closes))

candidate = re.sub(r"</\s*segments\s*>", "", candidate, flags=re.IGNORECASE)
candidate = re.sub(r"</\s*output\s*>", "", candidate, flags=re.IGNORECASE)
candidate = re.sub(r"</\s*example\s*>", "", candidate, flags=re.IGNORECASE)
candidate = candidate.rstrip() + "\n</segments>\n</output>\n</example>"

candidate = _normalize_speaker_tags(candidate)
candidate = _escape_bare_ampersands(candidate)

print("COUNTS:")
for tag in ["segment", "scene", "transcript", "audio", "output", "segments", "example"]:
    o = len(re.findall(fr"<{tag}>", candidate, flags=re.IGNORECASE))
    c = len(re.findall(fr"</{tag}>", candidate, flags=re.IGNORECASE))
    print(tag, o, c)

with open("debug_candidate_output.xml", "w", encoding="utf-8") as f:
    f.write(candidate)

try:
    ET.fromstring(candidate)
    print("CANDIDATE_XML_OK")
except ET.ParseError as exc:
    print(f"PARSE_ERROR: {exc}")
    m = re.search(r"line (\d+), column (\d+)", str(exc))
    if m:
        line_no = int(m.group(1))
        col_no = int(m.group(2))
        lines = candidate.splitlines()
        print(f"LINE {line_no}: {lines[line_no-1] if 1 <= line_no <= len(lines) else '<out of range>'}")
        if 1 <= line_no <= len(lines):
            line = lines[line_no - 1]
            start = max(0, col_no - 140)
            end = min(len(line), col_no + 140)
            print("CONTEXT:")
            print(line[start:end])
            print(" " * max(0, col_no - start) + "^")
