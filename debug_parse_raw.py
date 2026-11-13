import xml.etree.ElementTree as ET
from generate_example_with_carmind2_lora import (
    finalize_xmlish_record,
    build_xmlish_prefix_prompt,
)

raw_path = "new_example.xmlish.txt.raw.txt"
with open(raw_path, "r", encoding="utf-8") as f:
    t = f.read()

prefix = build_xmlish_prefix_prompt(
    creator="Daniel Hentschel",
    topic="Comedy",
    duration=42,
    sources=[],
    title="The amazing reason he's red.",
)
full = finalize_xmlish_record(prefix, t)

try:
    ET.fromstring(full)
    print("RAW_NORMALIZED_XML_OK")
except ET.ParseError as exc:
    print(f"PARSE_ERROR: {exc}")
    text = str(exc)
    # Parse `line X, column Y` and print neighborhood
    import re
    m = re.search(r"line (\d+), column (\d+)", text)
    if m:
        line_no = int(m.group(1))
        col_no = int(m.group(2))
        lines = full.splitlines()
        if 1 <= line_no <= len(lines):
            line = lines[line_no - 1]
            start = max(0, col_no - 120)
            end = min(len(line), col_no + 120)
            print("CONTEXT:")
            print(line[start:end])
            print(" " * max(0, col_no - start) + "^")
