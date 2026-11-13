import json
from pathlib import Path
from generate_example_with_carmind2_lora import _reclassify_segment_payload

p = Path(r"z:/Projects/AI-shorts-generator/new_example.xmlish.json")
data = json.loads(p.read_text(encoding="utf-8"))
segments = data.get("output", {}).get("segments", [])
changed = 0
for seg in segments:
    if isinstance(seg, dict) and _reclassify_segment_payload(seg):
        changed += 1

out = Path(r"z:/Projects/AI-shorts-generator/new_example.xmlish.reclassified.preview.json")
out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print("segments", len(segments))
print("changed", changed)
