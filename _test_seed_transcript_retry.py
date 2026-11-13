import pathlib

from generate_example_with_carmind2_lora import (
    _extract_segment_stream,
    _seed_first_bad_transcript_segment,
)


src = pathlib.Path(r"z:/Projects/AI-shorts-generator/new_example.xmlish.txt.normalized.raw.txt")
text = src.read_text(encoding="utf-8")
stream = _extract_segment_stream(text)
seeded = _seed_first_bad_transcript_segment(stream)

out = pathlib.Path(r"z:/Projects/AI-shorts-generator/_seed_transcript_retry_out.txt")
out.write_text(seeded or "<NONE>", encoding="utf-8")

print("FOUND", seeded is not None)
if seeded:
    print("HAS_TRANSCRIPT_SEED", "<transcript>[SPEAKER_" in seeded)
    print("TAIL", seeded[-220:])