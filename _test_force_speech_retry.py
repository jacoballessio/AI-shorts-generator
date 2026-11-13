from generate_example_with_carmind2_lora import (
    _seed_first_missing_or_empty_transcript_segment,
    finalize_xmlish_record,
)


def test_force_speech_seed() -> None:
    stream = (
        "<segment>\n"
        "<start>0</start>\n"
        "<end>3</end>\n"
        "<duration>3</duration>\n"
        "<scene>a person slips on a banana peel</scene>\n"
        "<transcript></transcript>\n"
        "</segment>\n"
    )
    seeded = _seed_first_missing_or_empty_transcript_segment(stream)
    print("SEEDED", seeded is not None)
    assert seeded is not None
    assert "</scene>\n<transcript>" in seeded
    assert any(f"<transcript>{ch}" in seeded for ch in ("T", "L", "S", "W"))


def test_force_speech_validation() -> None:
    prefix = (
        "<example>\n<input>\n<duration>3</duration>\n<creator>X</creator>\n<topic>Y</topic>\n<title>Z</title>\n"
        "</input>\n<output>\n<segments>\n"
    )
    completion = (
        "<segment>\n"
        "<start>0</start>\n"
        "<end>3</end>\n"
        "<duration>3</duration>\n"
        "<scene>a person slips on a banana peel</scene>\n"
        "<transcript></transcript>\n"
        "</segment>\n"
    )
    try:
        finalize_xmlish_record(prefix, completion, force_speech=True)
    except ValueError as exc:
        print("VALIDATION", str(exc))
        assert "empty/missing <transcript>" in str(exc)
    else:
        raise AssertionError("Expected force_speech validation to fail")


test_force_speech_seed()
test_force_speech_validation()
print("ok")