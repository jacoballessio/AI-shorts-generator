from generate_example_with_carmind2_lora import (
    _classify_description_chunk,
    _seed_first_missing_transcript_and_audio_segment,
)


def test_ambient_light_is_not_audio() -> None:
    chunk = "a 360-degree panoramic shot capturing a cozy indoor space bathed in warm ambient light"
    kind = _classify_description_chunk(chunk)
    print("KIND", kind)
    assert kind == "scene", f"expected scene, got {kind!r}"


def test_squeaky_sound_is_audio() -> None:
    chunk = "a high-pitched squeaky sound"
    kind = _classify_description_chunk(chunk)
    print("SQUEAK_KIND", kind)
    assert kind == "audio_effect", f"expected audio_effect, got {kind!r}"


def test_missing_transcript_and_audio_seeds_audio() -> None:
    stream = (
        "<segment>\n"
        "<start>0</start>\n"
        "<end>5</end>\n"
        "<duration>5</duration>\n"
        "<scene>a person stands in a room</scene>\n"
        "<transcript></transcript>\n"
        "</segment>\n"
    )
    seeded = _seed_first_missing_transcript_and_audio_segment(stream)
    print("SEEDED", seeded is not None)
    assert seeded is not None
    assert "<audio>a " in seeded


test_ambient_light_is_not_audio()
test_squeaky_sound_is_audio()
test_missing_transcript_and_audio_seeds_audio()
print("ok")