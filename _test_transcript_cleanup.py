from generate_example_with_carmind2_lora import _reclassify_segment_payload


def run_case(seg):
    seg = dict(seg)
    _reclassify_segment_payload(seg)
    return seg.get("transcript", "")


cases = [
    (
        {
            "scene_description": "",
            "audio_description": "",
            "transcript": "[SPEAKER_1][to hold the Deluxe Elfa package steadily, its vibrant colors and detailed design contrasting sharply with their casual attire, maintaining a lively engaging tone driven by their enthusiasm until the end of the clip.] Oh my god, the deluxe is finally here.",
        },
        "[SPEAKER_1] Oh my god, the deluxe is finally here.",
    ),
    (
        {
            "scene_description": "",
            "audio_description": "",
            "transcript": "[SPEAKER_1][\\\" in an excited high-pitched voice full of energy and delight.] Hey, Road Clown, let's see if we can find some more Christmas trees.",
        },
        "[SPEAKER_1][excited high-pitched voice] Hey, Road Clown, let's see if we can find some more Christmas trees.",
    ),
    (
        {
            "scene_description": "",
            "audio_description": "",
            "transcript": "[cheerful, encouraging tone]",
        },
        "",
    ),
]

for idx, (seg, expected) in enumerate(cases, start=1):
    actual = run_case(seg)
    print(idx, repr(actual))
    assert actual == expected, f"case {idx}: expected {expected!r}, got {actual!r}"

print("ok")