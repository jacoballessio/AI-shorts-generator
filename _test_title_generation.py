from generate_example_with_carmind2_lora import _sanitize_generated_title, build_xmlish_prefix_prompt


def test_sanitize_generated_title() -> None:
    raw = 'Title: "Deluxe Elf Chaos"\n<segment>'
    title = _sanitize_generated_title(raw, fallback_topic="Education")
    print("TITLE", title)
    assert title == "Deluxe Elf Chaos", title


def test_prompt_contains_concrete_title() -> None:
    prompt = build_xmlish_prefix_prompt(
        creator="Daniel Hentschel",
        topic="Education",
        duration=28,
        sources=[],
        title="Deluxe Elf Chaos",
    )
    print("HAS_TITLE_TAG", "<title>Deluxe Elf Chaos</title>" in prompt)
    assert "<title>Deluxe Elf Chaos</title>" in prompt
    assert "<title>\n" not in prompt


test_sanitize_generated_title()
test_prompt_contains_concrete_title()
print("ok")