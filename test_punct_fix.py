#!/usr/bin/env python3
"""Test the punctuation spacing function."""

from generate_example_with_carmind2_lora import fix_punctuation_spacing

test_cases = [
    ('Hello.World', 'Hello. World'),
    ('Word,another', 'Word, another'),
    ('[SPEAKER_1]This is your lobbyist, your Democrat lobbyist.Since', '[SPEAKER_1]This is your lobbyist, your Democrat lobbyist. Since'),
    ('saying"Sorry', 'saying "Sorry'),
    ('At 1s,.At 2s,.At 3s', 'At 1s, .At 2s, .At 3s'),
]

print("Testing punctuation spacing fixes:\n")
for before, expected in test_cases:
    fixed = fix_punctuation_spacing(before)
    status = "✓" if fixed == expected else "✗"
    print(f"{status} Input:    {before}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fixed}")
    if fixed != expected:
        print("  MISMATCH!")
    print()
