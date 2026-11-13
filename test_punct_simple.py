#!/usr/bin/env python3
"""Test the punctuation spacing function (no imports)."""

import re

def fix_punctuation_spacing(text: str) -> str:
    """Fix spacing around punctuation marks."""
    if not text:
        return text
    
    # Fix space before punctuation (remove extra spaces)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Add space after period/question mark/exclamation followed by capital letter (if missing)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Fix space after other punctuation followed by lowercase (common in generated text)
    text = re.sub(r'([.!?])([a-z])', r'\1 \2', text)
    
    # Fix space after comma if missing
    text = re.sub(r',([A-Za-z])', r', \1', text)
    
    # Fix space before/after quotes
    text = re.sub(r'\s*"\s*', ' "', text)
    text = re.sub(r'"\s+([.,:!?])', r'"\1', text)
    text = re.sub(r'"([A-Za-z])', r'" \1', text)
    
    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix ]. and ][ patterns
    text = re.sub(r'\]\s*\[', '][', text)
    
    return text.strip()

# Test cases
test_cases = [
    ('Hello.World', 'Hello. World'),
    ('Word,another', 'Word, another'),
    ('[SPEAKER_1]This is lobbyist.Since', '[SPEAKER_1]This is lobbyist. Since'),
    ('saying"Sorry', 'saying "Sorry'),
]

print("Testing punctuation spacing fixes:\n")
for before, expected in test_cases:
    fixed = fix_punctuation_spacing(before)
    status = "OK" if fixed == expected else "FAIL"
    print(f"[{status}] {before}")
    print(f"  -> {fixed}\n")
