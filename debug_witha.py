import postprocess_xmlish_text as p
from pathlib import Path

p.CUSTOM_WORDS.update(p._load_dictionary_words(Path('words.txt')))
p._invalidate_lexicon_cache()
lex = p._dictionary_lexicon()

# Simulate repair on the exact text before/after merge loop
raw_scene = "The video opens witha close - up selfie - style shot of a bald man wearing whiteface paint"
print("=== repair_text result ===")
result = p.repair_text(raw_scene)
print(result[:200])
print()

# Check witha specifically
print("=== _repair_nonword_windows on raw ===")
result2 = p._repair_nonword_windows(raw_scene)
print(result2[:200])

# Check witha token specifically
print()
lex2 = p._dictionary_lexicon()
tokens = raw_scene.split()
for i, tok in enumerate(tokens):
    low = tok.lower()
    if low == 'witha':
        from wordfreq import zipf_frequency
        _in_lex = low in lex2
        ws = p._word_score(low)
        print(f"witha: in_lex={_in_lex}, word_score={ws:.2f}")
        print(f"  in COMMON_WORDS: {low in p.COMMON_WORDS}")
        print(f"  in CUSTOM_WORDS: {low in p.CUSTOM_WORDS}")
        is_rare = (_in_lex and low not in p.COMMON_WORDS and len(low)<=5
                   and zipf_frequency is not None and ws < 3.5)
        print(f"  is_rare={is_rare}")
        print(f"  is_nonword={p._is_alpha_token(tok) and (not _in_lex or is_rare)}")
        start = max(0, i-2)
        end = min(len(tokens)-1, i+2)
        window = tokens[start:end+1]
        compact = "".join(window)
        seg, cov = p._segment_string_by_dictionary(compact, lex2)
        print(f"  window={window}")
        print(f"  compact={compact!r}")
        print(f"  segmented={seg}, covered={cov}")
