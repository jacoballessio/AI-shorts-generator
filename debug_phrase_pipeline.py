from pathlib import Path
import postprocess_xmlish_text as p

p.CUSTOM_WORDS.clear()
p.CUSTOM_WORDS.update(p._load_dictionary_words(Path("words.txt")))
p.WORD_FREQUENCIES.clear()
p.WORD_FREQUENCIES.update(
    p._load_word_frequencies(Path(r"C:/Users/jacob/Downloads/valid_words_sorted_by_frequency.txt"))
)
p._invalidate_lexicon_cache()

raw = Path("new_example.xmlish.txt.raw.txt").read_text(encoding="utf-8")
xml = p._normalize_broken_target_tags(raw)
xml = p._ensure_target_tags_closed(xml)
out = p._postprocess_with_regex(xml)

bad = "fol i age and as had y"
good = "foliage and a shady"
print("bad_count", out.count(bad))
print("good_count", out.count(good))
if bad in out:
    i = out.index(bad)
    print(out[max(0, i-80):i+120])
