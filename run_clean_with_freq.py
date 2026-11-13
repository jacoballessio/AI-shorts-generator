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
out = p.postprocess_xmlish(raw)
Path("new_example.cleaned.xmlish.txt").write_text(out, encoding="utf-8")

print("words", len(p.CUSTOM_WORDS))
print("freqs", len(p.WORD_FREQUENCIES))
print("witha", out.lower().count("witha"))
print("with a", out.lower().count("with a"))
print("segment_close", out.lower().count("</segment>"))
