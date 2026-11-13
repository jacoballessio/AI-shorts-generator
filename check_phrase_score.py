from pathlib import Path
import postprocess_xmlish_text as p

p.CUSTOM_WORDS.clear()
p.CUSTOM_WORDS.update(p._load_dictionary_words(Path("words.txt")))
p.WORD_FREQUENCIES.clear()
p.WORD_FREQUENCIES.update(p._load_word_frequencies(Path(r"C:/Users/jacob/Downloads/valid_words_sorted_by_frequency.txt")))
p._invalidate_lexicon_cache()
lex = p._dictionary_lexicon()

a = ["with", "fol", "i", "age", "and", "as", "had", "y", "figure", "to", "his", "left"]
b = ["with", "foliage", "and", "a", "shady", "figure", "to", "his", "left"]

score_a = p._sequence_value(a, lex) - len(a) * p.TOKEN_SPLIT_PENALTY
score_b = p._sequence_value(b, lex) - len(b) * p.TOKEN_SPLIT_PENALTY

print("A fragmented score:", round(score_a, 4))
print("B intended score  :", round(score_b, 4))
print("A > B?", score_a > score_b)

for w in ["fol", "age", "as", "had", "y", "foliage", "shady"]:
    print(w, "score=", round(p._word_score(w), 4), "value=", round(p._word_value(w), 4))
