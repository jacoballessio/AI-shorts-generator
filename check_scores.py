from pathlib import Path
import postprocess_xmlish_text as p

p.CUSTOM_WORDS.clear()
p.CUSTOM_WORDS.update(p._load_dictionary_words(Path('words.txt')))
p.WORD_FREQUENCIES.clear()
p.WORD_FREQUENCIES.update(p._load_word_frequencies(Path(r'C:/Users/jacob/Downloads/valid_words_sorted_by_frequency.txt')))
p._invalidate_lexicon_cache()

for w in ['aga', 'in', 'sta', 'clear', 'against', 'a', 'int', 'othe']:
    print(w, round(p._word_score(w), 3), round(p._word_value(w), 3))
