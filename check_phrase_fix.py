from pathlib import Path
import postprocess_xmlish_text as p

p.CUSTOM_WORDS.clear()
p.CUSTOM_WORDS.update(p._load_dictionary_words(Path('words.txt')))
p.WORD_FREQUENCIES.clear()
p.WORD_FREQUENCIES.update(p._load_word_frequencies(Path(r'C:/Users/jacob/Downloads/valid_words_sorted_by_frequency.txt')))
p._invalidate_lexicon_cache()

text = 'with fol i age and as had y figure to his left'
print('in :', text)
print('out:', p._repair_nonword_windows(text))
