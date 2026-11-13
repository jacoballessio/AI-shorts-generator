from pathlib import Path
import postprocess_xmlish_text as p

p.CUSTOM_WORDS.clear()
p.CUSTOM_WORDS.update(p._load_dictionary_words(Path('words.txt')))
p.WORD_FREQUENCIES.clear()
p.WORD_FREQUENCIES.update(p._load_word_frequencies(Path(r'C:/Users/jacob/Downloads/valid_words_sorted_by_frequency.txt')))
p._invalidate_lexicon_cache()
lex = p._dictionary_lexicon()
max_len = max((len(w) for w in lex), default=1)

compact = 'withfoliageandashadyfiguretohisleft'
seg, cov = p._segment_string_by_dictionary(compact, lex, max_len)
print(seg)
print(cov, len(compact), cov/len(compact))
print('score_new', p._sequence_value(seg, lex)-len(seg)*p.TOKEN_SPLIT_PENALTY)
old = ['with','fol','i','age','and','as','had','y','figure','to','his','left']
print('score_old', p._sequence_value(old, lex)-len(old)*p.TOKEN_SPLIT_PENALTY)
