import warnings; warnings.filterwarnings('ignore')
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained('SuperAGI/mistral-7B-PoSE-32k', trust_remote_code=True)

samples = [
    '<segment><start>0.0</start><scene>close-up shot of a pooping emoji</scene></segment>',
    'The video opens with a close-up shot of a pooping emoji from Twitter.',
]

for s in samples:
    ids = t.encode(s, add_special_tokens=False)
    tokens = t.convert_ids_to_tokens(ids)
    decoded = t.decode(ids, skip_special_tokens=True)
    print('INPUT:  ', s)
    print('TOKENS: ', tokens)
    print('DECODED:', repr(decoded))
    print()
