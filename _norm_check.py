import pathlib
import re
from generate_example_with_carmind2_lora import _prepare_xmlish_candidate, _extract_segment_stream, build_xmlish_prefix_prompt
raw = pathlib.Path(r'z:/Projects/AI-shorts-generator/new_example.xmlish.txt.raw.txt').read_text(encoding='utf-8')
prefix = build_xmlish_prefix_prompt('Daniel Hentschel', 'Education', 42, [], 'Why is it so hard?')
candidate = _prepare_xmlish_candidate(prefix, raw)
stream = _extract_segment_stream(candidate)
print('RAW_GT', raw.count('>>>>'))
print('STREAM_GT', stream.count('>>>>'))
print('STREAM_BAD_SPEAKER', 'SPEAKER B' in stream or 'SPEAKER C' in stream or 'SPEAKER D' in stream or 'SPEAKER_0' in stream)
print('STREAM_T_CHEVRON', 't> <segment' in stream or '</segment>t>' in stream)
print('STREAM_JOINED', '</segment><segment>' in stream)
