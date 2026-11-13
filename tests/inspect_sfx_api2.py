import inspect
from elevenlabs.client import ElevenLabs
print('class methods:', [m for m in dir(ElevenLabs) if not m.startswith('_')])
print('text_to_sound_effects attribute:', ElevenLabs.__dict__.get('text_to_sound_effects'))
