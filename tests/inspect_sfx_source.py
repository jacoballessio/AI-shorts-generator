import elevenlabs
from elevenlabs.client import ElevenLabs
import inspect

c = ElevenLabs(api_key='dummy')
print('attribute repr:', c.text_to_sound_effects)
print('type:', type(c.text_to_sound_effects))
cls = c.text_to_sound_effects.__class__
print('class name:', cls.__name__)
try:
    src = inspect.getsource(cls)
    print('class source:\n', src[:1000])
except Exception as e:
    print('could not get source:', e)
