import inspect
from elevenlabs.text_to_sound_effects.client import TextToSoundEffectsClient

print('methods:', [m for m in dir(TextToSoundEffectsClient) if not m.startswith('_')])
print('convert signature:', inspect.signature(TextToSoundEffectsClient.convert))
print('convert source:')
print(inspect.getsource(TextToSoundEffectsClient.convert))
