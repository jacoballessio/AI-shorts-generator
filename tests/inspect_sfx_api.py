from elevenlabs.client import ElevenLabs
client = ElevenLabs(api_key='dummy')
print('text_to_sound_effects attr:', client.text_to_sound_effects)
print('dir:', [m for m in dir(client.text_to_sound_effects) if not m.startswith('_')])
import inspect
print('signature:', inspect.signature(client.text_to_sound_effects))
try:
    help(client.text_to_sound_effects)
except Exception as e:
    print('help error', e)
