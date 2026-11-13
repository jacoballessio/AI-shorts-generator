import inspect
try:
    import elevenlabs
    print('elevenlabs imported, dir:')
    print(dir(elevenlabs))
    from elevenlabs.client import ElevenLabs
    print('ElevenLabs client methods:')
    print([m for m in dir(ElevenLabs) if not m.startswith('_')])
except Exception as e:
    print('import error', e)
