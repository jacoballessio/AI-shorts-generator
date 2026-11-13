import inspect, elevenlabs, pkgutil, importlib, os
module = importlib.import_module('elevenlabs')
print('searching in', module.__path__)
for path in module.__path__:
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.py'):
                p=os.path.join(root,f)
                with open(p,'r',encoding='utf-8',errors='ignore') as fh:
                    text=fh.read()
                    if 'text_to_sound_effects' in text:
                        print('found in',p)
