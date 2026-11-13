import pkgutil
print([m.name for m in pkgutil.iter_modules() if 'qwen' in m.name.lower()])
