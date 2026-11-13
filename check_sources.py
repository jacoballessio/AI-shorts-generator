import json

with open('data/sources/7P3T-NnyPhw_sources.json', 'r', encoding='utf-8') as f:
    sources = json.load(f)

print("Sources and citations:")
print("=" * 80)
for source in sources[:4]:
    print(f"Title: {source['title']}")
    print(f"Citation: {source['citation']}")
    print(f"Text preview: {source['text'][:100]}...")
    print("-" * 80)
