import os, sys, json

# add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.dataset_builder import DatasetBuilder

builder = DatasetBuilder()
analysis = json.load(open('data/analysis/sENByrMHfyk/analysis.json'))
metadata = json.load(open('data/raw_videos/sENByrMHfyk.json'))
labels = json.load(open('data/analysis/sENByrMHfyk/frames_analysis.json'))
sources = []
example = builder.build_training_example(metadata, labels, sources)
with open('data/training_dataset/sENByrMHfyk_example.json', 'w', encoding='utf-8') as f:
    json.dump(example, f, indent=2, ensure_ascii=False)
print('Updated training example with segment-level audio descriptions')