# XML-ish to JSON Scraper

Converts raw XML-ish formatted output from the example generator into clean, valid JSON suitable for video generation.

## Usage

```bash
python scrape_xmlish_to_json.py <input_file> [-o <output_file>] [--pretty]
```

### Arguments

- `input`: Path to raw XML-ish file (e.g., `example.xmlish.txt.raw.txt`)
- `-o, --output`: Output JSON file path (default: `{input_name}.json`)
- `--pretty`: Pretty-print JSON output with indentation

### Examples

```bash
# Basic conversion
python scrape_xmlish_to_json.py ./new_example.xmlish.txt.raw.txt

# With custom output path
python scrape_xmlish_to_json.py ./new_example.xmlish.txt.raw.txt -o ./output.json

# Pretty-printed output
python scrape_xmlish_to_json.py ./new_example.xmlish.txt.raw.txt --pretty
```

## Features

### Data Extraction
- Extracts metadata: `creator`, `topic`, `title` from XML tags
- Parses all segments with: `start`, `end`, `duration`, `scene`, `transcript`, `audio`
- Maps XML fields to JSON structure:
  - `scene` → `scene_description`
  - `audio` → `audio_description`
  - `transcript` stays as `transcript`

### Text Cleaning
- **Punctuation spacing**: Fixes spacing around `.!?,;"`
  - `"Hello.World"` → `"Hello. World"`
  - `"Word,another"` → `"Word, another"`  
  - `".The text"` → `". The text"`
- **Quote handling**: Normalizes spacing around quotes
- **Malformed data recovery**: Strips stray characters like `}` from numeric fields
- **Whitespace normalization**: Removes multiple spaces

### Robustness
- Handles malformed XML (missing closing tags)
- Gracefully handles whitespace in numeric fields
- Recovers from common generation artifacts
- Skips invalid segments with warnings

## Output Format

```json
{
  "input": {
    "sources": [],
    "duration": 12.08,
    "creator": "Creator Name",
    "topic": "Topic",
    "title": "Video Title"
  },
  "output": {
    "segments": [
      {
        "start_time": 0.0,
        "end_time": 7.14,
        "duration": 7.14,
        "scene_description": "...",
        "ui_elements_html": "",
        "transcript": "...",
        "audio_description": "..."
      }
    ]
  }
}
```

This JSON is compatible with `generate_video_from_example.py` for rendering videos.

## When to Use

Use this scraper instead of NLP-based cleaning when:
- The raw XML-ish output is mostly good (just has punctuation spacing issues)
- You want fast, deterministic processing
- You prefer to avoid LLM scoring overhead
- Spacing artifacts are minimal and just need basic fixes

## Workflow

```
generate_example_with_carmind2_lora.py
    ↓ (outputs raw XML-ish)
new_example.xmlish.txt.raw.txt
    ↓ (this scraper)
new_example.xmlish.txt.raw.json
    ↓ (feed to video generator)
generate_video_from_example.py
    ↓ (renders video)
output.mp4
```
