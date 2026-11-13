#!/usr/bin/env python3
"""
Convert raw XML-ish format to clean JSON format.
Extracts metadata and segments, fixes punctuation spacing.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def fix_punctuation_spacing(text: str) -> str:
    """Fix spacing around punctuation marks.
    
    Examples:
        "Hello . World" -> "Hello. World"
        "Word , another" -> "Word, another"
        ".The" -> ". The"
        "saying\"Sorry" -> 'saying "Sorry'
    """
    if not text:
        return text
    
    # Fix space before punctuation (remove extra spaces)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Add space after period/question mark/exclamation followed by capital letter (if missing)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Fix space after other punctuation followed by lowercase (common in generated text)
    text = re.sub(r'([.!?])([a-z])', r'\1 \2', text)
    
    # Fix space after comma if missing
    text = re.sub(r',([A-Za-z])', r', \1', text)
    
    # Fix space before/after quotes
    text = re.sub(r'\s*"\s*', ' "', text)
    text = re.sub(r'"\s+([.,:!?])', r'"\1', text)
    text = re.sub(r'"([A-Za-z])', r'" \1', text)
    
    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix ]. and ][ patterns
    text = re.sub(r'\]\s*\[', '][', text)
    
    return text.strip()


def extract_xml_value(text: str, tag: str) -> Optional[str]:
    """Extract value from XML-ish tag."""
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def parse_xmlish_file(filepath: str) -> Dict:
    """Parse raw XML-ish file and extract metadata + segments."""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata fields directly (not wrapped in <input> tag)
    creator = extract_xml_value(content, 'creator') or 'Unknown Creator'
    topic = extract_xml_value(content, 'topic') or 'General'
    title = extract_xml_value(content, 'title')
    
    # Fallback title if not provided
    if not title or title.strip() == '':
        title = f"{topic} Short"
    
    # Extract output section (segments) - might not have closing tag
    output_start = content.find('<output>')
    if output_start == -1:
        raise ValueError("No <output> section found in XML file")
    
    output_section = content[output_start:]
    
    # Extract all segments
    segment_pattern = r'<segment\s*>(.*?)</segment>'
    segment_matches = re.finditer(segment_pattern, output_section, re.DOTALL)
    
    segments = []
    for i, match in enumerate(segment_matches):
        segment_text = match.group(1)
        
        # Extract segment fields
        start = extract_xml_value(segment_text, 'start')
        end = extract_xml_value(segment_text, 'end')
        duration = extract_xml_value(segment_text, 'duration')
        scene = extract_xml_value(segment_text, 'scene') or ''
        transcript = extract_xml_value(segment_text, 'transcript') or ''
        audio = extract_xml_value(segment_text, 'audio') or ''
        ui_html = extract_xml_value(segment_text, 'ui_elements_html') or ''
        
        # Handle whitespace and malformed characters in start/end/duration
        try:
            # Clean up stray characters like }
            start_clean = (start or '').replace('}', '').strip() if start else ''
            end_clean = (end or '').replace('}', '').strip() if end else ''
            duration_clean = (duration or '').replace('}', '').strip() if duration else ''
            
            start_time = float(start_clean) if start_clean else 0.0
            end_time = float(end_clean) if end_clean else 0.0
            duration_val = float(duration_clean) if duration_clean else (end_time - start_time)
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse segment {i}: start={repr(start)}, end={repr(end)}, duration={repr(duration)}", file=sys.stderr)
            continue
        
        # Fix punctuation spacing in text fields
        scene = fix_punctuation_spacing(scene.strip())
        transcript = fix_punctuation_spacing(transcript.strip())
        audio = fix_punctuation_spacing(audio.strip())
        
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration_val,
            'scene_description': scene,
            'ui_elements_html': ui_html,
            'transcript': transcript,
            'audio_description': audio
        })
    
    if not segments:
        raise ValueError("No segments found in XML file")
    
    # Build output structure matching training JSON format
    return {
        'input': {
            'sources': [],
            'duration': segments[-1]['end_time'] if segments else 0,
            'creator': creator.strip(),
            'topic': topic.strip(),
            'title': title.strip()
        },
        'output': {
            'segments': segments
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert raw XML-ish format to clean JSON'
    )
    parser.add_argument('input', help='Path to raw XML-ish file')
    parser.add_argument('-o', '--output', help='Path to output JSON file (default: input.json)', default=None)
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}.json"
    
    try:
        print(f"Parsing XML-ish file: {input_path}")
        data = parse_xmlish_file(str(input_path))
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            if args.pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        print(f"✓ Successfully converted to JSON: {output_path}".encode('utf-8', errors='replace').decode('utf-8'))
        print(f"  Metadata: {data['input']['creator']} - {data['input']['title']}")
        print(f"  Segments: {len(data['output']['segments'])}")
        print(f"  Duration: {data['input']['duration']}s")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
