"""Fix malformed scene_description JSON in analysis files."""

import json
import re
from pathlib import Path

def extract_from_malformed_json(json_str: str):
    """Extract scene_description and ui_elements from malformed JSON string."""
    scene_desc = ""
    ui_elements = []
    
    # Extract scene_description
    scene_match = re.search(r'"scene_description"\s*:\s*"([^"]*)"', json_str)
    if scene_match:
        scene_desc = scene_match.group(1)
    
    # Extract ui_elements array
    ui_match = re.search(r'"ui_elements"\s*:\s*\[(.*?)\]\s*(?:\]|$)', json_str, re.DOTALL)
    if ui_match:
        ui_array_str = '[' + ui_match.group(1) + ']'
        try:
            # Clean up common issues
            ui_array_str = re.sub(r'"\w\n\s+"', '"', ui_array_str)  # Fix broken keys like "t\n            "type"
            ui_array_str = re.sub(r',\s*\]', ']', ui_array_str)  # Trailing commas
            ui_parsed = json.loads(ui_array_str)
            if isinstance(ui_parsed, list):
                ui_elements = ui_parsed
        except Exception as e:
            print(f"    Failed to parse UI elements: {e}")
    
    return scene_desc, ui_elements

def fix_analysis_file(video_id: str):
    """Fix malformed JSON in an analysis file."""
    base_dir = Path("data")
    analysis_path = base_dir / "analysis" / f"{video_id}_analysis.json"
    
    if not analysis_path.exists():
        print(f"Analysis file not found: {analysis_path}")
        return
    
    print(f"Fixing {analysis_path}...")
    
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    fixed_count = 0
    total_frames = 0
    
    for segment in analysis.get('segments', []):
        for frame in segment.get('key_frames', []):
            total_frames += 1
            description = frame.get('description', '')
            
            # Check if description contains JSON
            if description.strip().startswith('{') and 'scene_description' in description:
                # Extract clean values
                scene_desc, ui_elements = extract_from_malformed_json(description)
                
                if scene_desc:
                    frame['description'] = scene_desc
                    fixed_count += 1
                    print(f"  Fixed frame {frame.get('frame_number', '?')}: '{scene_desc}'")
                
                # Only override ui_elements if we successfully extracted some
                if ui_elements:
                    frame['ui_elements'] = ui_elements
                    print(f"    Extracted {len(ui_elements)} UI elements")
    
    # Save fixed analysis
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Fixed {fixed_count}/{total_frames} frames")
    print(f"Saved: {analysis_path}")

if __name__ == "__main__":
    fix_analysis_file("sENByrMHfyk")
