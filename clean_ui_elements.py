"""Clean UI elements from existing analysis by removing scene descriptions."""

import json
from pathlib import Path

def is_invalid_ui_element(elem, scene_desc):
    """Check if UI element is scene description or prompt leakage."""
    content = elem.get('content', '').strip()
    if not content:
        return True
    
    content_lower = content.lower()
    
    # Prompt leakage
    leak_phrases = [
        'describe what', 'provide scene', 'camera angle',
        'main video content', 'not ui overlays', 'if same as previous',
        'same shot', 'exact text visible', 'response format',
        'json only', 'ui elements', 'scene_description',
    ]
    if any(phrase in content_lower for phrase in leak_phrases):
        return True
    
    # Scene description keywords
    scene_keywords = [
        'camera', 'blurry', 'close up', 'selfie', 'looking at',
        'holding', 'wearing', 'with blonde', 'with brown',
        'in a suit', 'in a jacket', 'in a room', 'on keyboard',
        'face of', 'finger on', 'person\'s', 'man with', 'woman with',
        'young man', 'young woman',
    ]
    if any(keyword in content_lower for keyword in scene_keywords):
        return True
    
    # Similar to scene description
    if scene_desc and content_lower in scene_desc.lower():
        return True
    
    # Too long
    if elem.get('type') == 'text' and len(content) > 60:
        return True
    
    return False

def clean_analysis(video_id):
    """Clean UI elements in analysis file."""
    analysis_path = Path("data/analysis") / f"{video_id}_analysis.json"
    
    if not analysis_path.exists():
        print(f"Analysis not found: {analysis_path}")
        return
    
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    total_removed = 0
    total_kept = 0
    
    for segment in analysis.get('segments', []):
        for frame in segment.get('key_frames', []):
            ui_elements = frame.get('ui_elements', [])
            scene_desc = frame.get('description', '')
            
            # Filter
            filtered = []
            for elem in ui_elements:
                if is_invalid_ui_element(elem, scene_desc):
                    total_removed += 1
                    print(f"  Removed: '{elem.get('content', '')[:60]}'")
                else:
                    filtered.append(elem)
                    total_kept += 1
            
            frame['ui_elements'] = filtered
    
    # Save
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Cleaned {video_id}")
    print(f"   Removed: {total_removed} invalid UI elements")
    print(f"   Kept: {total_kept} valid UI elements")

if __name__ == "__main__":
    clean_analysis("sENByrMHfyk")
