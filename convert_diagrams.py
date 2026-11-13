"""
Convert Mermaid diagrams to PNG images
Extracts mermaid code blocks from markdown and converts them to PNG
"""

import re
import requests
import base64
import time
from pathlib import Path

def extract_mermaid_diagrams(markdown_file):
    """Extract all mermaid code blocks from markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    diagrams = re.findall(pattern, content, re.DOTALL)
    
    return diagrams

def convert_mermaid_to_png(mermaid_code, output_path):
    """Convert mermaid diagram to PNG using mermaid.ink API."""
    try:
        # Encode mermaid code to base64
        encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        
        # Use mermaid.ink free API
        url = f"https://mermaid.ink/img/{encoded}"
        
        print(f"  Downloading from mermaid.ink...")
        response = requests.get(url, timeout=60)  # Increased timeout
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            file_size = len(response.content) / 1024  # KB
            print(f"  ✅ Saved: {output_path} ({file_size:.1f} KB)")
            return True
        else:
            print(f"  ❌ Error: HTTP {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"  ⏱️  Timeout - trying alternative method...")
        # Try SVG endpoint instead (smaller)
        try:
            svg_url = f"https://mermaid.ink/svg/{encoded}"
            response = requests.get(svg_url, timeout=60)
            if response.status_code == 200:
                # Save as SVG instead
                svg_path = output_path.with_suffix('.svg')
                with open(svg_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✅ Saved as SVG: {svg_path}")
                return True
        except Exception as e2:
            print(f"  ❌ SVG fallback failed: {e2}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    markdown_file = "YOUTUBER_TRAINING_PIPELINE.md"
    output_dir = Path("diagrams")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Extracting mermaid diagrams from {markdown_file}...")
    diagrams = extract_mermaid_diagrams(markdown_file)
    
    print(f"Found {len(diagrams)} diagrams\n")
    
    diagram_names = [
        "1_training_pipeline",
        "2_training_example",
        "3_source_detective",
        "4_implementation_roadmap"
    ]
    
    for i, (diagram, name) in enumerate(zip(diagrams, diagram_names), 1):
        output_path = output_dir / f"{name}.png"
        
        # Skip if already exists
        if output_path.exists():
            print(f"[{i}/{len(diagrams)}] {name} - ✅ Already exists, skipping")
            continue
        
        print(f"[{i}/{len(diagrams)}] Converting {name}...")
        success = convert_mermaid_to_png(diagram, output_path)
        
        if not success:
            print(f"  ⚠️  Failed to convert {name}, continuing...")
        
        # Add delay to avoid rate limiting
        if i < len(diagrams):
            time.sleep(2)
        print()
    
    print(f"\n✅ All diagrams saved to {output_dir}/")
    print("\nGenerated files:")
    for png in sorted(output_dir.glob("*.png")):
        print(f"  - {png}")

if __name__ == "__main__":
    main()
