#!/usr/bin/env python3
"""Quick test of UI template rendering."""

import re
from ui_template_renderer import render_ui_template

html_content = """<div class="pos-top-left style-white">12:47</div>
<div class="pos-top-left style-white">iMessage</div>
<div class="pos-top-left style-white">@</div>
<div class="pos-top-left style-white">Hi</div>
<div class="pos-bottom style-bold">Hi</div>
<div class="pos-top-left style-white">hi</div>"""

scene_description = "Close-up of person typing on smartphone in coffee shop"

# Parse HTML
ui_elements = []

div_pattern = r'<div(?:\s+class="([^"]*)")?>(.*?)</div>'
div_matches = re.findall(div_pattern, html_content, re.DOTALL)

for classes, content in div_matches:
    content = content.strip()
    if not content:
        continue
    
    # Parse position from classes
    position = 'center'
    if 'pos-top-left' in classes:
        position = 'top-left'
    elif 'pos-bottom' in classes:
        position = 'bottom'
    
    style = 'white' if 'style-white' in classes else 'default'
    
    ui_elements.append({
        'content': content,
        'position': position,
        'style': style,
        'type': 'text'
    })
    
    print(f"Element: {content!r}, Position: {position}, Style: {style}")

# Render
print("\nDetected app: iMessage")
img = render_ui_template(ui_elements, scene_description=scene_description)
img.save('test_template_output.png')
print("✓ Saved: test_template_output.png")
