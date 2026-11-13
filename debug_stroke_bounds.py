"""
Debug script to test if stroke_width extends beyond TextClip bounds
"""
from moviepy import TextClip, ColorClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1080
HEIGHT = 1920

# Test text that's representative of what we actually use
test_text = "This is a sample caption with multiple words that will wrap"

print("\n" + "="*60)
print("TESTING TEXTCLIP BOUNDS WITH STROKE")
print("="*60)

# Test different configurations
configs = [
    {"name": "No Stroke", "stroke_width": 0},
    {"name": "Stroke 2", "stroke_width": 2},
    {"name": "Stroke 3", "stroke_width": 3},
    {"name": "Stroke 5", "stroke_width": 5},
]

for config in configs:
    print(f"\n{config['name']}:")
    
    # Create TextClip with our actual parameters
    txt_clip = TextClip(
        text=test_text,
        font_size=80,
        color='white',
        stroke_color='black',
        stroke_width=config['stroke_width'],
        font='Arial',
        method='caption',
        size=(WIDTH - 100, None)  # Our actual parameters
    )
    
    reported_width, reported_height = txt_clip.size
    print(f"  Reported size: {reported_width}x{reported_height}")
    
    # Test at different positions
    positions = [
        {"pct": 40, "y": int(HEIGHT * 0.40)},
        {"pct": 35, "y": int(HEIGHT * 0.35)},
        {"pct": 30, "y": int(HEIGHT * 0.30)},
    ]
    
    for pos in positions:
        y_pos = pos['y']
        bottom = y_pos + reported_height
        margin = HEIGHT - bottom
        
        # Estimate actual bottom with stroke (stroke adds ~stroke_width pixels on each side)
        estimated_actual_bottom = bottom + config['stroke_width']
        estimated_margin = HEIGHT - estimated_actual_bottom
        
        status = "✅ SAFE" if estimated_margin > 100 else "⚠️  TIGHT" if estimated_margin > 0 else "🔴 CLIPS"
        
        print(f"  {pos['pct']}% from top (y={y_pos}):")
        print(f"    Reported bottom: {bottom}px, margin: {margin}px")
        print(f"    Estimated actual bottom: {estimated_actual_bottom}px, margin: {estimated_margin}px {status}")

# Now create visual test at 30% from top with stroke=2 (our current config)
print("\n" + "="*60)
print("CREATING VISUAL TEST")
print("="*60)

bg = ColorClip(size=(WIDTH, HEIGHT), color=(20, 20, 20)).with_duration(1)

# Add reference lines
reference_lines = []
for pct in [30, 35, 40]:
    y = int(HEIGHT * (pct / 100))
    line = ColorClip(size=(WIDTH, 2), color=(0, 255, 0)).with_position((0, y)).with_duration(1)
    reference_lines.append(line)
    print(f"  Green line at {pct}% ({y}px)")

# Create caption at 30% from top with stroke=2
txt_clip = TextClip(
    text=test_text,
    font_size=80,
    color='white',
    stroke_color='black',
    stroke_width=2,
    font='Arial',
    method='caption',
    size=(WIDTH - 100, None)
).with_position(('center', int(HEIGHT * 0.30))).with_duration(1)

print(f"  Caption at 30%, size: {txt_clip.size}")

# Bottom boundary line (100px from bottom - safe zone)
safe_line = ColorClip(size=(WIDTH, 3), color=(255, 0, 0)).with_position((0, HEIGHT - 100)).with_duration(1)
print(f"  Red line at safe boundary: {HEIGHT - 100}px")

# Actual bottom line
bottom_line = ColorClip(size=(WIDTH, 5), color=(255, 0, 0)).with_position((0, HEIGHT - 5)).with_duration(1)

# Composite with explicit size
composite = CompositeVideoClip(
    [bg] + reference_lines + [safe_line, bottom_line, txt_clip],
    size=(WIDTH, HEIGHT)
)

# Save frame
frame = composite.get_frame(0)
img = Image.fromarray(frame)
img.save('test_stroke_bounds_30pct.png')
print(f"\n✅ Saved: test_stroke_bounds_30pct.png")

# Also test at 25% to be extra safe
txt_clip_25 = TextClip(
    text=test_text,
    font_size=80,
    color='white',
    stroke_color='black',
    stroke_width=2,
    font='Arial',
    method='caption',
    size=(WIDTH - 100, None)
).with_position(('center', int(HEIGHT * 0.25))).with_duration(1)

composite_25 = CompositeVideoClip(
    [bg, safe_line, bottom_line, txt_clip_25],
    size=(WIDTH, HEIGHT)
)

frame_25 = composite_25.get_frame(0)
img_25 = Image.fromarray(frame_25)
img_25.save('test_stroke_bounds_25pct.png')
print(f"✅ Saved: test_stroke_bounds_25pct.png")

print("\nTest complete! Check the images to see if captions are fully visible.")
