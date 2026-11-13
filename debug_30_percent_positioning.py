"""
Debug script to verify 30% caption positioning has adequate margins.
Shows exact measurements and safe zones.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def create_measurement_visualization():
    """Create debug image with measurements and markers."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    # Create background
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :] = [30, 30, 30]  # Dark gray background
    
    # Calculate caption position (30% from top)
    y_position = int(HEIGHT * 0.30)
    
    # Create the caption to get its actual size
    txt_clip = TextClip(
        text=test_text,
        font_size=80,
        color='white',
        stroke_color='black',
        stroke_width=2,  # Current setting in video_composer.py
        method='caption',
        size=(WIDTH - 100, None)
    )
    
    caption_height = txt_clip.h
    caption_width = txt_clip.w
    caption_bottom = y_position + caption_height
    
    # Draw measurement markers
    # Top boundary (green)
    img[0:3, :] = [0, 255, 0]
    
    # Caption Y position marker (blue)
    if y_position >= 0 and y_position < HEIGHT:
        img[max(0, y_position-2):min(HEIGHT, y_position+2), :] = [0, 100, 255]
    
    # Caption bottom marker (yellow)
    if caption_bottom >= 0 and caption_bottom < HEIGHT:
        img[max(0, caption_bottom-2):min(HEIGHT, caption_bottom+2), :] = [255, 255, 0]
    
    # Bottom boundary (red if clipping, green if safe)
    bottom_margin = HEIGHT - caption_bottom
    if bottom_margin < 0:
        img[-3:, :] = [255, 0, 0]  # Red = clipping!
    else:
        img[-3:, :] = [0, 255, 0]  # Green = safe
    
    # Draw safe zone markers (recommended minimum margins)
    safe_margin_top = 100
    safe_margin_bottom = 100
    
    # Top safe zone line (cyan)
    img[safe_margin_top-1:safe_margin_top+1, :] = [0, 255, 255]
    
    # Bottom safe zone line (cyan)
    img[HEIGHT-safe_margin_bottom-1:HEIGHT-safe_margin_bottom+1, :] = [0, 255, 255]
    
    background = ImageClip(img).with_duration(1)
    
    # Position the caption at 30% from top
    txt_clip = txt_clip.with_position(('center', y_position)).with_duration(1)
    
    # Composite with explicit size
    final = CompositeVideoClip(
        [background, txt_clip],
        size=(WIDTH, HEIGHT)
    )
    
    # Save
    final.save_frame("debug_30_percent.png", t=0)
    
    # Print detailed measurements
    print("="*70)
    print("CAPTION POSITIONING ANALYSIS (30% from top)")
    print("="*70)
    print(f"Video dimensions:        {WIDTH}x{HEIGHT}")
    print(f"Caption Y position:      {y_position}px (30% from top)")
    print(f"Caption dimensions:      {caption_width}x{caption_height}px")
    print(f"Caption bottom edge:     {caption_bottom}px")
    print()
    print(f"Top margin:              {y_position}px")
    print(f"Bottom margin:           {bottom_margin}px")
    print()
    
    # Safety analysis
    print("SAFETY ANALYSIS:")
    print(f"  Recommended safe margins: {safe_margin_top}px top, {safe_margin_bottom}px bottom")
    
    if y_position >= safe_margin_top:
        print(f"  ✅ Top margin ({y_position}px) is SAFE")
    else:
        print(f"  ⚠️ Top margin ({y_position}px) is below recommended ({safe_margin_top}px)")
    
    if bottom_margin >= safe_margin_bottom:
        print(f"  ✅ Bottom margin ({bottom_margin}px) is SAFE")
    else:
        print(f"  ⚠️ Bottom margin ({bottom_margin}px) is below recommended ({safe_margin_bottom}px)")
    
    if caption_bottom < HEIGHT:
        print(f"  ✅ Caption fits within video frame")
    else:
        print(f"  ❌ Caption EXCEEDS video frame by {caption_bottom - HEIGHT}px")
    
    print()
    print("VISUAL MARKERS in debug_30_percent.png:")
    print("  Green line (top):        Video top boundary")
    print("  Blue line:               Caption Y position (30% from top)")
    print("  Yellow line:             Caption bottom edge")
    print("  Cyan lines:              Recommended safe zone boundaries")
    print("  Bottom line:             Green = safe | Red = clipping")
    print("="*70)

if __name__ == "__main__":
    create_measurement_visualization()
