"""
Quick verification test with margin=(0, 25) parameter.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def verify_margin_25():
    """Test margin=(0, 25) matches actual video_composer.py settings."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    print("="*70)
    print("VERIFICATION TEST: margin=(0, 25)")
    print("="*70)
    
    try:
        # Exactly as used in video_composer.py _add_animated_captions
        txt_clip = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None),
            margin=(0, 25)
        )
        
        print(f"\nTextClip with margin=(0, 25):")
        print(f"  Reported size: {txt_clip.w}x{txt_clip.h}")
        
        # Position at 30% from top (as in video_composer.py)
        y_pos = int(HEIGHT * 0.30)
        
        # Create background with visual markers
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:, :] = [30, 30, 30]  # Dark gray
        
        # Mark boundaries
        img[0:3, :] = [0, 255, 0]  # Green = top boundary
        img[y_pos:y_pos+3, :] = [0, 100, 255]  # Blue = Y position (30%)
        expected_bottom = y_pos + txt_clip.h
        img[expected_bottom-3:expected_bottom, :] = [255, 255, 0]  # Yellow = expected bottom
        img[HEIGHT-3:HEIGHT, :] = [255, 0, 0]  # Red = video bottom boundary
        
        # Add safe zone marker (100px from bottom)
        safe_zone = HEIGHT - 100
        img[safe_zone:safe_zone+2, :] = [0, 255, 255]  # Cyan = safe zone
        
        bg = ImageClip(img).with_duration(1)
        txt_positioned = txt_clip.with_position(('center', y_pos)).with_duration(1)
        
        final = CompositeVideoClip([bg, txt_positioned], size=(WIDTH, HEIGHT))
        final.save_frame("verify_margin_25.png", t=0)
        
        # Calculate margins
        bottom_margin = HEIGHT - expected_bottom
        
        print(f"\nPositioning:")
        print(f"  Y position: {y_pos}px (30% from top)")
        print(f"  Expected bottom: {expected_bottom}px")
        print(f"  Bottom margin: {bottom_margin}px")
        print(f"  Safe zone at: {safe_zone}px")
        
        print(f"\nSafety check:")
        if expected_bottom < safe_zone:
            print(f"  ✅ Caption bottom ({expected_bottom}px) is ABOVE safe zone ({safe_zone}px)")
            print(f"     Margin: {safe_zone - expected_bottom}px clearance")
        else:
            print(f"  ⚠️ Caption bottom ({expected_bottom}px) extends below safe zone")
        
        if expected_bottom < HEIGHT:
            print(f"  ✅ Caption fits within video frame")
        else:
            print(f"  ❌ Caption EXCEEDS video frame by {expected_bottom - HEIGHT}px")
        
        print(f"\nSaved: verify_margin_25.png")
        print(f"\nVisual markers:")
        print(f"  Green (top):    Video top boundary")
        print(f"  Blue:           Caption Y position (30% from top)")
        print(f"  Yellow:         Caption bottom edge (where TextClip says it ends)")
        print(f"  Cyan:           Safe zone (100px from bottom)")
        print(f"  Red (bottom):   Video bottom boundary")
        print(f"\n  CHECK: Text should NOT extend beyond yellow line!")
        print(f"  CHECK: Yellow line should be ABOVE cyan safe zone!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*70)

if __name__ == "__main__":
    verify_margin_25()
