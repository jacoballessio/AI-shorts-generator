"""
Test if the margin parameter is causing the clipping issue.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def test_margin_parameter():
    """Test different margin settings."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    print("="*70)
    print("TESTING MARGIN PARAMETER")
    print("="*70)
    
    # Test 1: Default margin (None, None)
    try:
        txt_default = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        print(f"\n1. Default margin:")
        print(f"   Reported size: {txt_default.w}x{txt_default.h}")
        
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:, :] = [30, 30, 30]
        
        y_pos = (HEIGHT - txt_default.h) // 2
        txt_positioned = txt_default.with_position(('center', y_pos)).with_duration(1)
        
        img_with_markers = img.copy()
        img_with_markers[y_pos:y_pos+2, :] = [0, 100, 255]
        expected_bottom = y_pos + txt_default.h
        img_with_markers[expected_bottom-2:expected_bottom, :] = [255, 255, 0]
        
        bg_marked = ImageClip(img_with_markers).with_duration(1)
        
        final = CompositeVideoClip(
            [bg_marked, txt_positioned],
            size=(WIDTH, HEIGHT)
        )
        
        final.save_frame("test_margin_default.png", t=0)
        print(f"   Saved: test_margin_default.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: margin=(0, 0)
    try:
        txt_zero = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None),
            margin=(0, 0)
        )
        
        print(f"\n2. margin=(0, 0):")
        print(f"   Reported size: {txt_zero.w}x{txt_zero.h}")
        
        img2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img2[:, :] = [30, 30, 30]
        
        y_pos2 = (HEIGHT - txt_zero.h) // 2
        txt_positioned2 = txt_zero.with_position(('center', y_pos2)).with_duration(1)
        
        img2_with_markers = img2.copy()
        img2_with_markers[y_pos2:y_pos2+2, :] = [0, 100, 255]
        expected_bottom2 = y_pos2 + txt_zero.h
        img2_with_markers[expected_bottom2-2:expected_bottom2, :] = [255, 255, 0]
        
        bg_marked2 = ImageClip(img2_with_markers).with_duration(1)
        
        final2 = CompositeVideoClip(
            [bg_marked2, txt_positioned2],
            size=(WIDTH, HEIGHT)
        )
        
        final2.save_frame("test_margin_zero.png", t=0)
        print(f"   Saved: test_margin_zero.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Combine margin=(0,0) + interline=-10
    try:
        txt_combined = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None),
            margin=(0, 0),
            interline=-10
        )
        
        print(f"\n3. margin=(0, 0) + interline=-10:")
        print(f"   Reported size: {txt_combined.w}x{txt_combined.h}")
        
        img3 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img3[:, :] = [30, 30, 30]
        
        y_pos3 = (HEIGHT - txt_combined.h) // 2
        txt_positioned3 = txt_combined.with_position(('center', y_pos3)).with_duration(1)
        
        img3_with_markers = img3.copy()
        img3_with_markers[y_pos3:y_pos3+2, :] = [0, 100, 255]
        expected_bottom3 = y_pos3 + txt_combined.h
        img3_with_markers[expected_bottom3-2:expected_bottom3, :] = [255, 255, 0]
        
        bg_marked3 = ImageClip(img3_with_markers).with_duration(1)
        
        final3 = CompositeVideoClip(
            [bg_marked3, txt_positioned3],
            size=(WIDTH, HEIGHT)
        )
        
        final3.save_frame("test_margin_zero_interline.png", t=0)
        print(f"   Saved: test_margin_zero_interline.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("Check images - text should NOT extend beyond yellow line")
    print("="*70)

if __name__ == "__main__":
    test_margin_parameter()
