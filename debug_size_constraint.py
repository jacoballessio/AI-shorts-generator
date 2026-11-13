"""
Test if removing size constraint prevents clipping.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def test_no_size_constraint():
    """Test TextClip without size constraint."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    print("="*70)
    print("TESTING WITHOUT SIZE CONSTRAINT")
    print("="*70)
    
    # Test 1: With size constraint (current approach)
    try:
        txt_constrained = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None)  # WIDTH constraint
        )
        
        print(f"\n1. WITH size constraint (WIDTH-100, None):")
        print(f"   Reported size: {txt_constrained.w}x{txt_constrained.h}")
        
        y_pos = int(HEIGHT * 0.30)
        
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:, :] = [30, 30, 30]
        img[y_pos:y_pos+2, :] = [0, 100, 255]
        expected_bottom = y_pos + txt_constrained.h
        img[expected_bottom-2:expected_bottom, :] = [255, 255, 0]
        img[HEIGHT-3:HEIGHT, :] = [255, 0, 0]
        
        bg = ImageClip(img).with_duration(1)
        txt_positioned = txt_constrained.with_position(('center', y_pos)).with_duration(1)
        
        final = CompositeVideoClip([bg, txt_positioned], size=(WIDTH, HEIGHT))
        final.save_frame("test_with_size_constraint.png", t=0)
        print(f"   Saved: test_with_size_constraint.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: WITHOUT size constraint - use method='label'
    try:
        txt_no_constraint = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='label'  # Doesn't require size
        )
        
        print(f"\n2. WITHOUT size constraint (method='label'):")
        print(f"   Reported size: {txt_no_constraint.w}x{txt_no_constraint.h}")
        
        y_pos2 = int(HEIGHT * 0.30)
        
        img2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img2[:, :] = [30, 30, 30]
        img2[y_pos2:y_pos2+2, :] = [0, 100, 255]
        expected_bottom2 = y_pos2 + txt_no_constraint.h
        img2[expected_bottom2-2:expected_bottom2, :] = [255, 255, 0]
        img2[HEIGHT-3:HEIGHT, :] = [255, 0, 0]
        
        bg2 = ImageClip(img2).with_duration(1)
        txt_positioned2 = txt_no_constraint.with_position(('center', y_pos2)).with_duration(1)
        
        final2 = CompositeVideoClip([bg2, txt_positioned2], size=(WIDTH, HEIGHT))
        final2.save_frame("test_no_size_constraint.png", t=0)
        print(f"   Saved: test_no_size_constraint.png")
        print(f"   Note: Will be single line (very wide)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Larger size constraint to allow wrapping but not cropping
    try:
        txt_larger = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, HEIGHT)  # Add HEIGHT constraint too
        )
        
        print(f"\n3. LARGER size constraint (WIDTH-100, HEIGHT):")
        print(f"   Reported size: {txt_larger.w}x{txt_larger.h}")
        
        y_pos3 = int(HEIGHT * 0.30)
        
        img3 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img3[:, :] = [30, 30, 30]
        img3[y_pos3:y_pos3+2, :] = [0, 100, 255]
        expected_bottom3 = y_pos3 + txt_larger.h
        img3[expected_bottom3-2:expected_bottom3, :] = [255, 255, 0]
        img3[HEIGHT-3:HEIGHT, :] = [255, 0, 0]
        
        bg3 = ImageClip(img3).with_duration(1)
        txt_positioned3 = txt_larger.with_position(('center', y_pos3)).with_duration(1)
        
        final3 = CompositeVideoClip([bg3, txt_positioned3], size=(WIDTH, HEIGHT))
        final3.save_frame("test_larger_constraint.png", t=0)
        print(f"   Saved: test_larger_constraint.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: No method parameter (use default)
    try:
        txt_default_method = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2
        )
        
        print(f"\n4. Default method (no explicit method parameter):")
        print(f"   Reported size: {txt_default_method.w}x{txt_default_method.h}")
        
        y_pos4 = int(HEIGHT * 0.30)
        
        img4 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img4[:, :] = [30, 30, 30]
        img4[y_pos4:y_pos4+2, :] = [0, 100, 255]
        expected_bottom4 = y_pos4 + txt_default_method.h
        img4[expected_bottom4-2:expected_bottom4, :] = [255, 255, 0]
        img4[HEIGHT-3:HEIGHT, :] = [255, 0, 0]
        
        bg4 = ImageClip(img4).with_duration(1)
        txt_positioned4 = txt_default_method.with_position(('center', y_pos4)).with_duration(1)
        
        final4 = CompositeVideoClip([bg4, txt_positioned4], size=(WIDTH, HEIGHT))
        final4.save_frame("test_default_method.png", t=0)
        print(f"   Saved: test_default_method.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("LOOK FOR: Which image has NO clipping at the text's bottom edge?")
    print("="*70)

if __name__ == "__main__":
    test_no_size_constraint()
