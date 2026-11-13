"""
Test adding extra padding to TextClip size to account for stroke bleeding.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def test_padding_solution():
    """Test if adding padding to the clip fixes the bleeding."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    print("="*70)
    print("TESTING PADDING SOLUTION FOR STROKE BLEED")
    print("="*70)
    
    # Test 1: Current approach (NO padding)
    try:
        txt_no_padding = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        print(f"\n1. NO padding (current):")
        print(f"   Reported size: {txt_no_padding.w}x{txt_no_padding.h}")
        
        # Position at 30% from top
        y_pos = int(HEIGHT * 0.30)
        
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:, :] = [30, 30, 30]
        
        # Mark boundaries
        img_marked = img.copy()
        img_marked[y_pos:y_pos+2, :] = [0, 100, 255]  # Blue = top
        expected_bottom = y_pos + txt_no_padding.h
        img_marked[expected_bottom-2:expected_bottom, :] = [255, 255, 0]  # Yellow = expected bottom
        img_marked[HEIGHT-3:HEIGHT, :] = [255, 0, 0]  # Red = actual bottom boundary
        
        bg = ImageClip(img_marked).with_duration(1)
        txt_positioned = txt_no_padding.with_position(('center', y_pos)).with_duration(1)
        
        final = CompositeVideoClip([bg, txt_positioned], size=(WIDTH, HEIGHT))
        final.save_frame("test_no_padding.png", t=0)
        
        print(f"   Y position: {y_pos}")
        print(f"   Expected bottom: {expected_bottom}")
        print(f"   Margin to video bottom: {HEIGHT - expected_bottom}px")
        print(f"   Saved: test_no_padding.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Add EXTRA height padding (stroke_width * 4 on each side)
    try:
        stroke_padding = 2 * 4  # stroke_width * 4
        
        txt_with_padding = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        # Manually add padding by wrapping in a larger ImageClip
        padded_height = txt_with_padding.h + (stroke_padding * 2)
        padded_width = txt_with_padding.w
        
        print(f"\n2. WITH stroke padding ({stroke_padding}px top+bottom):")
        print(f"   Original size: {txt_with_padding.w}x{txt_with_padding.h}")
        print(f"   Padded size: {padded_width}x{padded_height}")
        
        # Account for padding in positioning
        y_pos2 = int(HEIGHT * 0.30)
        
        img2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img2[:, :] = [30, 30, 30]
        
        # Mark boundaries
        img2_marked = img2.copy()
        img2_marked[y_pos2:y_pos2+2, :] = [0, 100, 255]  # Blue = top
        # Account for padding in expected bottom
        expected_bottom2 = y_pos2 + padded_height
        img2_marked[expected_bottom2-2:expected_bottom2, :] = [255, 255, 0]  # Yellow
        img2_marked[HEIGHT-3:HEIGHT, :] = [255, 0, 0]  # Red = actual boundary
        
        bg2 = ImageClip(img2_marked).with_duration(1)
        # Position with padding offset
        txt_positioned2 = txt_with_padding.with_position(('center', y_pos2 + stroke_padding)).with_duration(1)
        
        final2 = CompositeVideoClip([bg2, txt_positioned2], size=(WIDTH, HEIGHT))
        final2.save_frame("test_with_padding.png", t=0)
        
        print(f"   Y position (with offset): {y_pos2 + stroke_padding}")
        print(f"   Expected bottom: {expected_bottom2}")
        print(f"   Margin to video bottom: {HEIGHT - expected_bottom2}px")
        print(f"   Saved: test_with_padding.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Use margin parameter with extra bottom margin
    try:
        txt_margin = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None),
            margin=(0, 20)  # (horizontal, vertical) - add 20px vertical margin
        )
        
        print(f"\n3. WITH margin=(0, 20):")
        print(f"   Reported size: {txt_margin.w}x{txt_margin.h}")
        
        y_pos3 = int(HEIGHT * 0.30)
        
        img3 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img3[:, :] = [30, 30, 30]
        
        img3_marked = img3.copy()
        img3_marked[y_pos3:y_pos3+2, :] = [0, 100, 255]
        expected_bottom3 = y_pos3 + txt_margin.h
        img3_marked[expected_bottom3-2:expected_bottom3, :] = [255, 255, 0]
        img3_marked[HEIGHT-3:HEIGHT, :] = [255, 0, 0]
        
        bg3 = ImageClip(img3_marked).with_duration(1)
        txt_positioned3 = txt_margin.with_position(('center', y_pos3)).with_duration(1)
        
        final3 = CompositeVideoClip([bg3, txt_positioned3], size=(WIDTH, HEIGHT))
        final3.save_frame("test_margin_padding.png", t=0)
        
        print(f"   Y position: {y_pos3}")
        print(f"   Expected bottom: {expected_bottom3}")
        print(f"   Margin to video bottom: {HEIGHT - expected_bottom3}px")
        print(f"   Saved: test_margin_padding.png")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Move 50px higher to compensate
    try:
        txt_moved = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        # Add extra margin by moving position up
        bleed_compensation = 50  # Compensate for stroke bleed
        y_pos4 = int(HEIGHT * 0.30) - bleed_compensation
        
        print(f"\n4. Moved {bleed_compensation}px HIGHER to compensate:")
        print(f"   Reported size: {txt_moved.w}x{txt_moved.h}")
        
        img4 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img4[:, :] = [30, 30, 30]
        
        img4_marked = img4.copy()
        img4_marked[y_pos4:y_pos4+2, :] = [0, 100, 255]
        expected_bottom4 = y_pos4 + txt_moved.h
        actual_bleed_bottom = expected_bottom4 + bleed_compensation
        img4_marked[expected_bottom4-2:expected_bottom4, :] = [255, 255, 0]  # Expected
        img4_marked[actual_bleed_bottom-2:actual_bleed_bottom, :] = [0, 255, 0]  # With bleed
        img4_marked[HEIGHT-3:HEIGHT, :] = [255, 0, 0]
        
        bg4 = ImageClip(img4_marked).with_duration(1)
        txt_positioned4 = txt_moved.with_position(('center', y_pos4)).with_duration(1)
        
        final4 = CompositeVideoClip([bg4, txt_positioned4], size=(WIDTH, HEIGHT))
        final4.save_frame("test_moved_higher.png", t=0)
        
        print(f"   Y position: {y_pos4} (30% - {bleed_compensation}px)")
        print(f"   Expected bottom: {expected_bottom4}")
        print(f"   With bleed estimate: {actual_bleed_bottom}")
        print(f"   Margin to video bottom: {HEIGHT - actual_bleed_bottom}px")
        print(f"   Saved: test_moved_higher.png")
        print(f"   Green line = estimated actual bottom with bleed")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("CHECK IMAGES:")
    print("  Red line at bottom = video boundary")
    print("  If text reaches red line = clipping")
    print("  Which test keeps text ABOVE red line?")
    print("="*70)

if __name__ == "__main__":
    test_padding_solution()
