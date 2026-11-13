"""
Debug script to investigate if method='caption' adds hidden padding/margins
that extend beyond the reported clip size.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def test_caption_method_padding():
    """Test different TextClip methods to see actual vs reported sizes."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    print("="*70)
    print("TESTING TEXTCLIP METHOD PADDING")
    print("="*70)
    
    # Test 1: method='caption' (current approach)
    try:
        txt_caption = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        print(f"\n1. method='caption':")
        print(f"   Reported size: {txt_caption.w}x{txt_caption.h}")
        
        # Create on a canvas to see actual rendered bounds
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[:, :] = [30, 30, 30]
        bg = ImageClip(img).with_duration(1)
        
        # Position at center vertically
        y_pos = (HEIGHT - txt_caption.h) // 2
        txt_positioned = txt_caption.with_position(('center', y_pos)).with_duration(1)
        
        # Mark the expected boundaries based on reported size
        img_with_markers = img.copy()
        # Blue lines at expected top/bottom
        img_with_markers[y_pos:y_pos+2, :] = [0, 100, 255]
        expected_bottom = y_pos + txt_caption.h
        img_with_markers[expected_bottom-2:expected_bottom, :] = [255, 255, 0]
        
        bg_marked = ImageClip(img_with_markers).with_duration(1)
        
        final = CompositeVideoClip(
            [bg_marked, txt_positioned],
            size=(WIDTH, HEIGHT)
        )
        
        final.save_frame("test_method_caption.png", t=0)
        print(f"   Saved: test_method_caption.png")
        print(f"   Blue line = Expected top (Y={y_pos})")
        print(f"   Yellow line = Expected bottom (Y={expected_bottom})")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: method='label' (alternative)
    try:
        txt_label = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='label'
        )
        
        print(f"\n2. method='label':")
        print(f"   Reported size: {txt_label.w}x{txt_label.h}")
        
        img2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img2[:, :] = [30, 30, 30]
        
        y_pos2 = (HEIGHT - txt_label.h) // 2
        txt_positioned2 = txt_label.with_position(('center', y_pos2)).with_duration(1)
        
        img2_with_markers = img2.copy()
        img2_with_markers[y_pos2:y_pos2+2, :] = [0, 100, 255]
        expected_bottom2 = y_pos2 + txt_label.h
        img2_with_markers[expected_bottom2-2:expected_bottom2, :] = [255, 255, 0]
        
        bg_marked2 = ImageClip(img2_with_markers).with_duration(1)
        
        final2 = CompositeVideoClip(
            [bg_marked2, txt_positioned2],
            size=(WIDTH, HEIGHT)
        )
        
        final2.save_frame("test_method_label.png", t=0)
        print(f"   Saved: test_method_label.png")
        print(f"   Blue line = Expected top (Y={y_pos2})")
        print(f"   Yellow line = Expected bottom (Y={expected_bottom2})")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: method='caption' with explicit interline parameter
    try:
        txt_interline = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(WIDTH - 100, None),
            interline=-20  # Try negative to reduce padding
        )
        
        print(f"\n3. method='caption' with interline=-20:")
        print(f"   Reported size: {txt_interline.w}x{txt_interline.h}")
        
        img3 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img3[:, :] = [30, 30, 30]
        
        y_pos3 = (HEIGHT - txt_interline.h) // 2
        txt_positioned3 = txt_interline.with_position(('center', y_pos3)).with_duration(1)
        
        img3_with_markers = img3.copy()
        img3_with_markers[y_pos3:y_pos3+2, :] = [0, 100, 255]
        expected_bottom3 = y_pos3 + txt_interline.h
        img3_with_markers[expected_bottom3-2:expected_bottom3, :] = [255, 255, 0]
        
        bg_marked3 = ImageClip(img3_with_markers).with_duration(1)
        
        final3 = CompositeVideoClip(
            [bg_marked3, txt_positioned3],
            size=(WIDTH, HEIGHT)
        )
        
        final3.save_frame("test_method_interline.png", t=0)
        print(f"   Saved: test_method_interline.png")
        print(f"   Blue line = Expected top (Y={y_pos3})")
        print(f"   Yellow line = Expected bottom (Y={expected_bottom3})")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Without stroke to isolate the issue
    try:
        txt_no_stroke = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        print(f"\n4. method='caption' WITHOUT stroke:")
        print(f"   Reported size: {txt_no_stroke.w}x{txt_no_stroke.h}")
        
        img4 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img4[:, :] = [30, 30, 30]
        
        y_pos4 = (HEIGHT - txt_no_stroke.h) // 2
        txt_positioned4 = txt_no_stroke.with_position(('center', y_pos4)).with_duration(1)
        
        img4_with_markers = img4.copy()
        img4_with_markers[y_pos4:y_pos4+2, :] = [0, 100, 255]
        expected_bottom4 = y_pos4 + txt_no_stroke.h
        img4_with_markers[expected_bottom4-2:expected_bottom4, :] = [255, 255, 0]
        
        bg_marked4 = ImageClip(img4_with_markers).with_duration(1)
        
        final4 = CompositeVideoClip(
            [bg_marked4, txt_positioned4],
            size=(WIDTH, HEIGHT)
        )
        
        final4.save_frame("test_no_stroke.png", t=0)
        print(f"   Saved: test_no_stroke.png")
        print(f"   Blue line = Expected top (Y={y_pos4})")
        print(f"   Yellow line = Expected bottom (Y={expected_bottom4})")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS:")
    print("  If text extends BEYOND the yellow line = hidden padding/margins")
    print("  Compare all 4 images to identify which setting causes overflow")
    print("="*70)

if __name__ == "__main__":
    test_caption_method_padding()
