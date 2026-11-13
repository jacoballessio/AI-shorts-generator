"""
Enhanced debug tool to visualize actual clipping issues.
Shows what parts of the caption are being cut off.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def test_clipping_visibility():
    """Create side-by-side comparison to identify clipping."""
    
    test_text = "Ever witnessed a moment so beautiful it felt surreal?"
    
    # Create test with extra padding to see what would be clipped
    extended_height = HEIGHT + 400  # Add 200px top and bottom
    extended_width = WIDTH + 200   # Add 100px left and right
    
    # Create extended canvas with red border
    img = np.zeros((extended_height, extended_width, 3), dtype=np.uint8)
    img[:, :] = [50, 50, 50]  # Gray background
    
    # Mark the actual video boundaries in bright red
    border_size = 100
    # Top border
    img[:border_size, :] = [255, 0, 0]
    # Bottom border  
    img[-border_size:, :] = [255, 0, 0]
    # Left border
    img[:, :border_size] = [255, 0, 0]
    # Right border
    img[:, -border_size:] = [255, 0, 0]
    
    # Mark the safe zone in green (actual video area)
    safe_zone_thickness = 5
    img[border_size:border_size+safe_zone_thickness, border_size:-border_size] = [0, 255, 0]
    img[-border_size-safe_zone_thickness:-border_size, border_size:-border_size] = [0, 255, 0]
    img[border_size:-border_size, border_size:border_size+safe_zone_thickness] = [0, 255, 0]
    img[border_size:-border_size, -border_size-safe_zone_thickness:-border_size] = [0, 255, 0]
    
    background = ImageClip(img).with_duration(1)
    
    # Create caption - positioned in the middle of the extended canvas
    y_pos = (extended_height // 2) - 100
    x_offset = border_size
    
    try:
        txt_clip = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=3,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        txt_clip = txt_clip.with_position((x_offset + 50, y_pos)).with_duration(1)
        
        # Composite
        final = CompositeVideoClip(
            [background, txt_clip],
            size=(extended_width, extended_height)
        )
        
        # Save extended version
        final.save_frame("test_extended_canvas.png", t=0)
        
        print("✅ Extended canvas created: test_extended_canvas.png")
        print(f"   Canvas: {extended_width}x{extended_height}")
        print(f"   Red zone = what gets clipped")
        print(f"   Green box = actual video boundaries")
        print(f"   If text touches red, it will be clipped")
        print()
        
        # Now create the actual cropped version
        # This simulates what happens when CompositeVideoClip crops to video size
        cropped_img = img[border_size:-border_size, border_size:-border_size]
        cropped_bg = ImageClip(cropped_img).with_duration(1)
        
        txt_clip_cropped = TextClip(
            text=test_text,
            font_size=80,
            color='white',
            stroke_color='black',
            stroke_width=3,
            method='caption',
            size=(WIDTH - 100, None)
        )
        
        # Position at 30% from top (current setting in video_composer.py)
        txt_clip_cropped = txt_clip_cropped.with_position(('center', int(HEIGHT * 0.30))).with_duration(1)
        
        final_cropped = CompositeVideoClip(
            [cropped_bg, txt_clip_cropped],
            size=(WIDTH, HEIGHT)
        )
        
        final_cropped.save_frame("test_cropped_actual.png", t=0)
        
        print("✅ Actual cropped version: test_cropped_actual.png")
        print(f"   This shows what the final video looks like")
        print()
        
        # Test without explicit size parameter (what might be causing issues)
        try:
            final_no_size = CompositeVideoClip([cropped_bg, txt_clip_cropped])
            final_no_size.save_frame("test_no_size_param.png", t=0)
            print("✅ No size parameter version: test_no_size_param.png")
            print(f"   Composite size: {final_no_size.w}x{final_no_size.h}")
        except Exception as e:
            print(f"⚠️ No size param test failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clipping_visibility()
    print("\n" + "="*60)
    print("COMPARE THE THREE IMAGES:")
    print("1. test_extended_canvas.png - Shows clipping zones in red")
    print("2. test_cropped_actual.png - What it should look like") 
    print("3. test_no_size_param.png - What might be causing clipping")
    print("="*60)
