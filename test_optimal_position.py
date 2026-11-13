"""
Test different margin values and positions to find optimal settings.
Goal: Captions close to bottom but NO clipping.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np

WIDTH = 1080
HEIGHT = 1920

def test_positions_and_margins():
    """Test various combinations to find sweet spot."""
    
    # Test with different text lengths to find worst case
    test_texts = [
        "Ever witnessed a moment so beautiful it felt surreal?",  # 2 lines
        "This is getting more and more interesting as we continue talking",  # Potentially 3 lines
    ]
    
    configs = [
        {"position": 0.60, "margin": 30, "desc": "60% from top, margin=30"},
        {"position": 0.65, "margin": 30, "desc": "65% from top, margin=30"},
        {"position": 0.65, "margin": 35, "desc": "65% from top, margin=35"},
        {"position": 0.70, "margin": 35, "desc": "70% from top, margin=35"},
    ]
    
    print("="*70)
    print("TESTING POSITIONS AND MARGINS")
    print("="*70)
    
    for idx, config in enumerate(configs):
        for text_idx, test_text in enumerate(test_texts):
            try:
                txt_clip = TextClip(
                    text=test_text,
                    font_size=80,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    method='caption',
                    size=(WIDTH - 100, None),
                    margin=(0, config["margin"])
                )
                
                y_pos = int(HEIGHT * config["position"])
                expected_bottom = y_pos + txt_clip.h
                bottom_margin = HEIGHT - expected_bottom
                
                # Create visual
                img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                img[:, :] = [30, 30, 30]
                
                img[y_pos:y_pos+2, :] = [0, 100, 255]  # Blue = Y position
                img[expected_bottom-2:expected_bottom, :] = [255, 255, 0]  # Yellow = expected bottom
                
                # Safe zone (100px from bottom)
                safe_zone = HEIGHT - 100
                img[safe_zone:safe_zone+2, :] = [0, 255, 255]  # Cyan
                
                # Video bottom
                if expected_bottom > HEIGHT:
                    img[HEIGHT-3:HEIGHT, :] = [255, 0, 0]  # Red = CLIPPING!
                else:
                    img[HEIGHT-3:HEIGHT, :] = [0, 255, 0]  # Green = safe
                
                bg = ImageClip(img).with_duration(1)
                txt_positioned = txt_clip.with_position(('center', y_pos)).with_duration(1)
                
                final = CompositeVideoClip([bg, txt_positioned], size=(WIDTH, HEIGHT))
                filename = f"test_config_{idx}_text_{text_idx}.png"
                final.save_frame(filename, t=0)
                
                # Status
                status = "✅" if expected_bottom < HEIGHT else "❌ CLIPS!"
                safe_status = "✅" if expected_bottom < safe_zone else "⚠️"
                
                print(f"\n{status} Config {idx}, Text {text_idx}: {config['desc']}")
                print(f"   Text: {test_text[:40]}...")
                print(f"   Size: {txt_clip.w}x{txt_clip.h}")
                print(f"   Y pos: {y_pos}px ({int(config['position']*100)}% from top)")
                print(f"   Bottom: {expected_bottom}px (margin: {bottom_margin}px)")
                print(f"   Safe zone: {safe_status}")
                print(f"   Saved: {filename}")
                
            except Exception as e:
                print(f"\n❌ Config {idx}, Text {text_idx}: Error - {e}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("  Look for config where:")
    print("  - Yellow line is ABOVE cyan (safe zone)")
    print("  - Bottom line is GREEN (not red)")
    print("  - Text doesn't extend beyond yellow line")
    print("  - Position is as low as possible while meeting above criteria")
    print("="*70)

if __name__ == "__main__":
    test_positions_and_margins()
