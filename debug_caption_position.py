"""
Debug tool to test caption positioning and avoid clipping.
Creates test frames with captions at different positions.
"""

from moviepy import TextClip, ImageClip, CompositeVideoClip
import numpy as np
from PIL import Image

# Video dimensions (9:16 aspect ratio)
WIDTH = 1080
HEIGHT = 1920

def create_test_frame(caption_text: str, y_position: int, output_path: str, test_params: dict = None):
    """Create a test frame with caption at specified position."""
    
    # Default params
    params = {
        'font_size': 80,
        'stroke_width': 3,
        'method': 'caption',
        'size': (WIDTH - 100, None)
    }
    if test_params:
        params.update(test_params)
    
    # Create a gradient background for visibility
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        img[y, :] = [30 + (y * 50 // HEIGHT), 50, 80]
    
    # Create background clip
    background = ImageClip(img).with_duration(1)
    
    # Add position markers
    marker_height = 50
    markers = []
    
    # Top marker (red)
    top_marker = np.zeros((marker_height, WIDTH, 3), dtype=np.uint8)
    top_marker[:, :] = [255, 0, 0]
    top_clip = ImageClip(top_marker).with_position((0, 0)).with_duration(1)
    markers.append(top_clip)
    
    # Bottom marker (red)
    bottom_marker = np.zeros((marker_height, WIDTH, 3), dtype=np.uint8)
    bottom_marker[:, :] = [255, 0, 0]
    bottom_clip = ImageClip(bottom_marker).with_position((0, HEIGHT - marker_height)).with_duration(1)
    markers.append(bottom_clip)
    
    # Create text clip with test parameters
    try:
        txt_clip = TextClip(
            text=caption_text,
            font_size=params['font_size'],
            color='white',
            stroke_color='black',
            stroke_width=params['stroke_width'],
            method=params['method'],
            size=params['size']
        )
        
        # Get text clip dimensions
        txt_width = txt_clip.w if hasattr(txt_clip, 'w') else 0
        txt_height = txt_clip.h if hasattr(txt_clip, 'h') else 0
        
        # Calculate actual position (MoviePy centers horizontally)
        x_pos = (WIDTH - txt_width) // 2
        
        # Add boundary markers
        if txt_width > 0 and txt_height > 0:
            # Draw box around caption bounds
            box_color = [0, 255, 0]  # Green
            box_thickness = 3
            
            # Left edge
            left_box = np.zeros((txt_height, box_thickness, 3), dtype=np.uint8)
            left_box[:, :] = box_color
            left_clip = ImageClip(left_box).with_position((x_pos, y_position)).with_duration(1)
            markers.append(left_clip)
            
            # Right edge
            right_box = np.zeros((txt_height, box_thickness, 3), dtype=np.uint8)
            right_box[:, :] = box_color
            right_clip = ImageClip(right_box).with_position((x_pos + txt_width - box_thickness, y_position)).with_duration(1)
            markers.append(right_clip)
            
            # Top edge
            top_box = np.zeros((box_thickness, txt_width, 3), dtype=np.uint8)
            top_box[:, :] = box_color
            top_edge_clip = ImageClip(top_box).with_position((x_pos, y_position)).with_duration(1)
            markers.append(top_edge_clip)
            
            # Bottom edge
            bottom_box = np.zeros((box_thickness, txt_width, 3), dtype=np.uint8)
            bottom_box[:, :] = box_color
            bottom_edge_clip = ImageClip(bottom_box).with_position((x_pos, y_position + txt_height - box_thickness)).with_duration(1)
            markers.append(bottom_edge_clip)
        
        # Position caption
        txt_clip = txt_clip.with_position(('center', y_position)).with_duration(1)
        
        # Composite all elements
        final = CompositeVideoClip([background] + markers + [txt_clip], size=(WIDTH, HEIGHT))
        
        # Save frame
        final.save_frame(output_path, t=0)
        
        bottom_position = y_position + txt_height
        margin_from_bottom = HEIGHT - bottom_position
        right_position = x_pos + txt_width
        margin_from_right = WIDTH - right_position
        
        status = "✅ SAFE" if (margin_from_bottom > 50 and margin_from_right > 0) else "❌ CLIPPING!"
        
        print(f"{status} {output_path}")
        print(f"   Params: font={params['font_size']}, stroke={params['stroke_width']}, method={params['method']}")
        print(f"   Clip size: {txt_width}x{txt_height}px")
        print(f"   Position: ({x_pos}, {y_position})")
        print(f"   Margins: bottom={margin_from_bottom}px, right={margin_from_right}px")
        print()
        
    except Exception as e:
        print(f"❌ Error creating caption: {e}")
        import traceback
        traceback.print_exc()

def test_caption_positions():
    """Test various caption positions and parameters."""
    
    real_example = "Ever witnessed a moment so beautiful it felt surreal?"
    
    y_pos = int(HEIGHT * 0.50)  # Test at middle position
    
    print("🎨 Testing different caption parameters...")
    print(f"   Video dimensions: {WIDTH}x{HEIGHT}")
    print(f"   Testing at Y={y_pos} (50% from top)")
    print(f"   Red bars = video edges")
    print(f"   Green box = caption bounding box\n")
    
    # Test different parameter combinations
    tests = [
        ("current", {'font_size': 80, 'stroke_width': 3, 'method': 'caption', 'size': (WIDTH - 100, None)}),
        ("no_stroke", {'font_size': 80, 'stroke_width': 0, 'method': 'caption', 'size': (WIDTH - 100, None)}),
        ("thin_stroke", {'font_size': 80, 'stroke_width': 1, 'method': 'caption', 'size': (WIDTH - 100, None)}),
        ("label_method", {'font_size': 80, 'stroke_width': 3, 'method': 'label', 'size': (WIDTH - 100, None)}),
        ("wider_size", {'font_size': 80, 'stroke_width': 3, 'method': 'caption', 'size': (WIDTH - 50, None)}),
        ("smaller_font", {'font_size': 60, 'stroke_width': 3, 'method': 'caption', 'size': (WIDTH - 100, None)}),
        ("no_size_limit", {'font_size': 80, 'stroke_width': 3, 'method': 'caption', 'size': (None, None)}),
    ]
    
    for name, params in tests:
        print(f"Testing: {name}")
        create_test_frame(real_example, y_pos, f"test_params_{name}.png", params)
    
    print("\n✅ All test frames created!")
    print("   Open PNG files to see which parameters avoid clipping")
    print("   Compare with actual video to identify the issue")

if __name__ == "__main__":
    test_caption_positions()
