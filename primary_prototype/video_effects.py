# video_effects.py

from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, vfx
from PIL import Image, ImageEnhance
from langchain.tools import tool
import os

def save_clip(clip, original_path):
    """
    Saves the modified clip to a new file, returns the new file path.
    
    Parameters:
    clip (VideoFileClip): The video clip to save.
    original_path (str): The original file path.
    
    Returns:
    str: The file path of the saved clip.
    """
    new_path = original_path.replace(".mp4", "_modified.mp4")
    clip.write_videofile(new_path, fps=24, codec='libx264')
    return new_path

def delete_original_files(file_paths):
    """
    Deletes the original video files.
    
    Parameters:
    file_paths (list): List of file paths to delete.
    """
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

# Transition: Fade In/Out
@tool
def fade_in_out(file_path, duration=1):
    """
    Applies a fade-in and fade-out transition to a video clip.
    
    Parameters:
    file_path (str): The path to the video clip.
    duration (float): Duration of the fade-in and fade-out in seconds.
    
    Returns:
    str: The path to the modified video clip.
    """
    clip = VideoFileClip(file_path)
    modified_clip = clip.fadein(duration).fadeout(duration)
    return save_clip(modified_clip, file_path)

# Transition: Crossfade
@tool
def crossfade(file_paths, duration=1):
    """
    Crossfades between multiple video clips.
    
    Parameters:
    file_paths (list of str): List of paths to video clips.
    duration (float): Duration of the crossfade transition in seconds.
    
    Returns:
    str: The path to the concatenated video clip with crossfade transitions.
    """
    clips = [VideoFileClip(path) for path in file_paths]
    modified_clip = concatenate_videoclips(clips, method="compose", transition=vfx.crossfadein(duration))
    result_path = save_clip(modified_clip, file_paths[0])
    delete_original_files(file_paths)
    return result_path

# Filter: Apply Brightness
@tool
def adjust_brightness(file_path, factor=1.5):
    """
    Adjusts the brightness of a video clip.
    
    Parameters:
    file_path (str): The path to the video clip.
    factor (float): The brightness factor (1.0 is the original brightness).
    
    Returns:
    str: The path to the modified video clip.
    """
    clip = VideoFileClip(file_path)
    modified_clip = clip.fl_image(lambda frame: ImageEnhance.Brightness(Image.fromarray(frame)).enhance(factor))
    return save_clip(modified_clip, file_path)

# Filter: Apply Contrast
@tool
def adjust_contrast(file_path, factor=1.5):
    """
    Adjusts the contrast of a video clip.
    
    Parameters:
    file_path (str): The path to the video clip.
    factor (float): The contrast factor (1.0 is the original contrast).
    
    Returns:
    str: The path to the modified video clip.
    """
    clip = VideoFileClip(file_path)
    modified_clip = clip.fl_image(lambda frame: ImageEnhance.Contrast(Image.fromarray(frame)).enhance(factor))
    return save_clip(modified_clip, file_path)

# Filter: Grayscale
@tool
def grayscale(file_path):
    """
    Converts a video clip to grayscale.
    
    Parameters:
    file_path (str): The path to the video clip.
    
    Returns:
    str: The path to the grayscale video clip.
    """
    clip = VideoFileClip(file_path)
    modified_clip = clip.fx(vfx.blackwhite)
    return save_clip(modified_clip, file_path)

# Text Overlay: Simple Text
@tool
def add_text_overlay(file_path, text, fontsize=70, color='white', position='center'):
    """
    Adds a simple text overlay to a video clip.
    
    Parameters:
    file_path (str): The path to the video clip.
    text (str): The text to overlay on the video.
    fontsize (int): The font size of the text.
    color (str): The color of the text.
    position (str or tuple): The position of the text ('center', 'top', 'bottom', or a tuple (x, y)).
    
    Returns:
    str: The path to the modified video clip.
    """
    clip = VideoFileClip(file_path)
    txt_clip = TextClip(text, fontsize=fontsize, color=color)
    txt_clip = txt_clip.set_pos(position).set_duration(clip.duration)
    modified_clip = CompositeVideoClip([clip, txt_clip])
    return save_clip(modified_clip, file_path)

# Animation: Zoom In/Out
@tool
def zoom_in_out(file_path, zoom_factor=1.2):
    """
    Applies a zoom in and zoom out effect to a video clip.
    
    Parameters:
    file_path (str): The path to the video clip.
    zoom_factor (float): The zoom factor (greater than 1 for zoom in, less than 1 for zoom out).
    
    Returns:
    str: The path to the modified video clip.
    """
    clip = VideoFileClip(file_path)
    modified_clip = clip.fx(vfx.resize, zoom_factor).fx(vfx.resize, 1/zoom_factor)
    return save_clip(modified_clip, file_path)

# Animation: Rotate
@tool
def rotate_clip(file_path, angle=90):
    """
    Rotates a video clip by a specified angle.
    
    Parameters:
    file_path (str): The path to the video clip.
    angle (float): The angle in degrees to rotate the clip.
    
    Returns:
    str: The path to the rotated video clip.
    """
    clip = VideoFileClip(file_path)
    modified_clip = clip.rotate(angle)
    return save_clip(modified_clip, file_path)

# Transition: Slide In/Out
@tool
def slide_in(file_path, direction='left', duration=1):
    """
    Slides a video clip in from a specified direction.
    
    Parameters:
    file_path (str): The path to the video clip.
    direction (str): The direction of the slide ('left', 'right').
    duration (float): Duration of the slide effect in seconds.
    
    Returns:
    str: The path to the modified video clip.
    """
    clip = VideoFileClip(file_path)
    w, h = clip.size
    if direction == 'left':
        modified_clip = clip.set_position(lambda t: ('%d%%' % (100 - (100 * t/duration)), 0), duration=duration)
    elif direction == 'right':
        modified_clip = clip.set_position(lambda t: ('%d%%' % (-100 + (100 * t/duration)), 0), duration=duration)
    return save_clip(modified_clip, file_path)

# Transition: Slide In/Out for Composite Clips
@tool
def slide_in_composite(file_paths, direction='left', duration=1):
    """
    Applies a slide-in effect to multiple video clips in a composite.
    
    Parameters:
    file_paths (list of str): List of paths to video clips.
    direction (str): The direction of the slide ('left', 'right').
    duration (float): Duration of the slide effect in seconds.
    
    Returns:
    str: The path to the composite video clip with the slide-in effect applied.
    """
    clips = [VideoFileClip(path) for path in file_paths]
    modified_clips = [slide_in(clip, direction, duration) for clip in clips]
    composite_clip = CompositeVideoClip(modified_clips)
    result_path = save_clip(composite_clip, file_paths[0])
    delete_original_files(file_paths)
    return result_path

# List of all tools in the module
VIDEO_EFFECTS = [
    fade_in_out,
    crossfade,
    adjust_brightness,
    adjust_contrast,
    grayscale,
    add_text_overlay,
    zoom_in_out,
    rotate_clip,
    slide_in,
    slide_in_composite,
]