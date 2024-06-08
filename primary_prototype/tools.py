import os
import tempfile
from langchain.tools import tool
import pyttsx3
import cv2
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, TextClip
import wave
from pydub import AudioSegment
from gtts import gTTS
from moviepy.audio.AudioClip import CompositeAudioClip

@tool
def generate_narration(summary: str = "", plan: str = "") -> str:
    """Generates the narration text based on the video summary and plan."""
    # Implement narration generation logic here
    narration_text = f"Based on the summary: {summary}\nAnd the plan: {plan}\nThe narration text is generated."
    return narration_text


@tool
def generate_tts(narration_text: str) -> str:
    """Converts the narration text into speech audio files using gTTS."""
    try:
        # Create the "tts" subdirectory if it doesn't exist
        os.makedirs("tts", exist_ok=True)
        
        # Generate a unique filename for the audio file
        temp_audio_path = tempfile.mkstemp(suffix='_narration.mp3', dir='tts')[1]
        
        # Convert the narration text to speech using gTTS
        tts = gTTS(text=narration_text, lang='en')
        tts.save(temp_audio_path)
        
        return temp_audio_path
    except Exception as e:
        print(f"Error generating TTS audio: {str(e)}")
        raise e
    
@tool
def extract_video_clips(video_path: str, timestamps: list) -> list:
    """
    Extracts video clips from the original video based on the provided timestamps. Keep each clip less than 5 seconds.
    
    Args:
        video_path (str): Path to the original video file.
        timestamps (list): List of tuples (start, end) in seconds for each clip.
        
    Returns:
        list: Absolute paths to the extracted video clip files.
    """

    # Create a directory to store the extracted clips
    output_dir = "extracted_clips"
    os.makedirs(output_dir, exist_ok=True)

    clip_paths = []
    for i, (start, end) in enumerate(timestamps):
        # Extract the video clip
        clip = VideoFileClip(video_path).subclip(start, end)

        # Generate the output file path
        output_path = os.path.join(output_dir, f"clip_{i}.mp4")

        # Write the clip to disk
        clip.write_videofile(output_path, codec="libx264")

        # Append the clip path to the list
        clip_paths.append(os.path.abspath(output_path))

    return clip_paths

@tool
def generate_captions(transcription: str) -> str:
    """Generates caption files (e.g., SRT, WebVTT) based on the new generated speech and clipped speech."""
    # Implement caption generation logic here
    # You can use libraries like webvtt or srt to generate caption files
    # For simplicity, let's assume the captions are directly derived from the transcription
    caption_file = "captions.txt"
    with open(caption_file, 'w') as f:
        f.write(transcription)
    return caption_file

@tool
def enhance_video(video_clip_path: str) -> str:
    """Applies video enhancement techniques to improve the visual quality of the video clip."""
    import cv2
    import numpy as np

    # Read the video clip
    video = cv2.VideoCapture(video_clip_path)

    # Get the video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the enhanced video
    enhanced_video_path = "enhanced_" + video_clip_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(enhanced_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Apply video enhancement (e.g., increase brightness by 30%)
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=0)

        # Write the enhanced frame to the output video
        out.write(enhanced_frame)

    # Release the video objects
    video.release()
    out.release()

    return enhanced_video_path

@tool
def assemble_audio(audio_paths: list, output_path: str) -> str:
    """Concatenates multiple audio files into a single audio file."""
    from pydub import AudioSegment
    
    # Create an empty AudioSegment to store the combined audio
    combined_audio = AudioSegment.empty()
    
    # Iterate over the audio file paths and concatenate them
    for audio_path in audio_paths:
        audio = AudioSegment.from_wav(audio_path)
        combined_audio += audio
    
    # Export the combined audio to the output file
    combined_audio.export(output_path, format="wav")
    
    return output_path

@tool
def assemble_video(clip_paths: list, audio_file: str, caption_file: str) -> str:
    """Stitches the extracted video clips, narration audio, and captions together into the final short-form video."""

    clips = [VideoFileClip(path) for path in clip_paths]

    # Concatenate the video clips
    final_clip = concatenate_videoclips(clips)

    # Load the narration audio clip
    audio_clip = AudioFileClip(audio_file)

    # Create a CompositeAudioClip with the narration audio
    composite_audio = CompositeAudioClip([audio_clip])

    # Set the audio of the final clip to the composite audio
    final_clip = final_clip.set_audio(composite_audio)

    # Write the final video to disk
    output_path = "final_video.mp4"
    final_clip.write_videofile(output_path, codec='libx264', preset='ultrafast', threads=4)

    return output_path
