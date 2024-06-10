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
import subprocess

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
        
        return "tts_audio_path: "+temp_audio_path
    except Exception as e:
        print(f"Error generating TTS audio: {str(e)}")
        raise e
    
@tool
def extract_video_clips(video_path: str, timestamps: list) -> list:
    """
    Extracts video clips from the original video based on the provided timestamps. Keep each clip between 0.3 and 5 seconds.
    
    Args:
        video_path (str): Path to the original video file.
        timestamps (list): List of tuples (start, end) in seconds for each clip.
        
    Returns:
        list: Absolute paths to the extracted video clip files.
    """
    try:
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
    except Exception as e:
        print(f"Error extracting video clips: {str(e)}")
        raise e

@tool
def generate_captions(transcription: str) -> str:
    """Generates caption files based on the transcription."""
    caption_file = "captions.txt"
    captions = []
    
    # Split the transcription into individual captions
    # (You may need to adjust this based on the actual format of the transcription)
    sentences = transcription.split('. ')
    
    start_time = 0.0
    duration = 3.0  # Adjust the duration as needed
    
    with open(caption_file, 'w') as f:
        for sentence in sentences:
            if sentence.strip():
                end_time = start_time + duration
                caption = f"{start_time},{end_time},{sentence.strip()}\n"
                f.write(caption)
                captions.append(caption)
                start_time = end_time
    
    return caption_file

@tool
def enhance_video(video_clip_path: str) -> str:
    """Applies video enhancement techniques to improve the visual quality of the video clip."""
    

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
def assemble_audio(audio_paths: list) -> str:
    """Concatenates multiple audio files into a single audio file."""
    os.makedirs("assembled_audio", exist_ok=True)

    output_path = "assembled_audio/final_audio.mp3"
    # Create an empty AudioSegment to store the combined audio
    combined_audio = AudioSegment.empty()
    
    # Iterate over the audio file paths and concatenate them
    for audio_path in audio_paths:
        try:
            audio = AudioSegment.from_mp3(audio_path)
            combined_audio += audio
        except pydub.exceptions.CouldntDecodeError as e:
            print(f"Error decoding audio file: {audio_path}")
            # Handle the error appropriately
    
    # Export the combined audio to the output file
    combined_audio.export(output_path, format="mp3")
    
    return os.path.abspath(output_path)

@tool
def assemble_video(clip_paths: list, audio_file: str) -> str:
    """Stitches the extracted video clips and narration audio together into the final short-form video."""
    # Load the video clips from the provided paths
    clips = [VideoFileClip(path).set_duration(VideoFileClip(path).duration + 0.1) for path in clip_paths]

    # Add a small buffer duration to each clip
    buffer_duration = 0.1
    clips = [clip.set_duration(clip.duration + buffer_duration) for clip in clips]

    # Concatenate the video clips one by one
    final_clip = clips[0]
    for clip in clips[1:]:
        final_clip = concatenate_videoclips([final_clip, clip])

    # Load the audio file
    audio_clip = AudioFileClip(audio_file)

    # Truncate the final video clip to match the audio duration
    final_clip = final_clip.subclip(0, audio_clip.duration)

    # Set the audio of the final video clip
    final_clip = final_clip.set_audio(audio_clip)

    # Crop the final video to 9:16 aspect ratio
    width, height = final_clip.size
    if width / height > 9 / 16:
        # Crop horizontally
        target_width = int(height * 9 / 16)
        x1 = (width - target_width) // 2
        x2 = x1 + target_width
        final_clip = final_clip.crop(x1=x1, x2=x2)
    else:
        # Crop vertically
        target_height = int(width * 16 / 9)
        y1 = (height - target_height) // 2
        y2 = y1 + target_height
        final_clip = final_clip.crop(y1=y1, y2=y2)

    # Write the final video to disk as MP4 with audio
    output_path = "final_video_with_audio.mp4"
    try:
        final_clip.write_videofile(output_path)
    except IndexError as e:
        print(f"Index error while assembling video, ignoring: "+str(e))
        #raise e


    # Close the audio and video clips
    audio_clip.close()
    final_clip.close()
    for clip in clips:
        clip.close()

    return output_path