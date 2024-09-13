import os
import tempfile
from langchain.tools import tool
import pyttsx3
import cv2
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips, concatenate_audioclips, AudioFileClip, CompositeVideoClip, TextClip
import wave
from pydub import AudioSegment
from gtts import gTTS
from moviepy.audio.AudioClip import CompositeAudioClip
import subprocess
from groq import Groq
import numpy as np
from collections import deque
from collections import deque


@tool
def generate_narration(summary: str = "", plan: str = "") -> str:
    """Generates the narration text based on the video summary and plan."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    narration_prompt = f"""
        Generate only the spoken narrative text for a short 5 to 30 second narration based on the following summary and plan. Do not include any production notes or instructions:
        Summary: {summary}
        Plan: {plan}
        Focus on clearly communicating the key points in an engaging narrative style while following the provided plan. Aim for a concise yet impactful narration using only the dialogue/voiceover text without any additional notes or directions.
    """
    narration_response = client.chat.completions.create(
        messages=[{"role": "user", "content": narration_prompt}],
        model="mixtral-8x7b-32768",
    )

    narration_text = narration_response.choices[0].message.content
    return narration_text

@tool
def generate_tts(narration_text: str) -> str:
    """Converts the narration text into speech audio files using gTTS."""
    try:
        # Create the "tts" subdirectory if it doesn't exist
        os.makedirs("temp/tts", exist_ok=True)
        
        # Generate a unique filename for the audio file
        temp_audio_path = tempfile.mkstemp(suffix='_narration.mp3', dir='temp/tts')[1]
        
        # Convert the narration text to speech using gTTS
        tts = gTTS(text=narration_text, lang='en')
        tts.save(temp_audio_path)
        
        return "tts_audio_path: "+temp_audio_path
    except Exception as e:
        print(f"Error generating TTS audio: {str(e)}")
        raise e
    

def crop_to_9_16(frame, face_center):
    """Crops a frame to a 9:16 aspect ratio with the face centered if detected."""
    frame_np = np.array(frame)
    
    # Get frame dimensions
    height, width, _ = frame_np.shape
    
    # Calculate cropping box dimensions to maintain 9:16 aspect ratio
    target_aspect_ratio = 9 / 16
    if width / height > target_aspect_ratio:
        # Image is wider than 9:16
        new_width = int(height * target_aspect_ratio)
        x1 = (width - new_width) // 2
        x2 = x1 + new_width
        y1 = 0
        y2 = height
    else:
        # Image is taller than 9:16
        new_height = int(width / target_aspect_ratio)
        y1 = (height - new_height) // 2
        y2 = y1 + new_height
        x1 = 0
        x2 = width

    # Crop the frame to 9:16 aspect ratio
    cropped_frame = frame_np[y1:y2, x1:x2]

    # If a face center is provided, adjust the cropping box to center the face
    if face_center is not None:
        face_x, face_y = face_center
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Adjust cropping box to center the face
        center_x = min(max(face_x, crop_width // 2), width - crop_width // 2)
        center_y = min(max(face_y, crop_height // 2), height - crop_height // 2)
        
        x1 = int(center_x - crop_width // 2)
        x2 = int(center_x + crop_width // 2)
        y1 = int(center_y - crop_height // 2)
        y2 = int(center_y + crop_height // 2)

        # Ensure cropping box is within bounds
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, width)
        y2 = min(y2, height)

        cropped_frame = frame_np[y1:y2, x1:x2]

    return cropped_frame




def detect_face_center(frame):
    """Detects the center of the first detected face in the frame."""
    gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return (x + w / 2, y + h / 2)
    return None

# Initialize a deque to store face centers over the last 10 frames
# Smoothing parameters
smoothing_window_size = 60
face_center_queue = deque(maxlen=smoothing_window_size)

def process_frame(frame):
    """Process each frame to crop it to 9:16 aspect ratio, centered on the smoothed face position."""
    global face_center_queue
    
    # Detect face center in the current frame
    face_center = detect_face_center(frame)
    
    if face_center is not None:
        # Add the current face center to the queue
        face_center_queue.append(face_center)
        
    # Calculate the average face center from the queue
    if face_center_queue:
        avg_face_center = np.mean(face_center_queue, axis=0)
    else:
        avg_face_center = None
    
    # Crop the frame based on the averaged face center
    return crop_to_9_16(frame, avg_face_center)

@tool
def extract_video_clips(video_path: str, timestamps: list) -> list:
    """
    Extracts video clips from the original video based on the provided timestamps. 
    Each clip is cropped to 9:16 aspect ratio with face-centered cropping.
    Only extract the parts you need.
    Args:
        video_path (str): Path to the original video file.
        timestamps (list): List of tuples (start, end) in seconds for each clip.
        
    Returns:
        list: Absolute paths to the extracted video clip files.
    """
    global face_cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

    try:
        # Create a directory to store the extracted clips
        output_dir = "temp/extracted_clips"
        os.makedirs(output_dir, exist_ok=True)
        buffer = 0.2
        clip_paths = []
        final_timestamp = timestamps[-1][1]
        for i, (start, end) in enumerate(timestamps):
            # Extract the video clip
            start = max(0, start-buffer)
            end = min(end+buffer, final_timestamp)
            clip = VideoFileClip(video_path).subclip(start, end)
            
            # Process the clip with face detection and cropping
            processed_clip = clip.fl_image(process_frame)
            
            # Generate the output file path
            output_path = os.path.join(output_dir, f"clip_{i}.mp4")
            
            # Write the clip to disk
            processed_clip.write_videofile(output_path, codec="libx264")
            
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
    os.makedirs("temp/assembled_audio", exist_ok=True)

    output_path = "temp/assembled_audio/final_audio.mp3"
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
def assemble_video(clip_paths: list, audio_clip_paths: list = None) -> str:
    """Stitches the extracted video clips and narration audio together into the final short-form video."""
    # Load the video clips from the provided paths
    clips = [VideoFileClip(path) for path in clip_paths]

    # Concatenate the video clips one by one
    final_clip = concatenate_videoclips(clips)

    if audio_clip_paths:
        # Concatenate the audio clips
        audio_clips = [AudioFileClip(path) for path in audio_clip_paths]
        audio_clip = concatenate_audioclips(audio_clips)

        # Ensure the audio duration matches the video duration
        audio_clip = audio_clip.set_duration(final_clip.duration)

        # Set the audio of the final video clip
        final_clip = final_clip.set_audio(audio_clip)
    else:
        # Use the original audio from the video clips
        final_clip = final_clip.set_audio(final_clip.audio)

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
    final_clip.write_videofile(output_path)

    # Close the audio and video clips
    final_clip.close()
    for clip in clips:
        clip.close()

    return os.path.abspath(output_path)