import whisper
import anthropic
import cv2
import argparse
import numpy as np
import os
from dotenv import load_dotenv
import tempfile
from scipy.io import wavfile
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from gtts import gTTS

# Load environment variables from .env file
load_dotenv(override=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='AI-Shorts-Generator')
parser.add_argument('input_video', help='Path to the input video file')
parser.add_argument('--summary_length', type=int, default=30, help='Desired length of the summary in seconds')
args = parser.parse_args()

# Load the Whisper model for transcription
whisper_model = whisper.load_model("base")

# Load the video and extract audio
cap = cv2.VideoCapture(args.input_video)
if not cap.isOpened():
    print(f"Error: Unable to open video file '{args.input_video}'")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = num_frames / fps

# Extract audio from the video
fourcc = cv2.VideoWriter_fourcc(*'avc1')
audio_file = 'temp_audio.wav'
command = [
    'ffmpeg',
    '-i', args.input_video,
    '-ab', '160k',
    '-ac', '1',
    '-ar', '16000',
    '-vn', audio_file
]
os.system(' '.join(command))

# Load the audio file
sample_rate, audio_data = wavfile.read(audio_file)
print(f"Sample rate: {sample_rate}, Audio data shape: {audio_data.shape}")

# Split the audio into smaller segments for transcription
segment_duration = 30  # Adjust the segment duration as needed
num_segments = int(duration) // segment_duration

transcript = ""
for i in range(num_segments):
    start_time = i * segment_duration
    end_time = (i + 1) * segment_duration
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    print(f"Segment {i+1}: Start sample: {start_sample}, End sample: {end_sample}")
    segment_audio = audio_data[start_sample:end_sample]

    # Save the audio segment as a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        wavfile.write(temp_audio_path, sample_rate, segment_audio)

    # Transcribe the audio segment using Whisper
    audio_segment = whisper.pad_or_trim(whisper.load_audio(temp_audio_path))
    mel = whisper.log_mel_spectrogram(audio_segment).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    language = max(probs, key=probs.get)
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    transcript += result.text + " "
    print(f"Segment {i+1} transcription: {result.text}")

    # Delete the temporary audio file
    os.unlink(temp_audio_path)

# Delete the temporary audio file
os.unlink(audio_file)

# Save the full transcript to a file
with open('transcription.txt', 'w') as transcript_file:
    transcript_file.write(transcript)

# Load the Vision-Language Model (VLM)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Extract key frames from the video
key_frames = []
frame_interval = int(fps * args.summary_length / 10)  # Extract 10 key frames
for i in range(10):
    frame_number = i * frame_interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        key_frames.append(frame)

# Generate captions for the key frames using the VLM
key_frame_captions = []
for frame in key_frames:
    pixel_values = image_processor(images=frame, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=50)
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    key_frame_captions.append(caption)

# Get the Claude API key from the environment variable
claude_api_key = os.environ.get("CLAUDE_API_KEY")

# Generate an outline for the short-form video using Claude
client = anthropic.Anthropic(api_key=claude_api_key)
outline_prompt = f"Please generate an outline for a short-form video based on the following transcript and visual information:\n\nTranscript:\n{transcript}\n\nVisual Information:\n{chr(10).join(key_frame_captions)}\n\nThe outline should include the following:\n1. When to use text-to-speech (TTS) and what the TTS should say\n2. When to display captions and what the captions should say\n3. When to display specific clips or frames and a description of the clip or frame\n\nPlease format the outline as follows:\n- [TTS]: <TTS content>\n- [Caption]: <Caption content>\n- [Clip/Frame]: <Description of clip or frame>"
outline_response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    messages=[{"role": "user", "content": outline_prompt}]
)
outline_text = outline_response.content[0].text.strip()
print(f"Generated outline: {outline_text}")

# Parse the outline and generate the short-form video
outline_items = outline_text.split('\n')
video_segments = []

for item in outline_items:
    if item.startswith('- [TTS]:'):
        tts_content = item[8:].strip()
        tts = gTTS(text=tts_content, lang='en', slow=False)
        tts_file = 'tts.mp3'
        tts.save(tts_file)
        video_segments.append(('tts', tts_file))
    elif item.startswith('- [Caption]:'):
        caption_content = item[12:].strip()
        video_segments.append(('caption', caption_content))
    elif item.startswith('- [Clip/Frame]:'):
        clip_description = item[15:].strip()
        # Find the relevant clip/frame based on the description
        # You can implement your own logic here to select the appropriate clip/frame
        # For simplicity, let's assume the first frame matches the description
        video_segments.append(('clip', key_frames[0]))

# Generate the output video based on the parsed outline
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp_output_file = 'temp_output.mp4'
out = cv2.VideoWriter(temp_output_file, fourcc, fps, (width, height))

audio_files = []

for segment in video_segments:
    segment_type, segment_data = segment
    
    if segment_type == 'tts':
        audio_files.append(segment_data)
    elif segment_type == 'caption':
        # Create a frame with the caption text
        caption_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(caption_frame, segment_data, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the caption frame for a certain duration (e.g., 3 seconds)
        for _ in range(int(fps * 3)):
            out.write(caption_frame)
    elif segment_type == 'clip':
        # Write the clip frame
        out.write(segment_data)

out.release()

# Concatenate the audio files
concatenated_audio_file = 'concatenated_audio.wav'
with open(concatenated_audio_file, 'wb') as outfile:
    for audio_file in audio_files:
        with open(audio_file, 'rb') as infile:
            outfile.write(infile.read())

# Merge the audio with the output video
output_file = 'output.mp4'
command = [
    'ffmpeg',
    '-i', temp_output_file,
    '-i', concatenated_audio_file,
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-map', '0:v',
    '-map', '1:a',
    '-shortest',
    output_file
]
os.system(' '.join(command))

# Clean up temporary files
os.remove(temp_output_file)
os.remove(concatenated_audio_file)
for audio_file in audio_files:
    os.remove(audio_file)

# Display the output video
cv2.namedWindow('Short-Form Video', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(output_file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Short-Form Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()