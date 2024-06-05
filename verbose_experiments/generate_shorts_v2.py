import whisper
import anthropic
import cv2
import argparse
import numpy as np
import os
from dotenv import load_dotenv
import tempfile
from scipy.io import wavfile

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

# Get the Claude API key from the environment variable
claude_api_key = os.environ.get("CLAUDE_API_KEY")

# Summarize the transcript using Claude
client = anthropic.Anthropic(api_key=claude_api_key)
summary_prompt = f"Please summarize the following text for a short form video (between 5 and 30 seconds) highlighting the key points in a punchy manner:\n\n{transcript}"
summary_response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    messages=[{"role": "user", "content": summary_prompt}]
)
summary_text = summary_response.content[0].text.strip()
print(f"Summary: {summary_text}")

# Split the summary text into smaller chunks
max_chars_per_line = 50
summary_lines = [summary_text[i:i+max_chars_per_line] for i in range(0, len(summary_text), max_chars_per_line)]

# Calculate the duration of each chunk
chunk_duration = args.summary_length / len(summary_lines)

# Generate the output video with text and relevant frames/clips
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp_output_file = 'temp_output.mp4'
out = cv2.VideoWriter(temp_output_file, fourcc, fps, (width, height))

audio_clips = []
for i, line in enumerate(summary_lines):
    start_time = i * chunk_duration
    end_time = (i + 1) * chunk_duration
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Extract relevant frames/clips from the original video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for j in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Add text to the frame
        cv2.putText(frame, line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    # Extract audio from the relevant clip
    temp_audio_file = f'temp_audio_{i}.wav'
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-t', str(chunk_duration),
        '-i', args.input_video,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        temp_audio_file
    ]
    os.system(' '.join(command))
    audio_clips.append(temp_audio_file)

out.release()

# Concatenate the audio clips
concatenated_audio_file = 'concatenated_audio.wav'
with open(concatenated_audio_file, 'wb') as outfile:
    for audio_clip in audio_clips:
        with open(audio_clip, 'rb') as infile:
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
for audio_clip in audio_clips:
    os.remove(audio_clip)

# Display the output video
cv2.namedWindow('Summary', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(output_file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Summary', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()