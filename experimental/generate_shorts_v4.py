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

class VideoSummarizer:
    def __init__(self, video_path, summary_length):
        self.video_path = video_path
        self.summary_length = summary_length
        self.whisper_model = whisper.load_model("base")
        self.vlm_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.claude_api_key = os.environ.get("CLAUDE_API_KEY")

    def extract_audio(self):
        audio_file = 'temp_audio.wav'
        command = [
            'ffmpeg',
            '-i', self.video_path,
            '-ab', '160k',
            '-ac', '1',
            '-ar', '16000',
            '-vn', audio_file
        ]
        os.system(' '.join(command))
        return audio_file

    def transcribe_audio(self, audio_file):
        sample_rate, audio_data = wavfile.read(audio_file)
        segment_duration = 30
        num_segments = int(len(audio_data) // (segment_duration * sample_rate))

        transcript = ""
        for i in range(num_segments):
            start_sample = i * segment_duration * sample_rate
            end_sample = (i + 1) * segment_duration * sample_rate
            segment_audio = audio_data[start_sample:end_sample]

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                wavfile.write(temp_audio_path, sample_rate, segment_audio)

            audio_segment = whisper.pad_or_trim(whisper.load_audio(temp_audio_path))
            mel = whisper.log_mel_spectrogram(audio_segment).to(self.whisper_model.device)
            _, probs = self.whisper_model.detect_language(mel)
            language = max(probs, key=probs.get)
            options = whisper.DecodingOptions()
            result = whisper.decode(self.whisper_model, mel, options)
            transcript += result.text + " "

            os.unlink(temp_audio_path)

        os.unlink(audio_file)
        return transcript

    def extract_key_frames(self, num_frames=10):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.summary_length / num_frames)

        key_frames = []
        for i in range(num_frames):
            frame_number = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                key_frames.append(frame)

        cap.release()
        return key_frames

    def generate_captions(self, frames):
        captions = []
        for frame in frames:
            pixel_values = self.image_processor(images=frame, return_tensors="pt").pixel_values
            generated_ids = self.vlm_model.generate(pixel_values, max_length=50)
            caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            captions.append(caption)
        return captions

    def generate_outline(self, transcript, captions):
        client = anthropic.Anthropic(api_key=self.claude_api_key)
        outline_prompt = f"Please generate an outline for a short-form video based on the following transcript and visual information:\n\nTranscript:\n{transcript}\n\nVisual Information:\n{chr(10).join(captions)}\n\nThe outline should include the following:\n1. When to use text-to-speech (TTS) and what the TTS should say\n2. When to display captions and what the captions should say\n3. When to display specific clips or frames and a description of the clip or frame\n\nPlease format the outline as follows:\n- [TTS]: <TTS content>\n- [Caption]: <Caption content>\n- [Clip/Frame]: <Description of clip or frame>"
        outline_response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": outline_prompt}]
        )
        outline_text = outline_response.content[0].text.strip()
        return outline_text

    def parse_outline(self, outline, frames):
        outline_items = outline.split('\n')
        video_segments = []

        for i, item in enumerate(outline_items):
            if item.startswith('- [TTS]:'):
                tts_content = item[8:].strip()
                tts = gTTS(text=tts_content, lang='en', slow=False)
                tts_file = f'tts_{i}.mp3'
                tts.save(tts_file)
                video_segments.append(('tts', tts_file))
            elif item.startswith('- [Caption]:'):
                caption_content = item[12:].strip()
                video_segments.append(('caption', caption_content))
            elif item.startswith('- [Clip/Frame]:'):
                clip_description = item[15:].strip()
                video_segments.append(('clip', frames[0]))

        return video_segments

    def generate_video(self, video_segments):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output_file = 'temp_output.mp4'
        out = cv2.VideoWriter(temp_output_file, fourcc, fps, (width, height))

        audio_files = []

        for segment in video_segments:
            segment_type, segment_data = segment

            if segment_type == 'tts':
                audio_files.append(os.path.abspath(segment_data))
            elif segment_type == 'caption':
                caption_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(caption_frame, segment_data, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                for _ in range(int(fps * 3)):
                    out.write(caption_frame)
            elif segment_type == 'clip':
                out.write(segment_data)

        out.release()
        cap.release()

        if audio_files:
            # Create a temporary file to store the list of audio files
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write('\n'.join([f"file '{audio_file}'" for audio_file in audio_files]))
                temp_file_path = temp_file.name

            concatenated_audio_file = 'concatenated_audio.wav'
            command = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_file_path,
                '-c', 'copy',
                concatenated_audio_file
            ]
            result = os.system(' '.join(command))

            if result == 0:
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
                os.remove(concatenated_audio_file)
            else:
                output_file = temp_output_file

            os.remove(temp_file_path)
            for audio_file in audio_files:
                try:
                    os.remove(audio_file)
                except FileNotFoundError:
                    print(f"Warning: File '{audio_file}' not found. Skipping removal.")
        else:
            output_file = temp_output_file

        return output_file

def main(video_path, summary_length):
    summarizer = VideoSummarizer(video_path, summary_length)

    audio_file = summarizer.extract_audio()
    transcript = summarizer.transcribe_audio(audio_file)
    with open('transcription.txt', 'w') as transcript_file:
        transcript_file.write(transcript)

    key_frames = summarizer.extract_key_frames()
    captions = summarizer.generate_captions(key_frames)

    outline = summarizer.generate_outline(transcript, captions)
    print(f"Generated outline: {outline}")

    video_segments = summarizer.parse_outline(outline, key_frames)
    output_file = summarizer.generate_video(video_segments)

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

if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description='AI-Shorts-Generator')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--summary_length', type=int, default=30, help='Desired length of the summary in seconds')
    args = parser.parse_args()

    main(args.input_video, args.summary_length)