import whisper
import anthropic
import cv2
import os
import time
import random
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from groq import Groq

class VideoSummarizer:
    def __init__(self, video_path, summary_length, cache_dir):
        self.video_path = video_path
        self.summary_length = summary_length
        self.whisper_model = whisper.load_model("base", download_root=cache_dir)
        self.vlm_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
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
        result = self.whisper_model.transcribe(audio_file)
        return result["text"]

    def generate_frame_descriptions(self, frames, timestamps):
        captions = []
        for frame, timestamp in zip(frames, timestamps):
            pixel_values = self.image_processor(images=frame, return_tensors="pt").pixel_values
            generated_ids = self.vlm_model.generate(pixel_values, max_length=50)
            caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            caption_with_timestamp = f"Timestamp: {timestamp:.2f}s - {caption}"
            captions.append(caption_with_timestamp)
        return captions

    def extract_key_frames(self, num_frames=20):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(total_frames / num_frames)

        key_frames = []
        timestamps = []
        for i in range(num_frames):
            frame_number = i * frame_interval
            timestamp = frame_number / fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                key_frames.append(frame)
                timestamps.append(timestamp)

        cap.release()
        return key_frames, timestamps

    def generate_summary(self, transcript, captions, max_retries=5, initial_delay=1, backoff_factor=2):
        retry_count = 0
        while retry_count < max_retries:
            try:
                client = Groq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                )
                summary_prompt = f"Please generate a comprehensive summary of the video based on the following transcript and visual information:\n\nTranscript:\n{transcript}\n\nVisual Information:\n{chr(10).join(captions)}\n\nSummary:"
                summary_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": summary_prompt}],
                    model="mixtral-8x7b-32768",
                )

                summary_text = summary_response.choices[0].message.content
                return summary_text
            except anthropic.InternalServerError as e:
                retry_count += 1
                if retry_count < max_retries:
                    delay = initial_delay * (backoff_factor ** (retry_count - 1))
                    delay = min(delay, 60)  # Limit the maximum delay to 60 seconds
                    print(f"Anthropic API server is overloaded. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(delay + random.uniform(0, 1))  # Add a small random jitter to the delay
                else:
                    raise e