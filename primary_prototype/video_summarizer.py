import whisper_timestamped as whisper
import anthropic
import cv2
import os
import time
import random
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from groq import Groq
import json
from openai import OpenAI

class VideoSummarizer:
    def __init__(self, video_path, shorts_length, cache_dir):
        self.video_path = video_path
        self.shorts_length = shorts_length
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.whisper_model = whisper.load_model("base", download_root=cache_dir)
        self.vlm_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.claude_api_key = os.environ.get("CLAUDE_API_KEY")
    
    def load_data_from_cache(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data['summary'], data['frame_descriptions'], data['transcript']
        return None

    def save_data_to_cache(self, summary, frame_descriptions, transcript, cache_file):
        data = {
            'summary': summary,
            'frame_descriptions': frame_descriptions,
            'transcript': transcript
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)
                
    def extract_audio(self):
        audio_file = 'temp/temp_audio.wav'
        command = [
            'ffmpeg',
            '-y',
            '-i', self.video_path,
            '-ab', '160k',
            '-ac', '1',
            '-ar', '16000',
            '-vn', audio_file
        ]
        os.system(' '.join(command))
        return audio_file
    
                
    # def transcribe_audio(self, audio_file):
    #     result = self.whisper_model.transcribe(audio_file)
    #     transcript_with_timestamps = ""
    #     for segment in result["segments"]:
    #         for word in segment:
    #             start = word["start"]
    #             end = word["end"]
    #             text = word["text"]
    #             transcript_with_timestamps += f"[{start:.2f} - {end:.2f}] {text}\n"
            
    #     return transcript_with_timestamps
    def transcribe_audio(self, audio_file):
        result = whisper.transcribe(self.whisper_model, audio_file)
        transcript_with_timestamps = ""
        
        sentence_start = None
        sentence_end = None
        sentence_text = ""

        for segment in result["segments"]:
            print(segment)
            for word in segment["words"]:
                start = word["start"]
                end = word["end"]
                text = word["text"]
                
                if sentence_start is None:
                    sentence_start = start
                
                sentence_text += text + " "
                
                if "." in word["text"] or "!" in word["text"] or "?" in word["text"]:
                    sentence_end = end
                    transcript_with_timestamps += f"[{sentence_start:.2f} - {sentence_end:.2f}] {sentence_text.strip()}\n"
                    sentence_start = None
                    sentence_end = None
                    sentence_text = ""
                    
        # Handle any remaining text if the last sentence does not end with punctuation
        if sentence_text:
            transcript_with_timestamps += f"[{sentence_start:.2f} - {end:.2f}] {sentence_text.strip()}\n"
        
        return transcript_with_timestamps

    def generate_frame_descriptions(self, frames, timestamps):
        captions = []
        for frame, timestamp in zip(frames, timestamps):
            pixel_values = self.image_processor(images=frame, return_tensors="pt").pixel_values
            generated_ids = self.vlm_model.generate(pixel_values, max_length=50)
            caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            caption_with_timestamp = f"Timestamp: {timestamp:.2f}s - {caption}"
            captions.append(caption_with_timestamp)
        return captions

    def extract_key_frames(self, num_frames=25):
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
    
    def chunk_text(self, text, max_tokens=None):
        if max_tokens is None:
            max_tokens = 1024
            
        # Ensure consistent usage of the tokenizer
        tokenizer = self.summarizer.tokenizer

        tokens = tokenizer.encode(text, return_tensors='pt')[0]
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i + max_tokens]
            decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=False)
            chunks.append(decoded_chunk)
        
        return chunks

    def generate_summary(self, transcript, captions, max_retries=5, initial_delay=1, backoff_factor=2):
        retry_count = 0
        # Combine transcript and captions
        combined_text = f"Transcript:\n{transcript}\n\nVisual Information:\n{chr(10).join(captions)}"
        
        # Chunk the text
        chunks = self.chunk_text(combined_text)

        # Summarize each chunk and combine summaries
        summary_chunks = []
        for i, chunk in enumerate(chunks):
            retry_count = 0
            print(f"chunk #{i} of {len(chunks)}")
            while retry_count < max_retries:
                print(f"failed. Retrying. {retry_count}/{max_retries}")
                try:
                    client = Groq(
                        api_key=os.environ.get("GROQ_API_KEY"),
                    )
                    summary_prompt = f"Identify 1-3 important segments from the transcript. Return the selected timestamps and summarize what was said and what appeared on screen.:\n\nTranscript:\n{chunk}\n\nVisual Information:\n{chr(10).join(captions)}\n\nSummary:"
                    print(f"_____________\n{summary_prompt}\n___________________________")
                    summary_response = client.chat.completions.create(
                        messages=[{"role": "user", "content": summary_prompt}],
                        model="mixtral-8x7b-32768",
                    )

                    summary_text = summary_response.choices[0].message.content
                    summary_chunks.append(summary_text)
                    print(summary_text)
                    retry_count=max_retries
                except Exception as e:
                    retry_count+=1
                    print("error generating summary", e)
        # Combine all chunk summaries into a final summary
        final_summary = " ".join(summary_chunks)
        return final_summary
    
    # def generate_summary(self, transcript, captions, max_retries=5, initial_delay=1, backoff_factor=2):
    #     retry_count = 0
    #     while retry_count < max_retries:
    #         try:
    #             client = Groq(
    #                 api_key=os.environ.get("GROQ_API_KEY"),
    #             )
    #             summary_prompt = f"Please generate a comprehensive summary of the video based on the following transcript and visual information:\n\nTranscript:\n{transcript}\n\nVisual Information:\n{chr(10).join(captions)}\n\nSummary:"
    #             summary_response = client.chat.completions.create(
    #                 messages=[{"role": "user", "content": summary_prompt}],
    #                 model="mixtral-8x7b-32768",
    #             )

    #             summary_text = summary_response.choices[0].message.content
    #             return summary_text
    #         except anthropic.InternalServerError as e:
    #             retry_count += 1
    #             if retry_count < max_retries:
    #                 delay = initial_delay * (backoff_factor ** (retry_count - 1))
    #                 delay = min(delay, 60)  # Limit the maximum delay to 60 seconds
    #                 print(f"Anthropic API server is overloaded. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
    #                 time.sleep(delay + random.uniform(0, 1))  # Add a small random jitter to the delay
    #             else:
    #                 raise e