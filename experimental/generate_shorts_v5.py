import whisper
import anthropic
import cv2
import argparse
import numpy as np
import os
from dotenv import load_dotenv
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
import time
import random
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models import ChatAnthropic

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

    def generate_summary(self, transcript, captions, max_retries=5, initial_delay=1, backoff_factor=2):
        retry_count = 0
        while retry_count < max_retries:
            try:
                client = anthropic.Anthropic(api_key=self.claude_api_key)
                summary_prompt = f"Please generate a comprehensive summary of the video based on the following transcript and visual information:\n\nTranscript:\n{transcript}\n\nVisual Information:\n{chr(10).join(captions)}\n\nSummary:"
                summary_response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    temperature=0.0,
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                summary_text = summary_response.content[0].text.strip()
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

@tool
def generate_short_plan(summary):
    """Generates a structured plan for creating a short-form video."""
    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    plan_prompt = f"Please generate a structured plan for creating a short-form video based on the following summary:\n\n{summary}\n\nThe plan should include the key elements, transitions, and overall flow of the short. Please format the plan as a numbered list."
    plan_response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=500,
        temperature=0.0,
        messages=[{"role": "user", "content": plan_prompt}]
    )
    plan_text = plan_response.content[0].text.strip()
    return plan_text

def main(video_path, summary_length, cache_dir):
    load_dotenv(override=True)

    summarizer = VideoSummarizer(video_path, summary_length, cache_dir)

    audio_file = summarizer.extract_audio()
    transcript = summarizer.transcribe_audio(audio_file)
    with open('transcription.txt', 'w') as transcript_file:
        transcript_file.write(transcript)

    key_frames = summarizer.extract_key_frames()
    captions = summarizer.generate_captions(key_frames)

    summary = summarizer.generate_summary(transcript, captions)
    print(f"Generated summary: {summary}")

    tools = [generate_short_plan]

    llm = ChatAnthropic(anthropic_api_key=summarizer.claude_api_key)
    agent = create_tool_calling_agent(llm, tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    plan = agent_executor.run(summary)
    print(f"Generated plan: {plan}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI-Shorts-Generator')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--summary_length', type=int, default=30, help='Desired length of the summary in seconds')
    parser.add_argument('--cache_dir', default='J:/temp', help='Directory to cache downloaded files')
    args = parser.parse_args()

    main(args.input_video, args.summary_length, args.cache_dir)