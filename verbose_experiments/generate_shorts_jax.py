import whisper_jax as whisper
import anthropic
import cv2
import argparse
import numpy as np
import os
from dotenv import load_dotenv
import tempfile
from scipy.io import wavfile
import jax.numpy as jnp
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

class VideoSummarizer:
    def __init__(self, video_path, summary_length, cache_dir):
        self.video_path = video_path
        self.summary_length = summary_length
        self.whisper_model, self.whisper_params = whisper.FlaxWhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2", _do_init=False, dtype=jnp.bfloat16, cache_dir=cache_dir, timeout=100000000
        )
        self.vlm_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir=cache_dir)
        self.claude_api_key = os.environ.get("CLAUDE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

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
            mel = whisper.log_mel_spectrogram(audio_segment)
            input_features = whisper.pad_or_trim(mel, self.whisper_model.config.n_frames)
            input_features = np.expand_dims(input_features, axis=0)
            pred_ids = self.whisper_model.generate(input_features, params=self.whisper_params).sequences
            transcript += self.whisper_model.tokenizer.decode(pred_ids[0])

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

    def generate_summary(self, transcript, captions):
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

    llm = OpenAI(openai_api_key=summarizer.openai_api_key)
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