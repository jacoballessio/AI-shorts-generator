import argparse
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain_anthropic import ChatAnthropic
import json
from video_summarizer import VideoSummarizer
from short_plan_generator import generate_short_plan
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import Tool
from tools import generate_narration, generate_tts, extract_video_clips, generate_captions, enhance_video, assemble_video, assemble_audio
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from video_effects import VIDEO_EFFECTS
from pytubefix import YouTube
from pytubefix.cli import on_progress
from langchain.callbacks import OpenAICallbackHandler

MODEL_NAME = "gpt-4o-2024-08-06"
MODEL_PRICES = {"gpt-4o": 5, "gpt-4o-2024-08-06": 2.5} # dollars per million tokens

class TokenCountingCallbackHandler(OpenAICallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_new_token(self, token: str, **kwargs):
        self.total_tokens += 1

    def on_llm_end(self, response, **kwargs):
        self.prompt_tokens = response['usage']['prompt_tokens']
        self.completion_tokens = response['usage']['completion_tokens']
        self.total_tokens = response['usage']['total_tokens']


def DownloadYouTubeVideo(link, download_path):
    try:
        yt = YouTube(link, on_progress_callback=on_progress)
        print(yt.title)
        
        ys = yt.streams.get_highest_resolution()
        downloaded_file_path = ys.download(download_path)
        
        # Rename the downloaded file to replace spaces with underscores
        base_name = os.path.basename(downloaded_file_path)
        new_name = base_name.replace(" ", "_")
        new_file_path = os.path.join(download_path, new_name)
        
        os.rename(downloaded_file_path, new_file_path)
        
        return new_file_path
    except FileExistsError:
        print(f"The file already exists: {new_file_path}")
        return new_file_path
        
    except Exception as e:
        print(f"An error occurred while downloading the video: {e}")
        return None
    
def main(shorts_length, visual_information_density, user_prompt, cache_dir, cache_summary, video_path=None, youtube_url=None):
    load_dotenv(override=True)
    if video_path == None and youtube_url != None and youtube_url != "":
        video_path = DownloadYouTubeVideo(youtube_url, cache_dir)
        print(video_path)

    if video_path == None:
        raise Exception("video path not found")
    
    summarizer = VideoSummarizer(video_path, shorts_length, cache_dir)

    if cache_summary:
        cache_file = f"{os.path.splitext(video_path)[0]}_data.json"
        data = summarizer.load_data_from_cache(cache_file)

        if data is None:
            audio_file = summarizer.extract_audio()
            transcript = summarizer.transcribe_audio(audio_file)
            key_frames, timestamps = summarizer.extract_key_frames()
            frame_descriptions = summarizer.generate_frame_descriptions(key_frames, timestamps)
            summary = summarizer.generate_summary(transcript, frame_descriptions)
            summarizer.save_data_to_cache(summary, frame_descriptions, transcript, cache_file)
        else:
            summary, frame_descriptions, transcript = data
    else:
        audio_file = summarizer.extract_audio()
        transcript = summarizer.transcribe_audio(audio_file)
        key_frames, timestamps = summarizer.extract_key_frames(visual_information_density)
        frame_descriptions = summarizer.generate_frame_descriptions(key_frames, timestamps)
        summary = summarizer.generate_summary(transcript, frame_descriptions)
    print(f"Generated summary: {summary}")

    plan_generation_instructions = f"Summary: {summary}\nTranscript: {transcript[:10000]}\nDesired video length: {shorts_length}\nUser Instructions:{user_prompt}"
    plan = generate_short_plan(plan_generation_instructions)
    print(f"Generated plan: {plan}")

    tools = [
        generate_narration,
        generate_tts,
        extract_video_clips,
        assemble_audio,
        assemble_video,
    ]

    # Initialize the token counting callback handler
    callback_handler = OpenAICallbackHandler()

    # Create the LLM and pass the callback
    llm = ChatOpenAI(model=MODEL_NAME, callbacks=[callback_handler])

    agent_prompt = PromptTemplate(
        template="""
        You are an AI agent specializing in generating highly engaging, bite-sized video content. Your task is to analyze the provided information from a longer video, identify the most compelling and essential point(s), and create an ultra-concise 5-30 second video highlight without any additional input from the user.
        When generating the short-form content, consider the following:

        Relevance: Focus on the single most relevant and important point or moment that encapsulates the core message or theme of the original video.
        Impact: Select the most attention-grabbing, surprising, or emotionally resonant moment that will leave a lasting impression on the viewer.
        Clarity: Ensure that the highlighted content is clear and easily understandable, even without the full context of the original video.
        Punchiness: Structure the short video with an impactful opening that immediately captures the viewer's attention and a powerful conclusion that leaves them wanting more.
        Visuals: Prioritize visually striking or memorable moments that will make the short video stand out and be highly shareable.
        Audio: Consider the role of key dialogue, sound effects, or background music in enhancing the impact and memorability of the short video.
        Target Platform: Tailor the short-form content to the specific requirements and best practices of the target platform, such as YouTube Shorts, Instagram Reels, or TikTok.

        Your goal is to create an ultra-concise video highlight that captures the essence of the original content in a highly engaging and shareable format. Utilize your creativity, storytelling skills, and understanding of attention-grabbing techniques to craft a memorable and impactful short-form video that leaves a lasting impression.
        The plan for generating the short video is: {plan}
        Given the summary of the original video: {summary}
        User prompt: {user_prompt}
        And additional information: {additional_information}
        Utilize the plan and tools to generate the video highlight.                         
        """,
        input_variables=["plan", "summary", "user_prompt", "additional_information"],
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_prompt=agent_prompt,
        handle_parsing_errors=True
    )

    additional_info = {
        "original_video_path": os.path.abspath(video_path),
        "frame_descriptions": ', '.join(frame_descriptions),
        "original_transcript": transcript,
        "desired_shorts_length": "The video should be exactly " + str(shorts_length) + " seconds long"
    }

    result = agent.run(
        {
            "input": f"Summary: {summary}\nPlan: {plan}\nUser Prompt: {user_prompt}\nAdditional Information: {additional_info}"
        }
    )
    print(f"Final result: {result}")

    # Calculate the cost based on the total tokens and the model's price per million tokens
    total_tokens = callback_handler.total_tokens
    model_price_per_million = MODEL_PRICES.get(MODEL_NAME, 0)
    cost = (model_price_per_million / 1_000_000) * total_tokens
    print(f"Total tokens used: {total_tokens}")
    print(f"Total cost: ${cost:.6f}")

    return cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI-Shorts-Generator')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--shorts_length', type=int, default=30, help='Desired length of the summary in seconds')
    parser.add_argument('--visual_information_density', type=int, default=30, help='Desired length of the summary in seconds')
    parser.add_argument('--user_prompt', type=str, default=None, help='Instructions from user to guide model')
    parser.add_argument('--cache_dir', default='J:/temp', help='Directory to cache downloaded files')
    parser.add_argument('--cache_summary', action='store_true', help='Cache the video summary')
    args = parser.parse_args()

    main(args.input_video, None, args.shorts_length, args.visual_information_density, args.user_prompt, args.cache_dir, args.cache_summary)