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

def main(video_path, summary_length, cache_dir, cache_summary):
    load_dotenv(override=True)
    anthropic_api_key = os.environ.get("CLAUDE_API_KEY")
    summarizer = VideoSummarizer(video_path, summary_length, cache_dir)

    if cache_summary:
        cache_file = f"{os.path.splitext(video_path)[0]}_summary.json"
        summary, frame_descriptions = summarizer.load_summary_from_cache(cache_file)
        
        if summary is None:
            audio_file = summarizer.extract_audio()
            transcript = summarizer.transcribe_audio(audio_file)
            with open('transcription.txt', 'w') as transcript_file:
                transcript_file.write(transcript)

            key_frames, timestamps = summarizer.extract_key_frames()
            frame_descriptions = summarizer.generate_frame_descriptions(key_frames, timestamps)

            summary = summarizer.generate_summary(transcript, frame_descriptions)
            summarizer.save_summary_to_cache(summary, frame_descriptions, cache_file)
    else:
        audio_file = summarizer.extract_audio()
        transcript = summarizer.transcribe_audio(audio_file)
        with open('transcription.txt', 'w') as transcript_file:
            transcript_file.write(transcript)

        key_frames, timestamps = summarizer.extract_key_frames()
        frame_descriptions = summarizer.generate_frame_descriptions(key_frames, timestamps)

        summary = summarizer.generate_summary(transcript, frame_descriptions)
    print(f"Generated summary: {summary}")

    plan = generate_short_plan(summary)
    print(f"Generated plan: {plan}")

    tools = [
        generate_narration,
        generate_tts,
        extract_video_clips,
        assemble_audio,
        assemble_video,
    ]
    
    #llm = ChatAnthropic(model="claude-3-opus-20240229", api_key=anthropic_api_key)
    #llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    llm = ChatOpenAI(model="gpt-4o")

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
        And additional information: {additional_information}
        Utilize the plan and tools to generate the video highlight.                         
        """,
        input_variables=["plan", "summary", "additional_information"],
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
        #"original_transcript": transcript
    }
    
    result = agent.run(
        {
            "input": f"Summary: {summary}\nPlan: {plan}\nAdditional Information: {additional_info}"
        }
    )
    print(f"Final result: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI-Shorts-Generator')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--summary_length', type=int, default=30, help='Desired length of the summary in seconds')
    parser.add_argument('--cache_dir', default='J:/temp', help='Directory to cache downloaded files')
    parser.add_argument('--cache_summary', action='store_true', help='Cache the video summary')
    args = parser.parse_args()

    main(args.input_video, args.summary_length, args.cache_dir, args.cache_summary)