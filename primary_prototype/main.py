import argparse
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic

from video_summarizer import VideoSummarizer
from short_plan_generator import generate_short_plan
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from tools import generate_narration, generate_tts, extract_video_clips, generate_captions, enhance_video, assemble_video, assemble_audio
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def main(video_path, summary_length, cache_dir):
    load_dotenv(override=True)
    anthropic_api_key = os.environ.get("CLAUDE_API_KEY")
    summarizer = VideoSummarizer(video_path, summary_length, cache_dir)

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

    tools = [generate_narration, generate_tts, extract_video_clips, generate_captions, assemble_audio, assemble_video]
    
    #llm = ChatAnthropic(model="claude-3-opus-20240229", api_key=anthropic_api_key)
    #llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    llm = ChatOpenAI(model="gpt-4o")

    agent_prompt = hub.pull("hwchase17/openai-tools-agent")
    agent_prompt.messages[0] = SystemMessage("""
        You are an AI agent specializing in generating engaging short-form video content. Your task is to analyze the provided information from a longer video, identify the most compelling and essential points, and create a concise yet captivating 5-30 second video summary without any additional input from the user.

        When generating the short-form content, consider the following:

        1. Relevance: Focus on the key elements that are most relevant and important to the overall message or theme of the original video.

        2. Engagement: Select the most interesting, surprising, or emotionally resonant moments that will capture and maintain the viewer's attention throughout the short video.

        3. Clarity: Ensure that the summarized content is clear, coherent, and easily understandable, even without the full context of the original video.

        4. Pacing: Structure the short video with a compelling opening, a well-paced narrative flow, and a satisfying conclusion that leaves a lasting impact on the viewer.

        5. Visuals: Prioritize visually striking or memorable moments that will make the short video stand out and leave a strong impression.

        6. Audio: Consider the role of audio, such as key dialogue, sound effects, or background music, in enhancing the emotional impact and engagement of the short video.

        7. Target Platform: Tailor the short-form content to the specific requirements and best practices of the target platform, such as YouTube Shorts, Instagram Reels, or TikTok.

        Your goal is to create a short video that effectively communicates the core message or theme of the original content while maximizing viewer engagement and shareability. Utilize your creativity, storytelling skills, and understanding of attention-grabbing techniques to craft a compelling and impactful short-form video.
        
        Utilize the plan and tools to generate the video.                                 
        """)
    agent_prompt.pretty_print()
    
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    additional_info = {
        "original_video_path": os.path.abspath(video_path),
        "frame_descriptions": ', '.join(frame_descriptions),
        "original_transcript": transcript
    }

    print(str(additional_info))
    
    result = agent_executor.invoke(
    {
        "input": "summary: "+summary+"\nplan: "+plan+"\nadditional information: "+str(additional_info)+"\nGenerate a short video based on this plan. Do not ask"
    },
    {"configurable": {"session_id": "unused"}}
    )
    print(f"Final result: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI-Shorts-Generator')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--summary_length', type=int, default=30, help='Desired length of the summary in seconds')
    parser.add_argument('--cache_dir', default='J:/temp', help='Directory to cache downloaded files')
    args = parser.parse_args()

    main(args.input_video, args.summary_length, args.cache_dir)