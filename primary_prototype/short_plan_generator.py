import os
import anthropic
from groq import Groq
from time import sleep
from transformers import pipeline
# claude
# def generate_short_plan(summary):
#     """Generates a structured plan for creating a short-form video."""
#     client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
#     plan_prompt = f"""
#     Please generate a structured plan for creating an engaging short-form video based on the following summary:

#     {summary}

#     The plan should include the key elements, transitions, and overall flow of the short video. Please format the plan as a numbered list.
#     """
#     plan_response = client.messages.create(
#         model="claude-3-opus-20240229",
#         max_tokens=500,
#         temperature=0.0,
#         messages=[{"role": "user", "content": plan_prompt}]
#     )
#     plan_text = plan_response.content[0].text.strip()
#     return plan_text

#groq
def chunk_text(text, max_tokens=3000):

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Ensure consistent usage of the tokenizer
    tokenizer = summarizer.tokenizer

    tokens = tokenizer.encode(text, return_tensors='pt')[0]
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=False)
        chunks.append(decoded_chunk)
    
    return chunks
def generate_short_plan(summary, max_retries=5):
    """Generates a structured plan for creating a short-form video."""
    retry_count = 0
    initial_delay = 40
    plan_text=""
    while retry_count < max_retries:
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )

            plan_prompt = f"""
            Please generate a structured plan for creating an engaging short-form video based on the following summary:

            {chunk_text(summary)[0]}

            The plan should include the key elements, transitions, and overall flow of the short video. Please format the plan as a numbered list.
            """
            plan_response = client.chat.completions.create(
                messages=[{"role": "user", "content": plan_prompt}],
                model="mixtral-8x7b-32768",
            )

            plan_text = plan_response.choices[0].message.content
            retry_count=max_retries
        except Exception as e:
            retry_count+=1
            print(f"generation failed...retrying in {initial_delay*retry_count} seconds")
            sleep(initial_delay*retry_count)
    if plan_text=="":
        raise Exception("Generation failed after max retries.")
    else:
        return plan_text