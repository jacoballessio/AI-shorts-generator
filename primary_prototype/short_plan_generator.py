import os
import anthropic
from groq import Groq

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
def generate_short_plan(summary):
    """Generates a structured plan for creating a short-form video."""

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    plan_prompt = f"""
    Please generate a structured plan for creating an engaging short-form video based on the following summary:

    {summary}

    The plan should include the key elements, transitions, and overall flow of the short video. Please format the plan as a numbered list.
    """
    plan_response = client.chat.completions.create(
        messages=[{"role": "user", "content": plan_prompt}],
        model="mixtral-8x7b-32768",
    )

    plan_text = plan_response.choices[0].message.content
    return plan_text
    
