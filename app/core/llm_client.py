# app/core/llm_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file in the current directory
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Make sure your app/.env file contains a line like:\n"
        "OPENAI_API_KEY=sk-...your-key..."
    )

# Create a single OpenAI client used by all agents
client = OpenAI(api_key=api_key)


def chat_llm(*prompt_parts: str) -> str:
    """
    Fast, lightweight chat helper.

    Supports:
        chat_llm("single prompt")
        chat_llm(part1, part2, part3...)

    All parts are joined with blank lines.
    """
    if not prompt_parts:
        raise ValueError("chat_llm requires at least one prompt string.")

    combined_prompt = "\n\n".join(str(p) for p in prompt_parts if p)

    response = client.chat.completions.create(
        # ⚡ use a smaller/faster model
        model="gpt-4o-mini",   # if you want even faster, you can later try "gpt-4o-mini" with shorter prompts
        messages=[
            {
                "role": "system",
                "content": "You are a concise, fast assistant. Keep answers short and focused."
            },
            {"role": "user", "content": combined_prompt},
        ],
        temperature=0.2,      # less creativity, faster convergence
        max_tokens=300,       # cap response length to keep latency low
    )
    return response.choices[0].message.content
