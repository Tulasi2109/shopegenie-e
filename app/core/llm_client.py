# app/core/llm_client.py

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 1) Load .env for local development
load_dotenv()

# 2) Read key from .env OR Streamlit Cloud Secrets
api_key = (
    st.secrets.get("OPENAI_API_KEY")  # deployed app
)

# 3) If still missing, stop execution with a helpful message
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n\n"
        "➤ Local: Add this line to your .env file:\n"
        "      OPENAI_API_KEY=sk-xxxx\n\n"
        "➤ Deployed: Add the same key to Streamlit → Settings → Secrets."
    )

# 4) Create OpenAI client
client = OpenAI(api_key=api_key)


def chat_llm(*prompt_parts: str) -> str:
    """
    Fast, lightweight chat helper.

    Supports:
        chat_llm("prompt")
        chat_llm(part1, part2, part3...)

    All parts are joined with blank lines.
    """
    if not prompt_parts:
        raise ValueError("chat_llm requires at least one prompt string.")

    combined_prompt = "\n\n".join(str(p) for p in prompt_parts if p)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 🔹 faster + cheaper than gpt-4o-mini
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise, fast assistant. "
                    "Keep answers short, focused, and easy to scan."
                ),
            },
            {"role": "user", "content": combined_prompt},
        ],
        temperature=0.3,
        max_tokens=160,  # 🔹 smaller cap for speed
    )

    return response.choices[0].message.content
