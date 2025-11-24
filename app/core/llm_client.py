# app/core/llm_client.py

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Prefer secrets in deployed app; env var works locally if you want to add it back
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n\n"
        "Add it to Streamlit Secrets for the deployed app."
    )

client = OpenAI(api_key=api_key)


def chat_llm(*prompt_parts: str) -> str:
    """
    LLM helper for JSON/explanations.
    If the API call fails (e.g., insufficient quota), returns an empty string
    so the rest of the pipeline can fall back gracefully.
    """
    if not prompt_parts:
        raise ValueError("chat_llm requires at least one prompt string.")

    combined_prompt = "\n\n".join(str(p) for p in prompt_parts if p)

    try:
        response = client.chat.completions.create(
            # when you have quota again, this is a good balance of quality + speed
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant that returns well-structured, "
                        "helpful answers and valid JSON when requested."
                    ),
                },
                {"role": "user", "content": combined_prompt},
            ],
            temperature=0.4,
            max_tokens=400,  # enough for multiple explanations but still fast
        )
        return response.choices[0].message.content

    except Exception as e:
        # Log the error to the terminal for debugging, but don't crash the app
        print("LLM error in chat_llm:", repr(e))
        # Returning empty string lets the reasoner fall back to default explanation
        return ""
