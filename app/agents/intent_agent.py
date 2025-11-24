from core.llm_client import chat_llm
import json

INTENT_PROMPT = """
Extract the user's shopping intent as STRICT JSON.

TASK:
Return ONLY a JSON object with these fields:

{
  "category": "laptop" | "phone" | "tablet",
  "budget_usd": number | null,
  "primary_goals": [string, ...],
  "hard_constraints": { string: number | string, ... },
  "notes": string
}

RULES:
- Infer category if not explicit.
- Detect any numeric constraints (RAM, storage, battery, screen, budget).
- Keep goals short (e.g., "battery", "performance", "portability").
- NO explanations or text outside JSON.
"""

def extract_intent(user_query: str) -> dict:
    raw = chat_llm(INTENT_PROMPT, user_query)

    # Remove accidental Markdown formatting if model wraps JSON
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # Remove optional "json"
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback if JSON fails
        return {
            "category": "laptop",
            "budget_usd": None,
            "primary_goals": [],
            "hard_constraints": {},
            "notes": user_query,
        }
