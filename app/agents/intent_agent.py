from core.llm_client import chat_llm
import json

INTENT_SYSTEM_PROMPT = """
You are the Intent Agent for ShopGenie-E.

Your job is to extract a strict JSON object from the user's request
for buying an electronic product.

Return JSON with the following fields:

- category: one of ["laptop", "phone", "tablet"] (guess if not explicit)
- budget_usd: integer value if mentioned, else null
- primary_goals: list of strings like ["battery_life", "performance", "portability"]
- hard_constraints: object with things like {"min_ram_gb": 16} if mentioned, else {}
- notes: short free-text note summarizing the intent

IMPORTANT:
- Respond with ONLY valid JSON.
- Do NOT include any extra text.
"""

def extract_intent(user_query: str) -> dict:
    raw = chat_llm(INTENT_SYSTEM_PROMPT, user_query)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a minimal intent if parsing fails
        return {
            "category": "laptop",
            "budget_usd": None,
            "primary_goals": [],
            "hard_constraints": {},
            "notes": user_query
        }
