# app/agents/reasoner_agent.py

from typing import Dict, Any, List
import math

import pandas as pd

from core.llm_client import chat_llm


def _normalize_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Min–max normalize a numeric series to [0, 1].
    If higher_is_better is False, the scale is inverted so that lower values score higher.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        # everything missing → neutral score
        return pd.Series([0.5] * len(s), index=series.index)

    min_val = s.min()
    max_val = s.max()
    if math.isclose(min_val, max_val):
        return pd.Series([0.5] * len(s), index=series.index)

    norm = (s - min_val) / (max_val - min_val)
    if not higher_is_better:
        norm = 1.0 - norm
    return norm.fillna(0.5)


def _compute_weights(intent: Dict[str, Any], category: str) -> Dict[str, float]:
    """
    Compute feature weights based on the user's primary_goals and device category.
    We keep this simple but explainable.
    """
    goals_raw = intent.get("primary_goals") or []
    if isinstance(goals_raw, str):
        goals = [goals_raw.lower()]
    else:
        goals = [str(g).lower() for g in goals_raw]

    # Base weights per category (price, performance, battery, screen)
    if category in ("laptop", "tablet", "phone"):
        weights = {
            "price": 0.30,
            "performance": 0.40,
            "battery": 0.30,
            "screen": 0.00,  # not used for these
        }
    else:  # monitor
        weights = {
            "price": 0.40,
            "performance": 0.10,  # e.g., refresh rate / panel – we don't model it here
            "battery": 0.00,
            "screen": 0.50,
        }

    def bump(key: str, delta: float = 0.15):
        if key in weights:
            weights[key] += delta

    # Simple mapping from textual goals to weight bumps
    for g in goals:
        if "performance" in g or "speed" in g or "gaming" in g:
            bump("performance")
        if "battery" in g or "long life" in g or "all day" in g:
            bump("battery")
        if "budget" in g or "cheap" in g or "affordable" in g or "price" in g:
            bump("price")
        if "screen" in g or "display" in g or "bigger" in g:
            bump("screen")

    # Normalize so weights sum to 1
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}

    return {k: v / total for k, v in weights.items()}


def _build_explanation(
    row: pd.Series,
    score: float,
    category: str,
    goals: List[str],
) -> str:
    """
    Ask the LLM to write a short, human explanation for this product.
    """

    goals_text = ", ".join(goals) if goals else "balanced everyday use"

    prompt = f"""
You are an expert electronics advisor.

The user cares about: {goals_text}.
Device category: {category}

Here is one candidate product (JSON-like):
{row.to_dict()}

This product has an overall score of {round(score * 100)} out of 100
based on price, performance, battery, and screen (where applicable).

Write a concise explanation (2–3 sentences) focusing on:
- Why this product is a good fit for the user, given their goals
- One clear trade-off or limitation

Avoid repeating raw numbers; instead, speak qualitatively
(e.g., "strong battery life", "lightweight", "a bit more expensive", etc.).
"""
    return chat_llm(prompt)


def rank_products(intent: Dict[str, Any], products: pd.DataFrame) -> Dict[str, Any]:
    """
    Rank products using a multi-criteria score and generate explanations.

    Returns a dict:
    {
        "results": [
            {"id": ..., "title": ..., "score": 94, "explanation": "..."},
            ...
        ],
        "category": "laptop/phone/tablet/monitor",
        "weights": {"price": ..., "performance": ..., "battery": ..., "screen": ...}
    }
    """
    if products is None or products.empty:
        return {"results": [], "category": None, "weights": {}}

    # Assume all rows are from the same category (enforced by retrieval_agent)
    category = (
        products["category"].iloc[0].strip().lower()
        if "category" in products.columns
        else "laptop"
    )

    # Compute weights based on intent + category
    weights = _compute_weights(intent, category)

    # Normalize relevant columns
    df = products.copy()

    price_score = (
        _normalize_series(df["price_usd"], higher_is_better=False)
        if "price_usd" in df.columns
        else pd.Series([0.5] * len(df), index=df.index)
    )

    # VERY simple "performance" proxy:
    # - for laptops/tablets/phones we use RAM (and storage as a small boost)
    # - for monitors we treat performance as neutral
    if category in ("laptop", "tablet", "phone"):
        ram_score = (
            _normalize_series(df["ram_gb"], higher_is_better=True)
            if "ram_gb" in df.columns
            else pd.Series([0.5] * len(df), index=df.index)
        )
        storage_score = (
            _normalize_series(df["storage_gb"], higher_is_better=True)
            if "storage_gb" in df.columns
            else pd.Series([0.5] * len(df), index=df.index)
        )
        performance_score = 0.7 * ram_score + 0.3 * storage_score
    else:
        performance_score = pd.Series([0.5] * len(df), index=df.index)

    battery_score = (
        _normalize_series(df["battery_wh"], higher_is_better=True)
        if "battery_wh" in df.columns
        else pd.Series([0.5] * len(df), index=df.index)
    )

    screen_score = (
        _normalize_series(df["screen_inches"], higher_is_better=True)
        if "screen_inches" in df.columns
        else pd.Series([0.5] * len(df), index=df.index)
    )

    # Weighted overall score in [0, 1]
    df["score_continuous"] = (
        weights["price"] * price_score
        + weights["performance"] * performance_score
        + weights["battery"] * battery_score
        + weights["screen"] * screen_score
    )

    # Sort best → worst
    df_sorted = df.sort_values("score_continuous", ascending=False)

    # Build explanations with LLM
    goals_raw = intent.get("primary_goals") or []
    if isinstance(goals_raw, str):
        goals_list = [goals_raw]
    else:
        goals_list = [str(g) for g in goals_raw]

    results = []
    for _, row in df_sorted.iterrows():
        score_float = float(row["score_continuous"])
        try:
            explanation = _build_explanation(row, score_float, category, goals_list)
        except Exception as e:
            explanation = f"(Could not generate explanation: {e})"

        results.append(
            {
                "id": row.get("id"),
                "title": row.get("title") or row.get("model") or "Unknown product",
                "score": int(round(score_float * 100)),  # 0–100
                "explanation": explanation,
            }
        )

    return {
        "results": results,
        "category": category,
        "weights": weights,
    }
