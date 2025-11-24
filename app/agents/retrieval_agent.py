# app/agents/retrieval_agent.py

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from core.llm_client import chat_llm  # kept only for future LLM-enhanced retrieval

# -------------------------------------------------------------------
# Resolve the catalog path relative to root (as in your original code)
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT_DIR / "data" / "electronics_catalog.csv"

try:
    _catalog_df = pd.read_csv(CATALOG_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Could not load electronics catalog CSV at: {CATALOG_PATH}"
    ) from e

# Ensure numeric columns are properly typed
for col in ["price_usd", "ram_gb", "storage_gb", "battery_wh",
            "weight_kg", "screen_inches"]:
    if col in _catalog_df.columns:
        _catalog_df[col] = pd.to_numeric(_catalog_df[col], errors="coerce")

CATEGORY_KEYWORDS = {
    "laptop":  ["laptop", "notebook", "ultrabook"],
    "phone":   ["phone", "smartphone", "mobile"],
    "tablet":  ["tablet", "ipad", "tab"],
    "monitor": ["monitor", "screen", "display"],
}


def _get_float(intent: Dict[str, Any], keys) -> float | None:
    """Try multiple possible keys in the intent dict and parse a float.
       Returns None if nothing usable is found."""
    for k in keys:
        if k in intent and intent[k] is not None:
            try:
                raw = str(intent[k]).replace("$", "").strip()
                return float(raw)
            except Exception:
                continue
    return None


def detect_category(intent: Dict[str, Any], user_query: str) -> str:
    """Decide which category to use (priority: explicit → keywords → fallback)."""
    # 1) explicit
    for key in ("category", "device_type", "product_type"):
        val = intent.get(key)
        if isinstance(val, str) and val.strip():
            v = val.strip().lower()
            if v in CATEGORY_KEYWORDS:
                return v

    # 2) keyword in query
    q = user_query.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return cat

    # 3) default
    return "laptop"


def _apply_min_filter(df: pd.DataFrame, col: str, min_val: float | None) -> pd.DataFrame:
    """Helper: Filter df[col] >= min_val if col exists."""
    if min_val is None or col not in df.columns:
        return df
    return df[df[col].fillna(0) >= min_val]


def _apply_max_filter(df: pd.DataFrame, col: str, max_val: float | None) -> pd.DataFrame:
    """Helper: Filter df[col] <= max_val if col exists."""
    if max_val is None or col not in df.columns:
        return df
    return df[df[col].fillna(float("inf")) <= max_val]


def filter_products(intent: Dict[str, Any], user_query: str) -> pd.DataFrame:
    """Optimized filtering with fast pandas operations + capping results."""

    df = _catalog_df.copy()

    # ---- CATEGORY ----
    category = detect_category(intent, user_query)
    df = df[df["category"] == category]

    # ---- BUDGET ----
    budget = _get_float(intent, ["budget_max", "max_price", "budget", "price_cap"])
    if budget is not None:
        # allow 10% flexibility above budget
        df = _apply_max_filter(df, "price_usd", budget * 1.1)

    # ---- HARD CONSTRAINTS ----
    min_ram = _get_float(intent, ["min_ram_gb", "ram_min", "ram_gb_min", "min_ram"])
    df = _apply_min_filter(df, "ram_gb", min_ram)

    min_storage = _get_float(intent, ["min_storage_gb", "storage_min", "min_storage"])
    df = _apply_min_filter(df, "storage_gb", min_storage)

    min_battery = _get_float(intent, ["min_battery_wh", "battery_min", "battery_wh_min"])
    df = _apply_min_filter(df, "battery_wh", min_battery)

    # If everything filtered out, fallback to category-only
    if df.empty:
        df = _catalog_df[_catalog_df["category"] == category].copy()

    # ---- RELEVANCE SORTING (important for speed in Reasoner) ----
    df["price_usd"] = pd.to_numeric(df.get("price_usd", 0), errors="coerce").fillna(0)
    df["ram_gb"] = pd.to_numeric(df.get("ram_gb", 0), errors="coerce").fillna(0)
    df["battery_wh"] = pd.to_numeric(df.get("battery_wh", 0), errors="coerce").fillna(0)

    if budget:
        df["relevance_price"] = (df["price_usd"] - budget).abs()
    else:
        df["relevance_price"] = 0.0

    df = df.sort_values(
        by=["relevance_price", "ram_gb", "battery_wh"],
        ascending=[True, False, False],
    )

    # ---- LIMIT RESULTS FOR SPEED ----
    TOP_N = 40  # adjust to 20 if needed
    df = df.head(TOP_N).reset_index(drop=True)

    return df
