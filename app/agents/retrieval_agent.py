# app/agents/retrieval_agent.py

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from core.llm_client import chat_llm  # kept in case you want LLM-based retrieval later

# -------------------------------------------------------------------
# Resolve the catalog path relative to the project root
# Project root = folder that contains "app" and "data"
# e.g. D:\shopgenie-e\
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT_DIR / "data" / "electronics_catalog.csv"

try:
    # Load catalog once at import time
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
    """
    Try multiple possible keys in the intent dict and parse a float.
    Returns None if nothing usable is found.
    """
    for k in keys:
        if k in intent and intent[k] is not None:
            try:
                raw = str(intent[k])
                raw = raw.replace("$", "").strip()  # basic cleanup
                return float(raw)
            except Exception:
                continue
    return None


def detect_category(intent: Dict[str, Any], user_query: str) -> str:
    """
    Decide which category to use (laptop / phone / tablet / monitor).
    Priority:
      1) explicit category/device_type in the intent (if provided)
      2) keyword matching on the raw user_query
      3) default to 'laptop'
    """
    # 1) from intent keys if present
    for key in ("category", "device_type", "product_type"):
        val = intent.get(key)
        if isinstance(val, str) and val.strip():
            normalized = val.strip().lower()
            if normalized in CATEGORY_KEYWORDS:
                return normalized

    # 2) from user_query keywords
    q = (user_query or "").lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return cat

    # 3) fallback
    return "laptop"


def filter_products(intent: Dict[str, Any], user_query: str) -> pd.DataFrame:
    """
    Filter the unified electronics catalog based on:
      - detected category (laptop / phone / tablet / monitor)
      - budget (if available in intent)
      - min RAM, storage, etc. when relevant
    """
    df = _catalog_df.copy()

    # ---- CATEGORY FILTERING ----
    category = detect_category(intent, user_query)
    df = df[df["category"] == category]

    # ---- BUDGET FILTERING ----
    budget_max = _get_float(intent, ["budget_max", "max_price",
                                     "budget", "price_cap"])
    if budget_max is not None and "price_usd" in df.columns:
        df = df[df["price_usd"] <= budget_max]

    # ---- MIN RAM FILTERING (mainly for laptops & tablets) ----
    min_ram = _get_float(intent, ["min_ram_gb", "ram_min",
                                  "ram_gb_min", "min_ram"])
    if min_ram is not None and "ram_gb" in df.columns:
        df = df[df["ram_gb"].fillna(0) >= min_ram]

    # ---- MIN STORAGE FILTERING ----
    min_storage = _get_float(intent, ["min_storage_gb",
                                      "storage_min", "min_storage"])
    if min_storage is not None and "storage_gb" in df.columns:
        df = df[df["storage_gb"].fillna(0) >= min_storage]

    # ---- MIN BATTERY FILTER (for portable devices) ----
    min_battery = _get_float(intent, ["min_battery_wh",
                                      "battery_min", "battery_wh_min"])
    if min_battery is not None and "battery_wh" in df.columns:
        df = df[df["battery_wh"].fillna(0) >= min_battery]

    # If, after filters, df is empty, fall back to unfiltered-for-this-category
    if df.empty:
        df = _catalog_df[_catalog_df["category"] == category]

    return df
