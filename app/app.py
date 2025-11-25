# app/app.py

import math
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from core.orchestrator import run_pipeline
from core.llm_client import chat_llm   # GenAI client


# =====================================================================
#   Helper: Product Card Renderer
# =====================================================================

def render_product_card(
    rank_index: int,
    rec: Dict[str, Any],
    products_df: pd.DataFrame,
    user_query: str,
    intent: Dict[str, Any],
):
    """
    Render a single recommendation card + 'Ask Genie why this fits me' follow-up.
    """

    # Try to match this recommendation back to the products dataframe
    row = None
    try:
        matched = products_df[products_df["id"] == rec.get("id")]
        if not matched.empty:
            row = matched.iloc[0]
    except Exception:
        row = None

    # Specs line for the compact line under title
    if row is not None:
        specs_line = (
            f"💰 ${row.get('price_usd', 'N/A')} • "
            f"🧠 {row.get('ram_gb', 'N/A')} GB RAM • "
            f"💾 {row.get('storage_gb', 'N/A')} GB storage • "
            f"🔋 {row.get('battery_wh', 'N/A')} Wh battery • "
            f"📺 {row.get('screen_inches', 'N/A')}-inch display • "
            f"⚖️ {row.get('weight_kg', 'N/A')} kg"
        )
    else:
        specs_line = (
            f"💰 ${rec.get('price_usd', 'N/A')} • "
            f"🧠 {rec.get('ram_gb', 'N/A')} GB RAM • "
            f"💾 {rec.get('storage_gb', 'N/A')} GB storage • "
            f"🔋 {rec.get('battery_wh', 'N/A')} Wh battery • "
            f"📺 {rec.get('screen_inches', 'N/A')}-inch display"
        )

    # Main card UI
    st.markdown(
        f"""
        <div style="
            border-radius: 24px;
            padding: 18px 22px;
            margin: 10px 0 14px 0;
            background-color: #020617;
            border: 1px solid #1f2937;
        ">
            <h3 style="margin: 0 0 10px 0; color:#f9fafb; font-size:1.35rem;">
                {rank_index}. {rec.get('title', 'Unknown Product')}
                <span style="font-size: 0.9rem; color: #9CA3AF;">
                    (Score: {rec.get('score', 'N/A')})
                </span>
            </h3>
            <p style="margin: 0 0 10px 0; color: #e5e7eb; font-size:0.95rem;">
                {specs_line}
            </p>
            <p style="margin: 0; color: #e5e7eb; font-size:0.98rem;">
                {rec.get('explanation', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------
    # Ask Genie why this fits me
    # ------------------------------
    if "why_fit" not in st.session_state:
        st.session_state["why_fit"] = {}

    follow_key = f"why_{rec.get('id', rank_index)}"

    ask_clicked = st.button(
        "🤔 Ask Genie why this fits me",
        key=follow_key,
        use_container_width=False,
    )

    if ask_clicked:
        try:
            follow_prompt = f"""
You are ShopGenie-E, an expert electronics assistant.

The user originally asked:
\"\"\"{user_query}\"\"\"

Parsed intent (JSON-like):
{intent}

Selected product (JSON-like):
{rec}

Write a short, very clear explanation of WHY this specific product was recommended.

Requirements:
- Start the answer with: "This was recommended because..."
- Explicitly connect what the user asked for to this product's strengths
  (e.g., performance, battery, portability, screen size, price).
- Mention 2–3 concrete benefits in natural language
  (e.g., "great for multitasking", "comfortable 15.6-inch screen", "battery that lasts a full workday").
- Mention exactly ONE honest trade-off (heavier, more expensive, smaller screen, etc.).
- Do NOT repeat the numeric specs or the word "score"; interpret them for the user instead.
- Keep it 3–4 sentences, conversational and easy to understand.
"""
            answer = chat_llm(follow_prompt)
        except Exception as e:
            answer = f"(Could not contact Genie for a follow-up explanation: {e})"
        st.session_state["why_fit"][follow_key] = answer

    if follow_key in st.session_state["why_fit"]:
        st.markdown(
            f"""
            <div style="
                margin: 6px 0 18px 0;
                padding: 10px 14px;
                border-radius: 12px;
                background-color: #e5f0ff;
                border: 1px solid #bfdbfe;
            ">
                <p style="color:#111827; font-size:0.9rem; margin:0;">
                    <strong>Genie says:</strong> {st.session_state['why_fit'][follow_key]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)


# =====================================================================
#   Page config + Global CSS
# =====================================================================

st.set_page_config(
    page_title="SHOPGENIE-E",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9fafb;
        color: #111827;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    h1, h2, h3 {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .hero-wrapper {
        background: transparent !important;
        padding: 0rem;
        margin-bottom: 2rem;
    }

    .hero-badge {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-size: 2.6rem;
        line-height: 1.1;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.8rem;
    }

    /* Search bar */
    div[data-baseweb="input"] {
        border-radius: 999px !important;
        overflow: hidden;
        transition: box-shadow 0.25s ease, transform 0.2s ease;
        box-shadow: 0 4px 10px rgba(15,23,42,0.08);
        background: #ffffff;
    }

    div[data-baseweb="input"] > div > input {
        border-radius: 999px !important;
        padding: 0.9rem 1.2rem;
        font-size: 0.98rem;
    }

    div[data-baseweb="input"]:focus-within {
        box-shadow: 0 0 0 3px rgba(37,99,235,0.35),
                     0 18px 30px rgba(15,23,42,0.18);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# TOP WHITE HEADER BAR
# ---------------------------------------------------------

st.markdown(
    """
    <div style="
        background-color: #ffffff;
        padding: 28px 40px;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        margin-bottom: 35px;
        text-align: center;
    ">
        <h1 style="
            margin: 0;
            font-size: 42px;
            font-weight: 800;
            color: #0f172a;
            text-transform: uppercase;
        ">
            🧞‍♂️ SHOPGENIE-E
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
#   HERO SECTION (no recent searches)
# =====================================================================

hero = st.container()

with hero:
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="hero-badge">🧞‍♂️ ShopGenie-E</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">Explainable AI-powered electronics recommendation system</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-subtitle'>Describe what you're looking for and let the genie compare laptops, phones, tablets and monitors for you.</div>",
            unsafe_allow_html=True,
        )

        # Text input with persistence via session_state
        user_query = st.text_input(
            label="",
            placeholder="e.g. best laptop under $1000 for data analyst",
            label_visibility="collapsed",
            key="hero_query",
        )

        # Main CTA button only (no recent search chips) — default Streamlit style
        generate_clicked = st.button("Get Recommendations", key="hero_button")

    with right:
        try:
            st.image("app/assets/genie.png", use_container_width=True)
        except Exception:
            st.write("🧞‍♂️ Genie illustration goes here.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
#   RESULTS SECTION
# =====================================================================

results_container = st.container()

if generate_clicked:
    if not st.session_state.get("hero_query", "").strip():
        results_container.error("Please enter a query.")
    else:
        user_query = st.session_state["hero_query"].strip()

        with results_container:
            st.subheader("Results")
            st.info("Running multi-agent reasoning…")

            with st.spinner("Agents thinking…"):
                try:
                    intent, products, ranking = run_pipeline(user_query)

                    results: List[Dict[str, Any]] = ranking.get("results", [])
                    summary_text = ""

                    # ------------------------------
                    # Short AI shopping summary
                    # ------------------------------
                    if results:
                        try:
                            summary_prompt = f"""
You are an expert electronics shopping assistant.

User query: {user_query}
Intent: {intent}
Top results (truncated): {results[:3]}

In 2–3 clean sentences:
• Who these products are ideal for
• Why the #1 product fits best
• One trade-off the user should know
"""
                            summary_text = chat_llm(summary_prompt)
                        except Exception as e:
                            summary_text = f"(AI summary error: {e})"

                    if summary_text:
                        st.markdown("### 🧾 AI Shopping Summary")
                        st.markdown(
                            f"<p style='color:#111827; font-size:1rem;'>{summary_text}</p>",
                            unsafe_allow_html=True,
                        )

                    # ------------------------------
                    # Candidate products table
                    # ------------------------------
                    if products is not None and not products.empty:
                        st.markdown("### 📦 Candidate Products (After Filters)")

                        # Dark-style dataframe to match cards
                        styled_products = (
                            products.style
                            .set_properties(
                                **{
                                    "background-color": "#020617",
                                    "color": "#e5e7eb",
                                    "border-color": "#1f2937",
                                }
                            )
                            .set_table_styles(
                                [
                                    {
                                        "selector": "th",
                                        "props": [
                                            ("background-color", "#020617"),
                                            ("color", "#e5e7eb"),
                                        ],
                                    }
                                ]
                            )
                        )
                        st.dataframe(styled_products, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No candidate products found after filtering.")

                    # ------------------------------
                    # Ranked recommendations
                    # ------------------------------
                    st.markdown("### ⭐ Ranked Recommendations")
                    if results:
                        for i, rec in enumerate(results, start=1):
                            render_product_card(i, rec, products, user_query, intent)
                    else:
                        st.warning("No ranked recommendations returned.")

                except Exception as e:
                    st.error(f"Error: {e}")
