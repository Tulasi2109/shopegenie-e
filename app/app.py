# app/app.py

import streamlit as st
from core.orchestrator import run_pipeline
from core.llm_client import chat_llm   # GenAI client

# =========================
#   Card renderer
# =========================

def render_product_card(rank_index: int, rec: dict, products_df):
    row = None
    try:
        matched = products_df[products_df["id"] == rec.get("id")]
        if not matched.empty:
            row = matched.iloc[0]
    except Exception:
        row = None

    if row is not None:
        specs_line = (
            f"💰 ${row.get('price_usd', 'N/A')} • "
            f"🧠 {row.get('ram_gb', 'N/A')} GB RAM • "
            f"💾 {row.get('storage_gb', 'N/A')} GB storage • "
            f"🔋 {row.get('battery_wh', 'N/A')} Wh battery • "
            f"⚖️ {row.get('weight_kg', 'N/A')} kg"
        )
    else:
        specs_line = "Specs not available in table."

    st.markdown(
        f"""
        <div style="
            border-radius: 18px;
            padding: 18px 20px;
            margin: 10px 0 14px 0;
            background-color: #0f172a;
            border: 1px solid #1f2937;
        ">
            <h3 style="margin: 0 0 8px 0; color:#f9fafb;">
                {rank_index}. {rec.get('title', 'Unknown Product')}
                <span style="font-size: 0.9rem; color: #9CA3AF;">
                    (Score: {rec.get('score', 'N/A')})
                </span>
            </h3>
            <p style="margin: 0 0 8px 0; color: #e5e7eb;">
                {specs_line}
            </p>
            <p style="margin: 0; color: #e5e7eb;">
                {rec.get('explanation', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
#   Page config + CSS
# =========================

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

    /* CTA button */
    div.stButton > button {
        background: #2563eb;
        color: #ffffff;
        border-radius: 999px;
        padding: 0.9rem 2.1rem;
        font-size: 1.05rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(37, 99, 235, 0.45);
    }

    /* Animated Search Bar */
    div[data-baseweb="input"] {
        border-radius: 999px !important;
        overflow: hidden;
        transition: box-shadow 0.25s ease, transform 0.2s ease;
        box-shadow: 0 4px 10px rgba(15,23,42,0.08);
        background: #ffffff;
        animation: genie-breathe 3s ease-in-out infinite;
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
        animation: none;
    }

    @keyframes genie-breathe {
        0%   { box-shadow: 0 4px 10px rgba(15,23,42,0.08); }
        50%  { box-shadow: 0 8px 18px rgba(37,99,235,0.18); }
        100% { box-shadow: 0 4px 10px rgba(15,23,42,0.08); }
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

# =========================
#   HERO SECTION
# =========================

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

        user_query = st.text_input(
            label="",
            placeholder="e.g. best laptop under $1000 for data analyst",
            label_visibility="collapsed",
            key="hero_query",
        )

        generate_clicked = st.button("Get Recommendations", key="hero_button")

    with right:
        try:
            st.image("app/assets/genie.png", use_container_width=True)
        except Exception:
            st.write("🧞‍♂️ Genie illustration goes here.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
#   RESULTS SECTION
# =========================

results_container = st.container()

if generate_clicked:
    if not user_query.strip():
        results_container.error("Please enter a query.")
    else:
        with results_container:
            st.subheader("Results")
            st.info("Running multi-agent reasoning… Please wait.")
            with st.spinner("Agents thinking…"):
                try:
                    intent, products, ranking = run_pipeline(user_query)

                    results = ranking.get("results", [])
                    summary_text = ""

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

                    if products is not None and not products.empty:
                        st.markdown("### 📦 Candidate Products (After Filters)")
                        st.dataframe(products, use_container_width=True)
                    else:
                        st.warning("No candidate products found after filtering.")

                    st.markdown("### ⭐ Ranked Recommendations")
                    if results:
                        for i, rec in enumerate(results, start=1):
                            render_product_card(i, rec, products)
                    else:
                        st.warning("No ranked recommendations returned.")

                except Exception as e:
                    st.error(f"Error: {e}")
