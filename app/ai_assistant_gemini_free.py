"""
ai_assistant_gemini_free.py
============================
Exactly the same as ai_assistant.py but uses Google Gemini FREE tier.

FREE limits (as of 2025):
  - 15 requests per minute
  - 1,500 requests per day
  - 0 rupees / 0 dollars cost

Setup (2 minutes):
  1. Go to https://aistudio.google.com
  2. Sign in with your Google account
  3. Click "Get API Key" -> "Create API key in new project"
  4. Copy the key (looks like: AIzaSy...)
  5. Put it in .streamlit/secrets.toml as:
         GEMINI_API_KEY = "AIzaSy..."
  6. pip install google-generativeai
"""

import streamlit as st
import google.generativeai as genai

BASE_SYSTEM_PROMPT = """
You are an expert assistant built into an Indian Forest Phenology Assessment app.
You help researchers and beginners understand:
- What phenology means (SOS, POS, EOS, LOS — Start/Peak/End/Length of Season)
- Why the app picks certain climate variables (like T2M) for modelling
- What R², MAE, LOO cross-validation, Ridge regression mean in plain language
- Why results can look odd (e.g. only 3 seasons, sparse met data, R²=-1)
- How to improve results (upload more years, use daily met data)
- What the NDVI graphs and correlation charts are showing
- What collinearity means and why it blocks CT2M from entering the model
- What the 15-day climate window is

Always use simple language first. Give concrete examples using actual numbers
the user sees in the app when possible. Be concise but thorough.
"""


def build_context_from_app_state(pheno_df=None, predictor=None,
                                  ndvi_info=None, met_info=None):
    lines = ["=== CURRENT APP RESULTS ==="]
    if ndvi_info:
        lines.append(f"NDVI: {ndvi_info.get('n_obs','?')} observations, "
                     f"{ndvi_info.get('year_range','?')}, "
                     f"cadence={ndvi_info.get('cadence_d','?')} days")
    if met_info:
        lines.append(f"Met parameters: {list(met_info.keys())}")
    if pheno_df is not None and len(pheno_df) > 0:
        lines.append(f"Seasons detected: {len(pheno_df)} ({list(pheno_df['Year'].values)})")
        for _, row in pheno_df.iterrows():
            yr  = int(row['Year'])
            sos = str(row.get('SOS_Date','?'))[:10]
            pos = str(row.get('POS_Date','?'))[:10]
            eos = str(row.get('EOS_Date','?'))[:10]
            los = row.get('LOS_Days','?')
            pk  = round(float(row.get('Peak_NDVI', 0)), 3)
            lines.append(f"  {yr}: SOS={sos}, POS={pos}, EOS={eos}, "
                         f"LOS={los}d, PeakNDVI={pk}")
    if predictor is not None:
        for ev in ['SOS', 'POS', 'EOS']:
            if ev not in predictor._fits:
                lines.append(f"{ev} model: not fitted")
                continue
            result   = predictor._fits[ev]
            best_fit = result['best_fit']
            r2   = round(predictor.r2.get(ev, 0), 3)
            mae  = round(predictor.mae.get(ev, 0), 1)
            feats = result.get('features', [])
            lines.append(f"{ev}: best={result['best_name']} R²={r2} "
                         f"MAE=±{mae}d features={feats}")
    return "\n".join(lines)


def ask_ai_gemini(question: str,
                  chat_history: list,
                  context_str: str = "",
                  api_key: str = "") -> str:
    """
    Send a question to Google Gemini and return the answer.
    Uses the FREE gemini-1.5-flash model.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=BASE_SYSTEM_PROMPT + (
            f"\n\n{context_str}" if context_str else ""
        ),
    )

    # Build conversation history for Gemini format
    history = []
    for turn in chat_history[-10:]:
        role = "user" if turn["role"] == "user" else "model"
        history.append({"role": role, "parts": [turn["content"]]})

    chat = model.start_chat(history=history)
    response = chat.send_message(question)
    return response.text


QUICK_QUESTIONS = [
    "Why is only T2M selected as predictor?",
    "What does R²(LOO) = -1.000 mean?",
    "Why only 3 seasons detected?",
    "How can I improve model accuracy?",
    "What is the 15-day climate window?",
    "Why does EOS go into next year?",
    "What is collinearity? Why blocks CT2M?",
    "What does MAE ± 9 days mean?",
]


def render_chat_tab(api_key: str,
                    pheno_df=None,
                    predictor=None,
                    ndvi_info=None,
                    met_info=None):
    """
    Call this inside a Streamlit tab.

    In your main app add this tab:
    --------------------------------
        tab1, ..., tab7 = st.tabs([..., "🤖 AI Assistant"])
        with tab7:
            from ai_assistant_gemini_free import render_chat_tab
            render_chat_tab(
                api_key   = st.secrets.get("GEMINI_API_KEY", ""),
                pheno_df  = st.session_state.get("pheno_df"),
                predictor = st.session_state.get("predictor"),
                ndvi_info = st.session_state.get("ndvi_info"),
                met_info  = st.session_state.get("met_info"),
            )
    """
    st.markdown("### 🤖 AI Research Assistant  *(powered by Google Gemini — free)*")
    st.markdown(
        "Ask me anything about your phenology results, model choices, "
        "or how to interpret the outputs."
    )

    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    context_str = build_context_from_app_state(
        pheno_df=pheno_df, predictor=predictor,
        ndvi_info=ndvi_info, met_info=met_info,
    )

    st.markdown("**Quick questions — click any to ask instantly:**")
    cols = st.columns(4)
    for i, q in enumerate(QUICK_QUESTIONS):
        if cols[i % 4].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")

    for turn in st.session_state["ai_chat_history"]:
        role  = turn["role"]
        label = "You" if role == "user" else "AI Assistant"
        bg    = "#f0f7ff" if role == "user" else "#f0fff4"
        st.markdown(
            f"<div style='background:{bg};padding:12px 16px;"
            f"border-radius:8px;margin:6px 0'>"
            f"<b>{label}:</b><br>{turn['content']}</div>",
            unsafe_allow_html=True,
        )

    pending    = st.session_state.pop("pending_question", "")
    user_input = st.text_input(
        "Type your question:",
        value=pending,
        placeholder="e.g. Why is my POS R² negative?",
        key="ai_input_box",
    )

    col_send, col_clear = st.columns([1, 5])
    send  = col_send.button("Send", type="primary")
    clear = col_clear.button("Clear chat")

    if clear:
        st.session_state["ai_chat_history"] = []
        st.rerun()

    if send and user_input.strip():
        with st.spinner("Thinking…"):
            try:
                answer = ask_ai_gemini(
                    question     = user_input.strip(),
                    chat_history = st.session_state["ai_chat_history"],
                    context_str  = context_str,
                    api_key      = api_key,
                )
            except Exception as e:
                answer = (
                    f"Error: {e}\n\n"
                    "Check that your GEMINI_API_KEY is correct in secrets.toml "
                    "and you have run: pip install google-generativeai"
                )

        st.session_state["ai_chat_history"].append(
            {"role": "user",      "content": user_input.strip()}
        )
        st.session_state["ai_chat_history"].append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()