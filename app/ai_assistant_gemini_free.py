"""
AI Assistant Tab — Google Gemini Free Tier
==========================================
Auto-detects available Gemini models — future-proof against model name changes.
Place this file next to your main app file.
Requires: pip install google-generativeai
"""

import streamlit as st
import google.generativeai as genai


# ─── AUTO-DETECT BEST AVAILABLE MODEL ────────────────────────

def get_best_model(api_key):
    """
    Automatically detect the best available Gemini model.
    Priority: flash > pro > any available.
    Returns model name string, or None if no models found.
    """
    try:
        genai.configure(api_key=api_key.strip())

        # Get all available models that support generateContent
        available = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available.append(m.name)

        if not available:
            return None

        # Priority order — most preferred first
        # Priority: gemini-1.5-flash has the best free tier limits
        # gemini-2.0-flash has limit:0 on many free accounts — avoid it
        preferred = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
        ]

        # Only keep genuine Gemini models (skip Gemma, embedding, etc.)
        # Also skip gemini-2.0 and gemini-2.5 which have limit:0 on free tier
        skip_models = ["gemini-2.0", "gemini-2.5", "gemma", "embedding", "aqa", "text-"]
        available = [m for m in available
                     if "gemini" in m.lower()
                     and not any(skip in m.lower() for skip in skip_models)]

        # Check preferred list first
        for pref in preferred:
            for avail in available:
                if pref in avail:
                    return avail

        # If none of preferred found, return first available gemini model
        if available:
            return available[0]
        # Last resort fallback
        return "gemini-1.5-flash"

    except Exception:
        # If listing fails, return best known default
        return "gemini-1.5-flash"


# ─── BUILD CONTEXT FROM APP STATE ────────────────────────────

def build_context_from_app_state(pheno_df, predictor, ndvi_info, met_info):
    """Build a context string from the current app session state."""
    import pandas as pd
    lines = []
    lines.append("=== PHENOLOGY APP CONTEXT ===")

    if ndvi_info:
        lines.append(f"\nNDVI Data:")
        lines.append(f"  Year range: {ndvi_info.get('year_range', 'N/A')}")
        lines.append(f"  Observations: {ndvi_info.get('n_obs', 'N/A')}")
        lines.append(f"  Mean NDVI: {ndvi_info.get('ndvi_mean', 'N/A')}")
        lines.append(f"  NDVI range (P5-P95): {ndvi_info.get('ndvi_p5','N/A')} to {ndvi_info.get('ndvi_p95','N/A')}")
        lines.append(f"  Evergreen index: {ndvi_info.get('evergreen_index', 'N/A')}")

    if met_info:
        lines.append(f"\nMeteorological Parameters: {', '.join(met_info.keys())}")

    if pheno_df is not None and len(pheno_df) > 0:
        lines.append(f"\nExtracted Seasons ({len(pheno_df)} total):")
        for _, row in pheno_df.iterrows():
            sos = row.get('SOS_Date')
            pos = row.get('POS_Date')
            eos = row.get('EOS_Date')
            los = row.get('LOS_Days', 'N/A')
            pk  = row.get('Peak_NDVI', 'N/A')
            sos_str = pd.Timestamp(sos).strftime('%b %d') if pd.notna(sos) else '?'
            pos_str = pd.Timestamp(pos).strftime('%b %d') if pd.notna(pos) else '?'
            eos_str = pd.Timestamp(eos).strftime('%b %d %Y') if pd.notna(eos) else '?'
            pk_val  = round(float(pk), 3) if pk != 'N/A' else 'N/A'
            lines.append(f"  Year {int(row['Year'])}: SOS={sos_str}, POS={pos_str}, EOS={eos_str}, LOS={los}d, Peak NDVI={pk_val}")

    if predictor is not None and hasattr(predictor, '_fits') and predictor._fits:
        lines.append(f"\nPredictive Models:")
        for ev in ['SOS', 'POS', 'EOS']:
            if ev not in predictor._fits:
                continue
            result  = predictor._fits[ev]
            feats   = result.get('features', [])
            r2      = predictor.r2.get(ev, 'N/A')
            mae     = predictor.mae.get(ev, 'N/A')
            model   = result.get('best_name', 'N/A')
            r2_str  = round(r2,  3) if isinstance(r2,  float) else r2
            mae_str = round(mae, 1) if isinstance(mae, float) else mae
            lines.append(f"  {ev}: model={model}, features={feats}, R²={r2_str}, MAE=±{mae_str} days")

    lines.append("\n=== END CONTEXT ===")
    return "\n".join(lines)


# ─── CALL GEMINI API ─────────────────────────────────────────

def ask_gemini(question, chat_history, context_str, api_key):
    """Send question to Gemini using auto-detected best model."""

    if not api_key or api_key.strip() == "":
        return "❌ No API key found. Please add GEMINI_API_KEY to .streamlit/secrets.toml and restart."

    try:
        genai.configure(api_key=api_key.strip())

        # Auto-detect best available model
        model_name = get_best_model(api_key)
        if model_name is None:
            return "❌ No Gemini models available for your API key. Check aistudio.google.com."

        system_prompt = f"""You are an expert AI assistant for a forest phenology analysis app.
You help users understand their NDVI and climate data, phenology results (SOS/POS/EOS/LOS),
model performance, and what the results mean for Indian forest ecosystems.

Current data context from the app:
{context_str}

Be concise, friendly, and specific to the user's data when possible.
If no data is loaded yet, guide the user to upload their files first.
"""
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )

        # Build conversation history
        history = []
        for msg in chat_history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history)
        response = chat.send_message(question)

        # Store which model was used (for display)
        st.session_state['_gemini_model_used'] = model_name
        return response.text

    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "invalid" in err.lower():
            return f"❌ API key is invalid. Go to aistudio.google.com and create a new key.\n\nDetail: {err}"
        elif "quota" in err.lower() or "rate" in err.lower():
            return f"⚠️ Rate limit reached. Wait 1 minute and try again.\n\nDetail: {err}"
        elif "404" in err or "not found" in err.lower():
            # Model not found — clear cached model and retry
            if '_gemini_model_used' in st.session_state:
                del st.session_state['_gemini_model_used']
            return f"⚠️ Model not available. Please click Send again — the app will auto-detect a new model.\n\nDetail: {err}"
        else:
            return f"❌ Error: {err}"


# ─── RENDER THE CHAT TAB ─────────────────────────────────────

def render_chat_tab(api_key, pheno_df, predictor, ndvi_info, met_info):
    """Render the full AI Assistant tab inside Streamlit."""

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
                padding: 20px 28px; border-radius: 14px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0 0 6px; font-size: 1.5rem;'>🤖 AI Assistant</h2>
        <p style='color: #C8E6C9; margin: 0; font-size: 0.9rem;'>
            Ask any question about your phenology results, data quality, model performance,
            or what the findings mean for your forest site.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── API key status ────────────────────────────────────────
    if not api_key or api_key.strip() == "":
        st.error("""
**API Key Missing** — The AI Assistant needs a free Google Gemini API key.

Steps to fix:
1. Go to aistudio.google.com
2. Click **Get API Key** → **Create API key**
3. Copy the key (starts with AIzaSy...)
4. Open `.streamlit/secrets.toml` in your project folder
5. Make sure it contains exactly:  `GEMINI_API_KEY = "your_key_here"`
6. Restart the app: press Ctrl+C then run again
        """)
        return

    # ── Show API key + auto-detected model ───────────────────
    col_k, col_m = st.columns(2)
    with col_k:
        st.success(f"✅ API key loaded (`{api_key.strip()[:12]}...`)")
    with col_m:
        # Show which model will be used
        if '_gemini_model_used' in st.session_state:
            st.info(f"🤖 Model: `{st.session_state['_gemini_model_used']}`")
        else:
            with st.spinner("Detecting available models..."):
                detected = get_best_model(api_key)
            if detected:
                st.session_state['_gemini_model_used'] = detected
                st.info(f"🤖 Model: `{detected}`")
            else:
                st.warning("⚠️ Could not detect model")

    # ── Context status ────────────────────────────────────────
    context_str = build_context_from_app_state(pheno_df, predictor, ndvi_info, met_info)
    if pheno_df is not None and len(pheno_df) > 0:
        st.info(f"📊 Context loaded: **{len(pheno_df)} seasons** of phenology data available.")
    else:
        st.warning("⚠️ No phenology data loaded yet. Upload NDVI and met files first, then come back here.")

    # ── Initialise chat history ───────────────────────────────
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # ── Quick question buttons ────────────────────────────────
    st.markdown("**Quick questions:**")
    quick_cols = st.columns(4)
    quick_questions = [
        "Summarise my results",
        "Is my data quality good?",
        "What does SOS trend mean?",
        "How to improve accuracy?",
        "Why only T2M as predictor?",
        "What is LOO R²?",
        "How many seasons needed?",
        "Explain peak NDVI value",
    ]
    for i, qq in enumerate(quick_questions):
        with quick_cols[i % 4]:
            if st.button(qq, key=f"qq_{i}", use_container_width=True):
                st.session_state.ai_chat_history.append({"role": "user", "content": qq})
                with st.spinner("Thinking..."):
                    answer = ask_gemini(qq, st.session_state.ai_chat_history, context_str, api_key)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown("---")

    # ── Chat history display ──────────────────────────────────
    for msg in st.session_state.ai_chat_history:
        if msg["role"] == "user":
            st.markdown(f"""<div style='background:#E3F2FD;padding:12px 16px;border-radius:10px;
                border-left:4px solid #1976D2;margin:8px 0;'>
                <b>You:</b><br>{msg["content"]}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style='background:#E8F5E9;padding:12px 16px;border-radius:10px;
                border-left:4px solid #2E7D32;margin:8px 0;'>
                <b>AI Assistant:</b><br>{msg["content"]}</div>""", unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────
    st.markdown("**Type your question:**")
    user_input = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. What does my SOS trend tell me about climate change?",
        key="ai_user_input"
    )

    col_send, col_clear, col_refresh = st.columns([2, 1, 1])
    with col_send:
        send_clicked = st.button("Send", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.ai_chat_history = []
            st.rerun()
    with col_refresh:
        if st.button("🔄 Re-detect model", use_container_width=True,
                     help="Click if AI stops working — detects the latest available Gemini model"):
            if '_gemini_model_used' in st.session_state:
                del st.session_state['_gemini_model_used']
            st.rerun()

    if send_clicked and user_input.strip():
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking..."):
            answer = ask_gemini(
                user_input.strip(),
                st.session_state.ai_chat_history,
                context_str,
                api_key
            )
        st.session_state.ai_chat_history.append({"role": "assistant", "content": answer})
        st.rerun()
