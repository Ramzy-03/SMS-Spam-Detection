import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
)

MODEL_PATHS = [Path("Saved_model/spam_pipeline.pkl"), Path("model/spam_pipeline.pkl")]
ENCODER_PATHS = [Path("Saved_model/label_encoder.pkl"), Path("model/label_encoder.pkl")]
DATA_PATH = Path("Dataset/spam.csv")

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_model():
    for model_path, encoder_path in zip(MODEL_PATHS, ENCODER_PATHS):
        if model_path.exists() and encoder_path.exists():
            pipeline = joblib.load(model_path)
            encoder = joblib.load(encoder_path)
            return pipeline, encoder, model_path.parent
    return None, None, None

@st.cache_data
def load_dataset_stats():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1], names=["label", "message"], header=0)
    df = df.dropna(subset=["message"])
    df["label"] = df["label"].astype(str).str.lower()

    spam_count = int((df["label"] == "spam").sum())
    ham_count = int((df["label"] != "spam").sum())

    return {
        "total": len(df),
        "spam": spam_count,
        "ham": ham_count,
        "spam_pct": (spam_count / len(df)) * 100,
        "sample_spam": df.loc[df["label"] == "spam", "message"].sample(1, random_state=42).iloc[0],
        "sample_ham": df.loc[df["label"] != "spam", "message"].sample(1, random_state=42).iloc[0],
    }


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def render_styles():
    st.markdown(
        """
        <style>
            .hero-banner {border-radius: 24px; padding: 28px; background: linear-gradient(135deg, #142850 0%, #27496d 100%); color: white;}
            .hero-heading {font-size: 2.9rem; font-weight: 800; letter-spacing: -0.04em; margin-bottom: 0.2rem;}
            .hero-subtitle {font-size: 1.05rem; opacity: 0.88; line-height: 1.65;}
            .shimmer {position: relative; overflow: hidden;}
            .shimmer::after {content: ""; position: absolute; top: 0; left: -100%; width: 120%; height: 100%; background: linear-gradient(120deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.18) 50%, rgba(255,255,255,0) 100%); animation: shimmer 2.8s infinite;}
            @keyframes shimmer {100% {left: 100%;}}
            .feature-card {border-radius: 20px; padding: 24px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.08);}
            .metric-small {color: #adb5bd;}
        </style>
        """,
        unsafe_allow_html=True,
    )


pipeline, encoder, model_folder = load_model()
stats = load_dataset_stats()
render_styles()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero-banner shimmer'>"
    "<div class='hero-heading'>SMS Spam Detection — Faster, Safer, Smarter</div>"
    "<div class='hero-subtitle'>Check incoming SMS content with an explainable machine learning pipeline that detects spam and gives confidence insights.</div>"
    "</div>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1], gap="large")
with col1:
    st.markdown("### Enter any SMS message below")
    user_input = st.text_area(
        "Message",
        value=st.session_state.get("current_message", ""),
        height=180,
        placeholder="Type or paste an SMS message here...",
    )

    example = st.selectbox(
        "Try an example message",
        [
            "Choose one...",
            "WINNER!! You have been selected for a £500 prize. Call 09061701461 now!",
            "Hey, are you coming to the meeting at 3pm today?",
            "Urgent! Your account has been suspended. Verify now at http://secure.example.com",
        ],
    )
    if example != "Choose one...":
        user_input = example
        st.session_state.current_message = example

with col2:
    st.markdown("### Project status")
    if pipeline is None or encoder is None:
        st.error("Model not found. Run the training script to create the saved model files.")
        st.caption("Expected paths: Saved_model/spam_pipeline.pkl and Saved_model/label_encoder.pkl")
    else:
        st.success(f"Loaded model from `{model_folder}`")
        st.caption("TF-IDF + Logistic Regression pipeline ready to classify new text.")

    if stats:
        st.metric("Total examples", f"{stats['total']:,}")
        st.metric("Spam ratio", f"{stats['spam_pct']:.1f}%")
        st.markdown("**Quick dataset preview**")
        st.write(f"💬 Example spam: {stats['sample_spam'][:80]}…")
        st.write(f"💬 Example ham: {stats['sample_ham'][:80]}…")

st.divider()

if st.button("🔎 Analyze message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    elif pipeline is None or encoder is None:
        st.error("Unable to analyze because the model is missing. Run `python train_model.py` first.")
    else:
        cleaned = clean_text(user_input)
        with st.spinner("Analyzing message with the spam detector..."):
            pred = pipeline.predict([cleaned])[0]
            label = encoder.inverse_transform([pred])[0]
            probability = None
            try:
                probs = pipeline.predict_proba([cleaned])[0]
                spam_index = int(encoder.transform(["spam"])[0])
                probability = float(probs[spam_index] * 100)
            except Exception:
                probability = None

        is_spam = label == "spam"
        if is_spam:
            st.error("🚨 SPAM detected")
        else:
            st.success("✅ Message looks HAM")
            st.balloons()

        if probability is not None:
            st.metric("Spam confidence", f"{probability:.1f}%")
            st.progress(min(max(int(probability), 0), 100))
        else:
            st.info("Confidence score is not available for this model.")

        with st.expander("🔍 Analysis details"):
            st.write("**Cleaned text processed by the model:**")
            st.code(cleaned)
            st.write(f"**Predicted label:** {label.upper()}")
            if probability is not None:
                st.write(f"**Spam probability:** {probability:.1f}%")

st.divider()

st.markdown(
    "### How it works"
    "\n" 
    "- The app cleans and normalizes SMS text before classification."
    "\n" 
    "- A TF-IDF transformer converts text into numeric features."
    "\n" 
    "- Logistic Regression predicts whether the message is spam or ham."
)

st.caption("Built with Streamlit, scikit-learn, and real SMS spam dataset signals.")
