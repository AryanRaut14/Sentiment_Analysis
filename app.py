import os
import re
import pickle
import time
import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Tweet Sentiment & Dual AI Insights",
    page_icon="⚡",
    layout="wide"
)

# --- 1. Load ML Artifacts ---
@st.cache_resource
def load_ml_assets():
    model_path = "models/sentiment_model.pkl" if os.path.exists("models/sentiment_model.pkl") else "../models/sentiment_model.pkl"
    vec_path = "models/vectorizer.pkl" if os.path.exists("models/vectorizer.pkl") else "../models/vectorizer.pkl"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

try:
    model, vectorizer = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure .pkl files are in 'models/'.")
    st.stop()

# --- 2. Initialize API Clients ---
@st.cache_resource
def init_ai_clients():
    gemini_client = None
    groq_client = None

    # Gemini
    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY")
        if gemini_key:
            gemini_client = genai.Client(api_key=gemini_key)
    except Exception:
        pass

    # Groq
    try:
        groq_key = st.secrets.get("GROQ_API_KEY")
        if groq_key:
            groq_client = Groq(api_key=groq_key)
    except Exception:
        pass

    return gemini_client, groq_client

gemini_client, groq_client = init_ai_clients()

# --- 3. Preprocessing & Sentiment Inference ---
def preprocess_text(text: str) -> str:
    cleaned = re.sub(r"http\S+|@\S+|#\S+|[^\w\s]", " ", text)
    return cleaned.lower().strip()

def predict_sentiment(text: str):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    confidence = float(max(probabilities))
    return sentiment, confidence

# --- 4. Resilient Multi-Provider Summarizer ---
def generate_summary(text: str) -> tuple[str, str]:
    """
    Attempts summarization with Gemini first.
    Falls back to Groq (LLaMA 3.3 70B) if rate-limited or unavailable.
    Returns: (summary_text, provider_used)
    """
    prompt = f"Provide a concise 1-2 sentence summary of what this tweet is about: \"{text}\""

    # Attempt 1: Gemini API
    if gemini_client:
        for gemini_model in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                response = gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip(), f"Gemini ({gemini_model})"
            except Exception:
                continue  # Fallthrough on error/rate limit

    # Attempt 2: Groq LLaMA 3 Fallback
    if groq_client:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a concise text summarizer."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=100
            )
            summary = chat_completion.choices[0].message.content.strip()
            return summary, "Groq (LLaMA 3.3 70B)"
        except Exception as e:
            return f"Groq Error: {str(e)}", "None"

    return "Summary unavailable: Both Gemini and Groq services are unconfigured or reaching rate limits.", "None"

# --- 5. Main UI ---
st.title("⚡ Real-time Tweet Sentiment & Multi-LLM Insights")
st.markdown("Combines a fine-tuned **Logistic Regression** model with **Gemini** & **Groq (LLaMA 3)** for resilient AI processing.")

# Preset Demos
st.markdown("### Quick Preset Demos")
col_p1, col_p2, col_p3 = st.columns(3)

default_text = "Just upgraded my setup with the new processor! Performance improved noticeably."
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = default_text

if col_p1.button("Preset 1 (Positive)"):
    st.session_state.tweet_input = "Just tested the new update on my setup! Render speeds doubled and the fan stays completely silent."
if col_p2.button("Preset 2 (Negative)"):
    st.session_state.tweet_input = "The latest release completely broke my production build. Spent 5 hours debugging with no response from support."
if col_p3.button("Preset 3 (Mixed/Complex)"):
    st.session_state.tweet_input = "The UI design looks very modern, but the new pricing tiers make no sense for freelance developers."

mode = st.radio("Select Input Mode:", ["Single Tweet Analysis", "Batch CSV Processing"], horizontal=True)

if mode == "Single Tweet Analysis":
    user_input = st.text_area("Input Tweet / Text:", value=st.session_state.tweet_input, height=100, key="tweet_input")
    
    if st.button("Analyze Tweet", type="primary"):
        if not user_input.strip():
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Processing local sentiment model & AI summarizer..."):
                start_time = time.time()
                
                # Model inference
                sentiment, confidence = predict_sentiment(user_input)
                
                # Multi-LLM Call
                summary, provider = generate_summary(user_input)
                
                latency = round(time.time() - start_time, 2)

            st.divider()
            
            # Output Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                color = "green" if sentiment == "POSITIVE" else "red"
                st.markdown("**Predicted Sentiment:**")
                st.subheader(f":{color}[{sentiment}]")
            with m2:
                st.markdown("**Model Confidence:**")
                st.subheader(f"{confidence * 100:.1f}%")
                st.progress(confidence)
            with m3:
                st.markdown("**Total Latency:**")
                st.subheader(f"{latency}s")

            # AI Summary Output
            st.markdown("---")
            st.markdown(f"### 🤖 AI Summary *(Powered by {provider})*")
            st.info(summary)

elif mode == "Batch CSV Processing":
    uploaded_file = st.file_uploader("Upload CSV (must contain 'text' or 'tweet' column):", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = "text" if "text" in df.columns else ("tweet" if "tweet" in df.columns else None)
        
        if text_col:
            st.write(f"Loaded {len(df)} rows. Preview:")
            st.dataframe(df.head(3))
            
            if st.button("Run Batch Sentiment Classification"):
                with st.spinner("Classifying dataset..."):
                    results = [predict_sentiment(str(t)) for t in df[text_col]]
                    df["Predicted_Sentiment"] = [r[0] for r in results]
                    df["Confidence"] = [round(r[1] * 100, 1) for r in results]
                
                st.success("Batch classification complete!")
                st.dataframe(df.head(10))
                
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Classified CSV", data=csv_data, file_name="classified_tweets.csv", mime="text/csv")
        else:
            st.error("CSV file must contain a 'text' or 'tweet' column.")