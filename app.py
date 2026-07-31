import re
import pickle
import time
from pathlib import Path
import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# Page Configuration
st.set_page_config(
    page_title="Tweet Sentiment & AI Insights",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_ml_assets():
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "sentiment_model.pkl"
    vec_path = project_dir / "models" / "vectorizer.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

try:
    model, vectorizer = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure model files exist in 'models/' directory.")
    st.stop()

@st.cache_resource
def init_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.sidebar.warning("⚠️ Gemini API Key not found in secrets.toml.")
        return None

client = init_gemini_client()

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

def summarize_with_gemini(text: str) -> str:
    if not client:
        return "Gemini API key not configured."
    try:
        prompt = f"Provide a brief 1-2 sentence summary/explanation of what this tweet is about: \"{text}\""
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        
        # Safe access check before calling .strip()
        if response and response.text:
            return response.text.strip()
        else:
            return "No text summary returned (the prompt may have triggered safety filters)."
            
    except APIError as e:
        return f"Gemini API Error: {str(e)}"
    except Exception as e:
        return f"Summarization unavailable ({str(e)})."

st.title("⚡ Real-time Tweet Sentiment Analyzer & AI Summarizer")
st.markdown("Combines a fine-tuned **Logistic Regression** model with **Gemini 2.5 Flash** for automated tweet insights.")


def set_preset(text: str) -> None:
    """Put a selected demo into the text-area widget before its next render."""
    st.session_state["tweet_text"] = text


st.session_state.setdefault("tweet_text", "")

st.markdown("### Quick Preset Demos")
col_p1, col_p2, col_p3 = st.columns(3)
col_p1.button(
    "Preset 1 (Positive)",
    on_click=set_preset,
    args=("Just tested the new firmware update on my setup! Battery life improved and it runs super smooth.",),
)
col_p2.button(
    "Preset 2 (Negative)",
    on_click=set_preset,
    args=("The latest release completely broke my database migration. Spent 4 hours debugging with zero help.",),
)
col_p3.button(
    "Preset 3 (Mixed/Complex)",
    on_click=set_preset,
    args=("The UI looks crisp and clean, but the subscription prices make no sense for students.",),
)

mode = st.radio("Select Input Mode:", ["Single Tweet Analysis", "Batch CSV Processing"], horizontal=True)

if mode == "Single Tweet Analysis":
    user_input = st.text_area(
        "Input Tweet / Text:",
        key="tweet_text",
        placeholder="Type or paste text here...",
        height=100,
    )
    
    if st.button("Analyze Tweet", type="primary"):
        if not user_input.strip():
            st.warning("Please enter text or select a preset.")
        else:
            with st.spinner("Classifying sentiment and generating AI summary..."):
                start_time = time.time()
                
                sentiment, confidence = predict_sentiment(user_input)
                
                summary = summarize_with_gemini(user_input)
                
                latency = round(time.time() - start_time, 2)

            st.divider()

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
                st.markdown("**Inference Latency:**")
                st.subheader(f"{latency}s")

            st.markdown("---")
            st.markdown("### 🤖 Gemini Executive Summary")
            st.info(summary)

elif mode == "Batch CSV Processing":
    uploaded_file = st.file_uploader("Upload CSV (must contain a 'text' or 'tweet' column):", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = "text" if "text" in df.columns else ("tweet" if "tweet" in df.columns else None)
        
        if text_col:
            st.write(f"Loaded {len(df)} rows. Preview:")
            st.dataframe(df.head(3))
            
            if st.button("Run Batch Analysis"):
                with st.spinner("Classifying dataset..."):
                    results = [predict_sentiment(str(t)) for t in df[text_col]]
                    df["Predicted_Sentiment"] = [r[0] for r in results]
                    df["Confidence"] = [round(r[1] * 100, 1) for r in results]
                
                st.success("Batch classification complete!")
                st.dataframe(df.head(10))

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Processed CSV", data=csv_data, file_name="classified_tweets.csv", mime="text/csv")
        else:
            st.error("CSV file must contain a 'text' or 'tweet' column.")
