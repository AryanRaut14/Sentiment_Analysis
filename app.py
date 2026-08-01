import os
import re
import pickle
import time
import streamlit as st
from google import genai
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Tweet Sentiment & Dual AI Insights",
    page_icon="⚡",
    layout="wide"
)

# 1. Load ML Artifacts
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

# 2. Initialize API Clients
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

# 3. Preprocessing & Sentiment Inference
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase and strip URLs, mentions, and hashtags
    cleaned = re.sub(r"http\S+|@\S+|#\S+", " ", text.lower())
    
    # 2. Preserve contractions before stripping punctuation
    cleaned = re.sub(r"can't", "can not", cleaned)
    cleaned = re.sub(r"n't", " not", cleaned)
    cleaned = re.sub(r"won't", "will not", cleaned)
    
    # 3. Replace non-alphanumeric characters (except spaces)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    
    words = cleaned.split()
    
    # 4. Prefix words following a negation term with "NOT_"
    negation = False
    processed_words = []
    negation_words = {"not", "no", "never", "neither", "nor", "none"}
    words_since_negation = 0
    
    for word in words:
        if word in negation_words:
            negation = True
            words_since_negation = 0
            processed_words.append(word)
        elif negation:
            processed_words.append(f"NOT_{word}")
            words_since_negation += 1
            # Reset negation after 3 words to avoid over-tagging
            if words_since_negation >= 3:
                negation = False
        else:
            processed_words.append(word)
            
    return " ".join(processed_words)

def predict_sentiment(text: str):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    confidence = float(max(probabilities))
    return sentiment, confidence

# 4. Resilient Multi-Provider Summarizer
def generate_summary(text: str) -> tuple[str, str]:
    """
    Attempts summarization with Gemini first.
    Falls back to Groq (LLaMA 3.3 70B) if rate-limited or unavailable.
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
                continue

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
            content = chat_completion.choices[0].message.content
            summary = str(content).strip() if content else ""
            return summary, "Groq (LLaMA 3.3 70B)"
        except Exception as e:
            return f"Groq Error: {str(e)}", "None"

    return "Summary unavailable: Both Gemini and Groq services are unconfigured or reaching rate limits.", "None"

# 5. Main UI
st.title("⚡ Real-time Tweet Sentiment & Multi-LLM Insights")
st.markdown("Combines a fine-tuned **Logistic Regression** model with **Gemini** & **Groq (LLaMA 3)** for resilient AI processing.")

# Preset Demos
st.markdown("### Quick Preset Demos")
default_text = "Just upgraded my setup with the new processor! Performance improved noticeably."
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = default_text

preset_options = {
    "Positive": [
        "Just tested the new update on my setup! Render speeds doubled and the fan stays completely silent.",
        "The support team resolved my issue within minutes. This service keeps getting better!",
        "Really impressed with the new feature set—simple to use, fast, and exactly what I needed.",
    ],
    "Negative": [
        "The latest release completely broke my production build. Spent 5 hours debugging with no response from support.",
        "After the update, the app crashes every time I try to save my work. Very frustrating.",
        "The product arrived late and damaged, and customer service has not replied to my messages.",
    ],
    "Mixed / Complex": [
        "The UI design looks very modern, but the new pricing tiers make no sense for freelance developers.",
        "Performance is noticeably better, although setting everything up was far more difficult than expected.",
        "I love the new capabilities, but the frequent notifications are becoming distracting.",
    ],
}

def load_preset(preset_key: str) -> None:
    selected_text = st.session_state[preset_key]
    if selected_text:
        st.session_state["tweet_input"] = selected_text

col_p1, col_p2, col_p3 = st.columns(3)
for column, (sentiment_type, options) in zip((col_p1, col_p2, col_p3), preset_options.items()):
    preset_key = f"preset_{sentiment_type.lower().replace(' / ', '_')}"
    with column:
        st.selectbox(
            f"{sentiment_type} presets",
            options=[""] + options,
            format_func=lambda text: "Choose an example..." if not text else text,
            key=preset_key,
            on_change=load_preset,
            args=(preset_key,),
        )

user_input = st.text_area("Input Tweet / Text:", key="tweet_input", height=100)

if st.button("Analyze Tweet", type="primary"):
    if not user_input.strip():
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Processing local sentiment model & AI summarizer..."):
            start_time = time.time()

            sentiment, confidence = predict_sentiment(user_input)
            summary, provider = generate_summary(user_input)
            
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
            st.markdown("**Total Latency:**")
            st.subheader(f"{latency}s")

        st.markdown("---")
        st.markdown(f"### 🤖 AI Summary *(Powered by {provider})*")
        st.info(summary)