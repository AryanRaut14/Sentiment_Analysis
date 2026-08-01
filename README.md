# ⚡ Real-Time Tweet Sentiment Analyzer & Multi-LLM Summarizer

An end-to-end, production-grade NLP application combining a fine-tuned **Logistic Regression classification model** with a high-availability **Dual LLM Architecture (Google Gemini + Groq LLaMA 3)** to perform real-time sentiment analysis and generate contextual AI tweet summaries.

---

## 📌 Project Overview

Understanding public sentiment on social media requires both quantitative scoring (positive vs. negative classification) and qualitative understanding (what the tweet is actually discussing). This project bridges classic Machine Learning with modern Generative AI:

1. **Custom ML Sentiment Engine:** Classifies tweets into `POSITIVE` or `NEGATIVE` sentiment along with probability confidence scores using a Logistic Regression model trained on 1.6M+ Kaggle tweets.
2. **Dual-LLM Resilient Architecture:** Uses **Google Gemini 2.0 Flash** as the primary summarizer and automatically falls back to **Meta LLaMA 3.3 70B via Groq** if API rate limits (`429`) or traffic spikes occur.
3. **Production Web Dashboard:** Built with **Streamlit**, supporting single-tweet testing, three preset dropdowns with example texts, latency metrics, and AI-generated summaries.

---

## 🚀 Key Features

- **Text Preprocessing Pipeline:** Strips noise (URLs, `@mentions`, hashtags, punctuation), applies lowercasing normalization, stopword removal, and Porter Stemming via `NLTK`.
- **TF-IDF Feature Extraction:** Transforms raw tweet text into high-dimensional unigram/bigram numerical feature vectors.
- **Fast Local Inference:** Loads serialized binary model artifacts (`.pkl`) in milliseconds for high-throughput prediction.
- **High-Availability AI Failover:** Zero-downtime integration combining Gemini 2.0 Flash and Groq LLaMA 3.3 70B to eliminate rate-limit bottlenecks during live demos.
- **Preset Examples:** Choose from three positive, three negative, or three mixed/complex examples to quickly populate the analysis input.

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 3.10+
- **Machine Learning & NLP:** Scikit-Learn, NLTK, Pandas, NumPy
- **Generative AI SDKs:** `google-genai` (Gemini API), `groq` (Groq LLaMA API)
- **Web Dashboard:** Streamlit
- **Serialization:** Pickle / Joblib
- **Environment & Tools:** VS Code, Git

---

## 📂 Project Directory Structure

```text
sentiment-gemini-app/
├── .streamlit/
│   └── secrets.toml          # API keys for Gemini & Groq (git-ignored)
├── data/
│   └── tweets.csv            # Training dataset (Kaggle Sentiment140)
├── models/
│   ├── sentiment_model.pkl   # Serialized Logistic Regression model
│   └── vectorizer.pkl        # Serialized TF-IDF vectorizer
├── src/
│   └── sentiment_training.ipynb  # Model training & evaluation notebook
├── .gitignore                # Git exclusion rules
├── app.py                    # Streamlit web application
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
