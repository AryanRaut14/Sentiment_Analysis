# ⚡ Real-Time Tweet Sentiment Analyzer & Multi-LLM Summarizer

An end-to-end, production-grade NLP application combining an optimized **Calibrated Linear Support Vector Classifier (LinearSVC)** with a high-availability **Dual-LLM Architecture (Google Gemini + Groq LLaMA 3)** to perform real-time sentiment analysis and generate contextual AI tweet summaries.

---

## 📌 Project Overview

Understanding public sentiment on social media requires both quantitative scoring (positive vs. negative classification) and qualitative understanding (what the tweet is actually discussing). This project bridges classic Machine Learning with modern Generative AI:

1. **Custom ML Sentiment Engine:** Classifies tweets into `POSITIVE` or `NEGATIVE` sentiment with calibrated probability confidence scores using a LinearSVC model trained on 1.6M+ Kaggle tweets.
2. **Dual-LLM Resilient Architecture:** Uses **Google Gemini 2.0 Flash** as the primary summarizer and automatically falls back to **Meta LLaMA 3.3 70B via Groq** if API rate limits (`429`) or traffic spikes occur.
3. **Production Web Dashboard:** Built with **Streamlit**, supporting single-tweet testing, curated preset dropdowns for Positive/Negative/Mixed examples, latency tracking, and AI summaries.

---

## 📈 Model Performance & Comparative Benchmark

To address false positives on complex, negated, or mixed-sentiment tweets (e.g., *"spent 5 hours debugging"* or *"pricing tiers make no sense"*), the classification engine was upgraded from standard Logistic Regression to a **Calibrated Linear Support Vector Classifier (LinearSVC)** paired with **Negation-Aware Preprocessing** and **Trigram TF-IDF Vectorization**.

### Evaluation Results (Tested on 320,000 Unseen Tweets)

| Metric | Logistic Regression (Baseline) | Calibrated LinearSVC (Upgraded) |
| :--- | :---: | :---: |
| **Accuracy** | ~75.2% | **78.16%** |
| **Negative Precision / Recall** | 0.74 / 0.72 | **0.79 / 0.76** |
| **Positive Precision / Recall** | 0.75 / 0.77 | **0.77 / 0.80** |
| **Macro F1-Score** | 0.75 | **0.78** |
| **Inference Latency** | < 2 ms | **< 2 ms** |

### Detailed Classification Report (LinearSVC)

```text
              precision    recall  f1-score   support

    Negative       0.79      0.76      0.78    160000
    Positive       0.77      0.80      0.79    160000

    accuracy                           0.78    320000
   macro avg       0.78      0.78      0.78    320000
weighted avg       0.78      0.78      0.78    320000