---
title: AI Stock Market Sentiment Engine
emoji: 📊
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: true
license: apache-2.0
---

# 📊 AI Stock Market Sentiment Engine

This application uses a custom **Deep Neural Network** to analyze real-time financial news sentiment. 

### 🧠 Technical Architecture
- **Model:** 64/32 Layer Deep Neural Network (Keras/TensorFlow).
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2) producing 384-dim vectors.
- **Data Ingestion:** Live scraping via `yfinance` and `newspaper3k`.
- **Accuracy:** 72.6% on unseen testing data.



### 🚀 Live Demo
You can try the live application hosted on Hugging Face here: 
https://huggingface.co/spaces/gousemohiddin/stock-news-sentiment-app

### 🛠 Installation
To run this locally:
1. Clone the repo: `git clone https://github.com/gousemohiddin/stock-news-sentiment-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

---
