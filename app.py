Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Pricing


Spaces:
gousemohiddin
/
stock-news-sentiment-app


like
0

App
Files
Community
Settings
stock-news-sentiment-app
/
app.py

gousemohiddin's picture
gousemohiddin
Update app.py
ed98984
verified
32 minutes ago
raw

Copy download link
history
blame
edit
delete

8.03 kB
# 1. CRITICAL: yfinance first to prevent Chrome Impersonation errors
import yfinance as yf
from newspaper import Article
import gradio as gr
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import time

# ---------------------------------------------------------
# 1. LOAD MODELS AND ASSETS
# ---------------------------------------------------------
print("🚀 Initializing Application...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
model = tf.keras.models.load_model('st_nn_v1_champion.keras')
le = joblib.load('label_encoder.pkl')

# ---------------------------------------------------------
# 2. CORE LOGIC FUNCTIONS
# ---------------------------------------------------------

# This function updates the UI labels so the user isn't confused
def update_context_label(input_mode):
    if input_mode == "Live Ticker (e.g., NVDA)":
        return "🏢 **Entity Sentiment:** Analyzing company-specific news and performance."
    elif input_mode == "Article URL":  # FIXED: Ensures no red error badge pops up
        return "🌍 **Macro Sentiment:** Analyzing global market news and broad economic trends."
    else:
        return "✍️ **Manual Sentiment:** Analyzing the specific text provided."

def analyze_news(user_input, input_type):
    try:
        if input_type == "Live Ticker (e.g., NVDA)":
            ticker_symbol = user_input.strip().upper()
            ticker = yf.Ticker(ticker_symbol)
            try:
                news_list = ticker.news
            except Exception:
                time.sleep(1)
                news_list = ticker.news
            
            if not news_list:
                return "No recent news found.", "⚠️ N/A", "0.00%", "No news available"
            
            headline = f"Recent update for {ticker_symbol}"
            for item in news_list:
                if 'title' in item:
                    headline = item['title']
                    break
                elif 'content' in item and 'title' in item['content']:
                    headline = item['content']['title']
                    break
            
        elif input_type == "Article URL":
            article = Article(user_input.strip())
            article.download()
            article.parse()
            headline = article.title
            
        else: # Manual Headline
            headline = user_input.strip()

        # Inference
        embedding = encoder.encode([headline])
        predictions = model.predict(embedding, verbose=0)
        class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions) * 100
        sentiment_val = le.inverse_transform([class_index])[0]
        
        # Sentiment Display
        if sentiment_val == 1:
            sentiment_display = "🟢 POSITIVE (BULLISH)"
        elif sentiment_val == -1:
            sentiment_display = "🔴 NEGATIVE (BEARISH)"
        else:
            sentiment_display = "⚪ NEUTRAL"

        # Certainty Calibration
        if confidence_score > 85:
            certainty = "🔥 High Conviction (Strong Signal)"
        elif confidence_score > 65:
            certainty = "⚖️ Moderate Conviction"
        else:
            certainty = "⚠️ Low Conviction (Mixed/Conflicting News)"

        return headline, sentiment_display, f"{confidence_score:.2f}%", certainty
        
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
            return "⚠️ Yahoo Rate Limit. Use 'Manual Headline' for now.", "❌ Limit", "0.00%", "N/A"
        return f"Error: {error_msg}", "❌ Error", "0.00%", "N/A"

# ---------------------------------------------------------
# 3. GRADIO UI
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 📊 AI Stock Market Sentiment Engine")
    gr.Markdown("### *UT Austin Capstone Project: Generative AI for Business Applications*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛠 Input Analysis")
            input_mode = gr.Radio(
                choices=["Live Ticker (e.g., NVDA)", "Article URL", "Manual Headline"], 
                value="Live Ticker (e.g., NVDA)", 
                label="Choose Data Source",
                interactive=True
            )
            # DYNAMIC LABEL
            sentiment_context = gr.Markdown("🏢 **Entity Sentiment:** Analyzing company-specific news.")
            
            input_val = gr.Textbox(label="Enter Data", placeholder="Ticker, URL, or Headline...", interactive=True)
            btn = gr.Button("RUN AI INFERENCE", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 AI Prediction (Short-Term Outlook)")
            out_headline = gr.Textbox(label="Analyzed Headline", interactive=False)
            with gr.Row():
                out_sentiment = gr.Label(label="Sentiment Result")
                out_conf = gr.Textbox(label="Model Confidence", interactive=False)
            out_tier = gr.Textbox(label="Certainty Level", interactive=False)

    # Note on Time Horizon moved directly below the main UI boxes
    gr.Markdown(
        "> **Note on Time Horizon:** This prediction reflects current market mood based on the "
        "most recent headline. It is designed for **short-term** sentiment shifts rather than "
        "long-term fundamental value."
    )

    # ---------------------------------------------------------
    # 4. EVENT LISTENERS
    # ---------------------------------------------------------
    
    input_mode.change(fn=update_context_label, inputs=input_mode, outputs=sentiment_context)
    btn.click(fn=analyze_news, inputs=[input_val, input_mode], outputs=[out_headline, out_sentiment, out_conf, out_tier])
    input_val.submit(fn=analyze_news, inputs=[input_val, input_mode], outputs=[out_headline, out_sentiment, out_conf, out_tier])

    gr.Examples(
        examples=[
            ["NVDA", "Live Ticker (e.g., NVDA)"], 
            ["SNDK", "Live Ticker (e.g., NVDA)"],
            ["https://finance.yahoo.com/news/stock-market-today-sp-500-nasdaq-093554129.html", "Article URL"],
            ["Federal Reserve announces unexpected interest rate cuts.", "Manual Headline"]
        ],
        inputs=[input_val, input_mode]
    )

    # ---------------------------------------------------------
    # 5. TECHNICAL FOOTER (Moved to the very bottom)
    # ---------------------------------------------------------
    gr.Markdown("---")
    gr.Markdown("### 🧠 Technical Architecture & Project Details")
    gr.Markdown(
        "**Which Model is Used?**\n"
        "This application uses a custom **Deep Neural Network (64/32 Layers)** built with Keras/TensorFlow. "
        "Before the text enters the network, it is processed by a **Sentence-Transformer (all-MiniLM-L6-v2)**, "
        "which converts the human language into a 384-dimensional mathematical vector.\n\n"
        "**How does it work?**\n"
        "1. **Data Ingestion:** The app scrapes near-real-time headlines using `yfinance` or `newspaper3k`.\n"
        "2. **Inference:** The Neural Network analyzes the vector and uses a Softmax activation layer to output a probability distribution across three classes: Bearish (-1), Neutral (0), and Bullish (1).\n"
        "3. **Calibration:** The app reports the AI's confidence level, flagging ambiguous news as 'Low Conviction' to protect the user from over-relying on weak signals.\n\n"
        "**Why this architecture?**\n"
        "Compared to traditional Word2Vec or standard Random Forest models, the combination of Sentence Transformers "
        "and a multi-layer Neural Network captures the deep contextual nuances of financial jargon. "
        "The champion model achieved a **72.6% accuracy** and an **F1-Score of 0.71** on unseen testing data."
    )

if __name__ == "__main__":
    demo.launch()
