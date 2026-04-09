"""
Streamlit web application for real-time emotion detection.

Loads the fine-tuned DistilBERT model from models/emotion_model/
and provides an interactive UI for text classification.

Usage:
    streamlit run app/streamlit_app.py
"""

import os

import streamlit as st
import torch
from transformers import pipeline

# Resolve model path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "emotion_model")

st.title("Emotion Detection from Text")

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_DIR):
        st.error(
            f"Model not found at `{MODEL_DIR}`.\n\n"
            "Train the model first:\n"
            "```\npython training/train_distilbert.py\n```\n"
            "Or download from Kaggle and extract to `models/emotion_model/`."
        )
        st.stop()
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        device=device,
    )

classifier = load_classifier()

text = st.text_area("Enter text")

label_map = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise",
}

if st.button("Detect Emotion"):
    if text:
        result = classifier(text)[0]
        raw_label = result["label"]
        emotion = label_map.get(raw_label, raw_label)
        confidence = result["score"]

        st.success(f"Emotion: **{emotion.upper()}**")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text first!")
