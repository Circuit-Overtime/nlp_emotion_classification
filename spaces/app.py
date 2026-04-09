"""
Emotion Detection - Hugging Face Spaces (CPU)

Gradio app for free deployment on HF Spaces.
DistilBERT is small enough (~268MB) for fast CPU inference.
"""

import gradio as gr
from transformers import pipeline

EMOJI_MAP = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
}

LABEL_MAP = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise",
}

classifier = pipeline(
    "text-classification",
    model="./models/emotion_model",
    tokenizer="./models/emotion_model",
)


def predict(text: str) -> str:
    if not text.strip():
        return "Please enter some text."

    result = classifier(text)[0]

    raw_label = result["label"]
    emotion = LABEL_MAP.get(raw_label, raw_label)
    confidence = result["score"]
    emoji = EMOJI_MAP.get(emotion, "")

    return f"{emoji} **{emotion.upper()}**\n\nConfidence: {confidence:.4f}"


examples = [
    ["I am so happy today, everything is going great!"],
    ["I feel terrible and nothing seems to work out."],
    ["This is absolutely terrifying, I can't stop shaking."],
    ["I can't believe you would do something like that to me!"],
    ["You are the most wonderful person I have ever met."],
    ["Wow, I never expected that to happen!"],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter text",
        placeholder="Type a sentence to detect its emotion...",
        lines=3,
    ),
    outputs=gr.Markdown(label="Prediction"),
    title="Emotion Detection from Text",
    description=(
        "Detects emotions in English text using a fine-tuned **DistilBERT** model. "
        "Classifies into 6 categories: sadness, joy, love, anger, fear, surprise. "
        "Achieves **93.75% accuracy** on the test set."
    ),
    examples=examples,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
