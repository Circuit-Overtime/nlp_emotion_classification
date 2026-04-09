# Emotion Detection from Text using NLP

A multi-approach NLP system that classifies English text into six emotion categories — **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise** — using progressively advanced models, culminating in a fine-tuned DistilBERT transformer achieving **93.75% test accuracy**.

## Dataset

**[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** — 20,000 labeled English text samples from HuggingFace.

| Split | Samples |
|-------|---------|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

The dataset is imbalanced: **joy** (33.5%) and **sadness** (29.2%) dominate, while **surprise** (3.6%) and **love** (8.2%) are underrepresented. Despite this, the transformer model generalizes well across all classes.

## Models & Results

### 1. TF-IDF + Logistic Regression (Baseline) — 89.1%

Traditional pipeline using TF-IDF vectorization (10K features, 1-2 gram range, English stopwords removed) feeding a Logistic Regression classifier. Strong baseline that outperforms the deep learning approach below, demonstrating that classical methods remain competitive on small datasets.

### 2. LSTM — ~35%

Custom architecture: Embedding(20K vocab, 128d) → LSTM(128) → Dropout(0.5) → Dense(6, softmax). Trained with Adam optimizer and sparse categorical crossentropy for 10 epochs. The model failed to converge — stuck predicting the majority class — likely due to insufficient embedding quality and no pre-trained weights. Demonstrates the gap between training from scratch vs. leveraging pre-trained representations.

### 3. DistilBERT (Fine-tuned) — 93.75%

Fine-tuned `distilbert-base-uncased` (66.9M parameters) using HuggingFace Trainer. DistilBERT retains 97% of BERT's language understanding at 60% the size and 2x inference speed.

**Training progress:**

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|--------------|-----------------|----------|
| 1 | 0.6862 | 0.5305 | 91.65% |
| 2 | 0.3889 | 0.3299 | 93.65% |
| 3 | 0.2855 | 0.3163 | **93.75%** |

**Training configuration:**
- Learning rate: 2e-5 with linear warmup
- Batch size: 32 (Kaggle T4 x2)
- Epochs: 3 (sweet spot — validation loss plateaus at epoch 3)
- Weight decay: 0.01
- FP16 mixed precision on GPU
- Total training time: ~10-15 min (T4 GPU)

**Per-class performance (test set):**

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Sadness | 0.95 | 0.96 | 0.95 | 581 |
| Joy | 0.97 | 0.94 | 0.95 | 695 |
| Love | 0.79 | 0.89 | 0.84 | 159 |
| Anger | 0.92 | 0.92 | 0.92 | 275 |
| Fear | 0.89 | 0.88 | 0.88 | 224 |
| Surprise | 0.78 | 0.79 | 0.78 | 66 |

The model is most confident on **sadness** and **joy** (highest support), and struggles most with **surprise** and **love** (lowest support + semantic overlap with joy).

## Key Takeaways

- **Pre-training is everything**: DistilBERT (pre-trained on massive text corpora) crushes the from-scratch LSTM by 57+ percentage points. On small datasets, transfer learning >> training from scratch.
- **Classical ML holds up**: TF-IDF + LogReg achieves 89% with seconds of training — only 3.5% behind the transformer. For latency-sensitive or resource-constrained deployments, it's a viable option.
- **3 epochs is optimal**: Fine-tuning BERT-family models on small datasets converges fast. Beyond 3 epochs, validation loss increases — the model memorizes training data.

## Confusion Matrix

<img width="554" height="468" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/14cb131e-c3a7-42ad-916e-e4effbf7448a" />

## Application

A Streamlit web app loads the fine-tuned DistilBERT model and provides real-time emotion detection via a text input interface.

<img width="964" height="688" alt="Streamlit Application Interface" src="https://github.com/user-attachments/assets/1a1c3bd9-9a6d-45a2-86b4-c96ff10c06af" />

<img width="964" height="703" alt="Emotion Detection Example 1" src="https://github.com/user-attachments/assets/7657db4c-684e-45a9-bbfa-0e26e6567249" />

<img width="964" height="731" alt="Emotion Detection Example 2" src="https://github.com/user-attachments/assets/a66def35-8234-48c3-802c-00e96a06c867" />

## Tech Stack

| Component | Technology |
|-----------|------------|
| Transformer fine-tuning | PyTorch, HuggingFace Transformers |
| Sequence model | TensorFlow / Keras |
| Classical ML | scikit-learn |
| Data | HuggingFace Datasets |
| Visualization | Matplotlib, Seaborn |
| Web app | Streamlit |
| Python | 3.11 |

## License

Open source for educational and research purposes.
