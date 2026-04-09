# Emotion Detection from Text using NLP

A comprehensive NLP project that detects emotions from text using multiple ML approaches — from a traditional TF-IDF baseline to a fine-tuned DistilBERT transformer achieving **92.55% accuracy**. Includes a production-ready Streamlit web app.

## 📁 Project Structure

```
Emotion-Detection/
├── .python-version              # Python 3.11
├── requirements.txt             # Pinned dependencies
├── README.md
│
├── data/
│   └── download.py              # Download dair-ai/emotion dataset
│
├── training/
│   ├── explore_data.py          # Dataset exploration + visualizations
│   ├── train_baseline.py        # TF-IDF + Logistic Regression
│   ├── train_lstm.py            # LSTM (TensorFlow/Keras)
│   └── train_distilbert.py      # DistilBERT fine-tuning (best model)
│
├── notebooks/                   # Original Jupyter notebooks
│   ├── model_trials/            # Exploration & model experiments
│   │   ├── data_exploration.ipynb
│   │   ├── baseline_model.ipynb
│   │   ├── lstm.ipynb
│   │   ├── distilbert.ipynb
│   │   └── kaggle_distilbert.ipynb
│   └── distilbert_finetune/
│       └── nlp-emotion-classification-distilbert.ipynb
│
├── models/
│   └── emotion_model/           # Trained model files (after training)
│
└── app/
    └── streamlit_app.py         # Streamlit web application
```

## 📊 Dataset

**[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** from HuggingFace:
- **20,000** English text samples across **6 emotion classes**
- Splits: 16,000 train / 2,000 validation / 2,000 test
- Classes: sadness, joy, love, anger, fear, surprise

## 🤖 Models

| Model | Script | Accuracy |
|-------|--------|----------|
| TF-IDF + Logistic Regression | `training/train_baseline.py` | 89.1% |
| LSTM | `training/train_lstm.py` | ~35% |
| **DistilBERT (fine-tuned)** | `training/train_distilbert.py` | **92.55%** |

## 🚀 Setup

### Prerequisites
- Python 3.11
- pip

### Install

```bash
git clone <repository-url>
cd Emotion-Detection

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

> For GPU training, install the CUDA version of PyTorch from [pytorch.org](https://pytorch.org/)

## 💻 Usage

### 1. Download Dataset (optional — scripts auto-download)

```bash
python data/download.py
python data/download.py --csv    # also export as CSV
```

### 2. Train Models

```bash
# Explore the dataset
python training/explore_data.py

# Baseline (fast, ~89% accuracy)
python training/train_baseline.py

# LSTM
python training/train_lstm.py --epochs 10

# DistilBERT (best, ~93% accuracy)
python training/train_distilbert.py
python training/train_distilbert.py --epochs 5 --batch-size 32 --lr 3e-5
```

### 3. Train on Kaggle (recommended for GPU)

1. Upload `notebooks/kaggle_distilbert.ipynb` to [Kaggle](https://www.kaggle.com/code)
2. Enable **GPU T4** and **Internet** in Settings
3. Run All — takes ~10-15 min
4. Download `emotion_model.zip` from the Output panel
5. Extract to `models/emotion_model/`

### 4. Run the Web App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, enter text, and detect emotions.

## 📈 Results

### Confusion Matrix

<img width="554" height="468" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/14cb131e-c3a7-42ad-916e-e4effbf7448a" />

### Application

<img width="964" height="688" alt="Streamlit Application Interface" src="https://github.com/user-attachments/assets/1a1c3bd9-9a6d-45a2-86b4-c96ff10c06af" />

<img width="964" height="703" alt="Emotion Detection Example 1" src="https://github.com/user-attachments/assets/7657db4c-684e-45a9-bbfa-0e26e6567249" />

<img width="964" height="731" alt="Emotion Detection Example 2" src="https://github.com/user-attachments/assets/a66def35-8234-48c3-802c-00e96a06c867" />

## 🛠 Tech Stack

- **PyTorch** + **Transformers** — DistilBERT fine-tuning
- **TensorFlow/Keras** — LSTM model
- **scikit-learn** — Baseline ML + metrics
- **Streamlit** — Web application
- **HuggingFace Datasets** — Data loading
- **Matplotlib/Seaborn** — Visualization

## 📝 Notes

- DistilBERT training on CPU takes several hours — use Kaggle (free T4 GPU) or a local GPU
- The model works best on English text
- Model files (`models/emotion_model/`) are `.gitignore`d due to size (~260MB)

## 📄 License

Open source for educational and research purposes.

---

**Built with Python and modern NLP techniques**
