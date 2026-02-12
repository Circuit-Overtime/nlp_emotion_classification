# Emotion Detection from Text using NLP

A comprehensive Natural Language Processing project that detects emotions from text using multiple machine learning approaches. This project implements and compares three different models: a traditional TF-IDF + Logistic Regression baseline, a deep learning LSTM model, and a state-of-the-art fine-tuned DistilBERT transformer model.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project demonstrates emotion classification from textual data using various NLP techniques. The system can classify text into six distinct emotion categories: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**. The project includes both model training notebooks and a production-ready Streamlit web application for real-time emotion detection.

## ✨ Features

- **Multiple Model Implementations**: Compare traditional ML, deep learning, and transformer-based approaches
- **Interactive Web Application**: Real-time emotion detection through an intuitive Streamlit interface
- **Comprehensive Evaluation**: Detailed performance metrics and confusion matrix analysis
- **Production Ready**: Deployed model with easy-to-use API interface

## 📊 Dataset

The project uses the **dair-ai/emotion** dataset from Hugging Face, which contains:
- **Training set**: 16,000 samples
- **Validation set**: 2,000 samples
- **Test set**: 2,000 samples
- **6 emotion classes**: sadness, joy, love, anger, fear, surprise

The dataset consists of English text samples labeled with their corresponding emotional states, making it ideal for supervised learning approaches.

## 🤖 Models

### 1. TF-IDF + Logistic Regression (Baseline)
A traditional machine learning approach using:
- **TF-IDF Vectorization**: Converts text to numerical features using term frequency-inverse document frequency
- **Logistic Regression**: Linear classifier for multi-class emotion classification
- **Features**: 10,000 max features with 1-2 gram ranges

### 2. LSTM (Long Short-Term Memory)
A deep learning model built from scratch using TensorFlow/Keras:
- **Embedding Layer**: Converts words to dense vector representations
- **LSTM Layers**: Captures sequential patterns and long-term dependencies in text
- **Dropout**: Regularization to prevent overfitting
- **Architecture**: Custom-built neural network optimized for emotion classification

### 3. DistilBERT (Fine-tuned)
A state-of-the-art transformer model:
- **Base Model**: DistilBERT (distilbert-base-uncased) - a lighter, faster version of BERT
- **Fine-tuning**: Customized on the emotion dataset for optimal performance
- **Best Performance**: Achieves 92.55% accuracy on the test set
- **Deployment**: Used in the production Streamlit application

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Emotion-Detection
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt includes PyTorch with CPU support. If you have a CUDA-compatible GPU and want to use GPU acceleration, you may need to install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/).

## 💻 Usage

### Running the Streamlit Web Application

1. **Ensure the trained model is available**: The application expects a fine-tuned DistilBERT model in the `./emotion_model` directory. If you haven't trained the model yet, you'll need to:
   - Run the `distilbert.ipynb` notebook to train the model
   - Save the model to the `./emotion_model` directory

2. **Launch the Streamlit app**:

```bash
streamlit run app.py
```

3. **Access the application**: Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

4. **Use the application**:
   - Enter or paste text in the text area
   - Click the "Detect Emotion" button
   - View the predicted emotion and confidence score

### Training Models

The project includes Jupyter notebooks for training each model:

- **`data_exploration.ipynb`**: Explore and understand the dataset
- **`baseline_model.ipynb`**: Train the TF-IDF + Logistic Regression baseline model
- **`lstm.ipynb`**: Train the LSTM model from scratch
- **`distilbert.ipynb`**: Fine-tune the DistilBERT model (produces the best results)

To train models:
1. Open the desired notebook in Jupyter Lab or Jupyter Notebook
2. Run all cells sequentially
3. The notebooks will download the dataset, preprocess the data, train the model, and evaluate performance

## 📁 Project Structure

```
Emotion-Detection/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data_exploration.ipynb      # Dataset exploration and analysis
├── baseline_model.ipynb        # TF-IDF + Logistic Regression model
├── lstm.ipynb                  # LSTM model implementation
├── distilbert.ipynb            # DistilBERT fine-tuning
│
└── emotion_model/              # Trained DistilBERT model (created after training)
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Description |
|-------|----------|-------------|
| **DistilBERT** | **92.55%** | Fine-tuned transformer model (Best) |
| LSTM | ~85-90% | Deep learning model from scratch |
| TF-IDF + Logistic Regression | ~80-85% | Traditional ML baseline |

### Confusion Matrix

The confusion matrix below shows the performance of the DistilBERT model across all six emotion classes:

<img width="554" height="468" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/14cb131e-c3a7-42ad-916e-e4effbf7448a" />

### Application Screenshots

<img width="964" height="688" alt="Streamlit Application Interface" src="https://github.com/user-attachments/assets/1a1c3bd9-9a6d-45a2-86b4-c96ff10c06af" />

<img width="964" height="703" alt="Emotion Detection Example 1" src="https://github.com/user-attachments/assets/7657db4c-684e-45a9-bbfa-0e26e6567249" />

<img width="964" height="731" alt="Emotion Detection Example 2" src="https://github.com/user-attachments/assets/a66def35-8234-48c3-802c-00e96a06c867" />

## 🛠 Technologies Used

- **Python**: Core programming language
- **PyTorch**: Deep learning framework for transformer models
- **Transformers (Hugging Face)**: Pre-trained models and training utilities
- **TensorFlow/Keras**: LSTM model implementation
- **scikit-learn**: Traditional ML models and evaluation metrics
- **Streamlit**: Web application framework
- **datasets (Hugging Face)**: Dataset loading and management
- **Jupyter Notebooks**: Interactive development and experimentation

## 📝 Notes

- The DistilBERT model requires significant computational resources for training. Training on CPU may take several hours.
- For production use, consider deploying the model using cloud services or containerization for better scalability.
- The model performs best on English text and may have reduced accuracy on other languages or heavily informal text.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is open source and available for educational and research purposes.

---

**Built with ❤️ using Python and modern NLP techniques**
