#  Human Emotion Detection App

A Streamlit-based web application that detects human emotions from text input using a fine-tuned Transformer model.

---

## 🔗 Live Demo
 **Click here:** [Human Emotion Detection App](https://human-emotion-detection-app.streamlit.app/)


##  Project Overview

This application analyzes user-entered text and predicts the underlying human emotion such as:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

It uses a **pre-trained NLP model fine-tuned DistilBERT Transformer with Hugging Face Transformers and PyTorch, generating probabilistic emotion predictions via softmax-based confidence scoring for emotion classification** and provides an interactive UI using **Streamlit**.

---

## Model Details

- Model Format: `safetensors`
- Framework: Hugging Face Transformers
- Tokenizer: Included in the `emotion_model/` directory
- Model Loading: Git LFS (Large File Storage)

---

##  Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- NLTK
- Git & Git LFS

---

##  Project Structure

Human Emotion Detection App/
│

├── app/

│ └── app.py

│
├── emotion_model/

│ ├── config.json

│ ├── model.safetensors

│ ├── tokenizer.json

│ ├── tokenizer_config.json

│ ├── special_tokens_map.json

│ └── vocab.txt

│
├── requirements.txt

├── README.md

└── .gitattributes


---

##  Run Locally

### 1️ Create virtual environment

python -m venv venv
venv\Scripts\activate

---

### 2️ Install dependencies
pip install -r requirements.txt

---

3️ Run Streamlit app
streamlit run app/app.py

---

Author

Eshani Banik
B.Tech CSE

Machine Learning & NLP Enthusiast





