# 🧠 Human Emotion Detection App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://human-emotion-detection-app-link-here.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)
A high-performance web application that utilizes Deep Learning and Natural Language Processing (NLP) to analyze raw text and instantly predict the underlying human emotion.

🔗 **[Live Demo: Try the App Here!](https://human-emotion-detection-app.streamlit.app/)** 
---
## 🎯 Project Overview
This system processes user-entered text and performs multi-class sentiment analysis to dynamically categorize the psychological nuances of the sentence into one of six distinct emotional states:
- 😄 **Joy**
- 😢 **Sadness**
- 😠 **Anger**
- 😨 **Fear**
- ❤️ **Love**
- 😲 **Surprise**
By leveraging a fine-tuned **DistilBERT** Transformer architecture, the application outputs probabilistic emotion predictions via softmax-based confidence scoring, exposing the model's certainty to the user through an interactive UI.
## 🚀 Key Achievements
*   **Designed** an end-to-end sentiment analysis system capable of detecting complex human emotions from raw text, utilizing Hugging Face Transformers and PyTorch.
*   **Optimized** a pre-trained DistilBERT architecture to execute multi-class classification across 6 distinct emotional categories, yielding a **92.8%** predictive accuracy.
*   **Constructed** a specialized natural language preprocessing pipeline featuring NLTK tokenization, slang normalization, and negation handling to heavily reduce contextual misclassifications.
*   **Published** the interactive web application on Streamlit Cloud, successfully managing and serving large `safetensors` model weights in production utilizing Git Large File Storage (LFS).
---
## 🛠️ Technology Stack
*   **Core Logic:** Python
*   **Deep Learning Framework:** PyTorch
*   **NLP Architecture:** Hugging Face Transformers (`DistilBERT`)
*   **Data Processing:** NLTK (Natural Language Toolkit)
*   **Model Serialization:** `.safetensors` (for highly secure, fast weight loading)
*   **Frontend Web Framework:** Streamlit
*   **Version Control & Hosting:** GitHub, Git LFS, Streamlit Cloud
---
## 📂 Project Structure
```text
Human-Emotion-Detection-App/
│
├── app/
│   └── app.py                      # Main Streamlit application script
│
├── emotion_model/                  # Fine-tuned model directory (Managed by Git LFS)
│   ├── config.json                 # Model architecture parameters
│   ├── model.safetensors           # Compressed & secure model weights
│   ├── special_tokens_map.json     # BERT special tokens (CLS, SEP, etc.)
│   ├── tokenizer.json              # FastTokenizer configuration
│   ├── tokenizer_config.json       # Preprocessing parameters
│   └── vocab.txt                   # Model vocabulary
│
├── .gitattributes                  # LFS tracking definitions
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project documentation
---
---

💻 How to Run Locally
Follow these steps to deploy the application on your local machine.

1. Clone the Repository
Note: Because this project uses Large File Storage for the model weights, ensure you have Git LFS installed before cloning.

bash
git lfs install
git clone https://github.com/Esha034/Human-Emotion-Detection-App
cd Human-Emotion-Detection-App
2. Set Up Virtual Environment
Isolate your dependencies to prevent package conflicts:

bash
# Create the virtual environment
python -m venv venv
# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate
3. Install Requirements
Install the necessary ML and web framework libraries:

bash
pip install -r requirements.txt
4. Launch the Application
Run the Streamlit server:

bash
streamlit run app/app.py
The application will automatically open in your default web browser at http://localhost:8501.

👨‍💻 Author
Eshani Banik
Data Scientist | Machine Learning Engineer
The application will automatically open in your default web browser at http://localhost:8501.

👨‍💻 Author
Eshani Banik
Data Scientist | Machine Learning Engineer
