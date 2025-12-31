import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Human Emotion Detection"
MODEL_PATH = "emotion_model"
MAX_SEQ_LEN = 128
NEGATION_WINDOW = 3
HISTORY_LIMIT = 5

EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="centered"
)

st.title("Human Emotion Detection")
st.caption("Industry-grade transformer-based emotion classification")

st.markdown(
    """
    <style>
    textarea {
        border: 1px solid #d0d0d0 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-size: 15px !important;
    }
    textarea:focus {
        border-color: #4f8bf9 !important;
        box-shadow: 0 0 0 1px #4f8bf9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# NLTK SETUP (CACHED)
# =========================================================
@st.cache_resource(show_spinner=False)
def setup_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    return set(stopwords.words("english"))

STOP_WORDS = setup_nltk()

# =========================================================
# TEXT NORMALIZATION RESOURCES
# =========================================================
EMOJI_MAP = {
    "😢": "sad", "😭": "sad",
    "😡": "angry", "😠": "angry",
    "❤️": "love", "😍": "love",
    "😊": "happy", "😂": "happy",
    "😨": "fear", "😱": "fear",
    "😮": "surprise", "😲": "surprise"
}

SLANG_MAP = {
    "idk": "i do not know",
    "lol": "laughing",
    "omg": "oh my god",
    "wtf": "what the fuck",
    "brb": "be right back",
    "btw": "by the way"
}

# =========================================================
# NLP UTILITIES
# =========================================================
def handle_negation(text: str, window: int = NEGATION_WINDOW) -> str:
    words = text.split()
    negators = {"not", "no", "never", "n't"}
    result, negate_count = [], 0

    for word in words:
        if word in negators:
            negate_count = window
            result.append(word)
        elif negate_count > 0:
            result.append(f"NOT_{word}")
            negate_count -= 1
        else:
            result.append(word)

    return " ".join(result)


def clean_text(text: str) -> str:
    text = text.lower()

    for emoji, meaning in EMOJI_MAP.items():
        text = text.replace(emoji, meaning)

    for slang, meaning in SLANG_MAP.items():
        text = re.sub(rf"\b{slang}\b", meaning, text)

    text = handle_negation(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens)

# =========================================================
# MODEL LOADING (CACHED & SAFE)
# =========================================================
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(" Failed to load emotion model.")
        st.stop()

TOKENIZER, MODEL = load_model()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)

# =========================================================
# PREDICTION LOGIC
# =========================================================
def predict_emotion(text: str) -> dict:
    cleaned = clean_text(text)

    inputs = TOKENIZER(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LEN
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    return dict(
        sorted(
            {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []
from datetime import timedelta

def relative_time(timestamp: datetime) -> str:
    diff = datetime.now() - timestamp

    if diff < timedelta(seconds=30):
        return "Just now"
    elif diff < timedelta(minutes=1):
        return "Less than a minute ago"
    elif diff < timedelta(minutes=60):
        return f"{int(diff.seconds / 60)} minute(s) ago"
    else:
        return timestamp.strftime("%b %d, %I:%M %p")

# =========================================================
# UI
# =========================================================
user_input = st.text_area(
    "✍️ Enter text",
    placeholder="Type your message here...",
    height=120,
    help="Enter a sentence expressing emotion."
)

col1, col2 = st.columns([1, 2])
with col1:
    detect_btn = st.button("Detect Emotion", use_container_width=True)

if detect_btn:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing emotion..."):
            results = predict_emotion(user_input)

        top_emotion, top_score = next(iter(results.items()))

        st.subheader("Emotion Confidence")
        for emo, score in results.items():
            st.progress(score, text=f"{emo.capitalize()} — {score:.2f}")

        st.success(f"Detected Emotion: **{top_emotion.upper()}** ({top_score * 100:.2f}%)"
)


        st.session_state.history.append({
    "time": datetime.now(),   # store full datetime
    "emotion": top_emotion,
    "text": user_input
})


# =========================================================
# HISTORY VIEW
# =========================================================
if st.session_state.history:
    st.subheader("Recent Predictions")
    for item in reversed(st.session_state.history[-HISTORY_LIMIT:]):
        time_label = relative_time(item["time"])
        st.markdown(
            f"- **{item['emotion'].capitalize()}** · {time_label}<br>"
            f"<span style='color: #555;'>{item['text']}</span>",
            unsafe_allow_html=True
        )