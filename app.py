import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="EmotiSense AI",
    page_icon="🧠",
    layout="centered"
)

# -------------------- SIMPLE STYLE --------------------
st.markdown("""
<style>
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #777;
    margin-bottom: 2rem;
}
.result {
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 12px;
    text-align: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 EmotiSense AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Emotion Classification using Bidirectional GRU & NLP</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a sentence and EmotiSense AI will predict the emotion expressed in the text."
)

# These labels follow the same order used during notebook training.
LABELS = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

# Keep the same sequence length used during training.
MAX_LENGTH = 50

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Artifacts" / "BiGRU_Modle.keras"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer.pkl"

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

text = st.text_area(
    "Enter text",
    placeholder="Example: I am very happy today because I achieved my goal!",
    height=130
)

if st.button("🔍 Detect Emotion", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            model, tokenizer = load_artifacts()

            sequence = tokenizer.texts_to_sequences([text.strip()])
            padded = pad_sequences(
                sequence,
                maxlen=MAX_LENGTH,
                padding="post",
                truncating="post"
            )

            probabilities = model.predict(padded, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            emotion = LABELS[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

            emoji = {
                "Sadness": "😢",
                "Joy": "😊",
                "Love": "❤️",
                "Anger": "😠",
                "Fear": "😨",
                "Surprise": "😲"
            }.get(emotion, "🧠")

            st.success(f"{emoji} Predicted Emotion: **{emotion}**")
            st.progress(min(max(confidence / 100, 0.0), 1.0))
            st.caption(f"Model confidence: {confidence:.2f}%")

            with st.expander("View all emotion probabilities"):
                for label, probability in zip(LABELS, probabilities):
                    st.write(f"**{label}:** {float(probability) * 100:.2f}%")

        except FileNotFoundError:
            st.error(
                "Model files were not found. Make sure the Artifacts folder contains "
                "BiGRU_Modle.keras and tokenizer.pkl."
            )
        except Exception as error:
            st.error("The prediction could not be completed.")
            st.code(f"{type(error).__name__}: {error}")

st.divider()

st.subheader("Emotions detected")
cols = st.columns(3)
emotions = [
    "😢 Sadness", "😊 Joy", "❤️ Love",
    "😠 Anger", "😨 Fear", "😲 Surprise"
]
for i, emotion in enumerate(emotions):
    cols[i % 3].write(emotion)

st.divider()
st.caption(
    "EmotiSense AI • Bidirectional GRU • NLP • TensorFlow • Streamlit"
)
