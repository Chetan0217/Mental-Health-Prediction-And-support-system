import streamlit as st
import joblib  # Use joblib for loading models
import re
import emoji
import contractions
import spacy
import os
import numpy as np
import sys

# --- 1. SETUP AND CONFIGURATION ---

st.set_page_config(page_title="🧠 Mental Health Predictor", layout="centered")

# Load the spaCy model once for efficiency, caching it
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        st.stop()

nlp = load_spacy_model()
stop_words = nlp.Defaults.stop_words

# --- 2. LOAD MODEL ARTIFACTS ---

# Define the relative path to the models directory
MODEL_DIR = "models"

# Safe loader function using joblib
@st.cache_resource
def safe_load_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        st.error(f"❌ Missing required file: `{file_name}` in `{MODEL_DIR}/` folder.")
        st.info("Please run the 'run_project.py' or 'train_model.py' script to generate the model files.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading `{file_name}`. It might be corrupted. Details: {e}")
        st.stop()

# Load the trained components
model = safe_load_model("xgb_model.pkl")
vectorizer = safe_load_model("tfidf_vectorizer.pkl")
label_encoder = safe_load_model("label_encoder.pkl")

# --- 3. PREPROCESSING FUNCTION (MUST MATCH TRAINING) ---

def full_preprocess(text):
    """Performs the exact same preprocessing steps used during model training."""
    if not isinstance(text, str): return ""
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(lemmas)

# --- 4. STREAMLIT UI ---

st.title("🧠 Mental Health Prediction System")
st.markdown("""
Welcome! This tool uses a Machine Learning model (XGBoost) to predict potential mental health conditions based on the text you provide.
*Disclaimer: This is an AI-based prediction and not a medical diagnosis. Please consult a healthcare professional for any health concerns.*
""")

user_input = st.text_area(
    "Enter your text here:", 
    height=150, 
    placeholder="e.g., 'I've been feeling really overwhelmed and anxious lately...'"
)

if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            clean_text = full_preprocess(user_input)
            if len(clean_text.split()) < 3:
                st.warning("⚠️ Text is too short. Please provide more details.")
            else:
                try:
                    text_vector = vectorizer.transform([clean_text])
                    prediction_proba = model.predict_proba(text_vector)
                    prediction = np.argmax(prediction_proba, axis=1)
                    predicted_label = label_encoder.inverse_transform(prediction)[0]
                    confidence_score = prediction_proba[0][prediction[0]]

                    st.success(f"**Prediction: {predicted_label}** (Confidence: {confidence_score:.2%})")

                    explanations = {
                        "Anxiety": "The language shows patterns associated with worry or nervousness.",
                        "Bipolar": "The text may contain expressions related to mood extremes.",
                        "Depression": "The model detected words linked to sadness or low energy.",
                        "Normal": "The text does not show significant indicators of the trained conditions.",
                        "Personality disorder": "The language suggests patterns related to identity or relationship challenges.",
                        "Stress": "The model identified terms associated with feeling under pressure or overwhelmed.",
                        "Suicidal": "⚠️ **CRITICAL:** The text contains language indicative of suicidal thoughts. **Please seek immediate help.**"
                    }
                    if predicted_label in explanations:
                        st.info(f"**Interpretation:** {explanations[predicted_label]}")
                    if predicted_label == "Suicidal":
                        st.error("**Resources:**\n- **National Suicide Prevention Lifeline:** 988\n- **Crisis Text Line:** Text HOME to 741741")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
