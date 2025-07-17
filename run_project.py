import pandas as pd
import re
import emoji
import contractions
import spacy
from tqdm import tqdm
import os
import joblib
import shutil
import subprocess
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# --- This is an all-in-one script to run the entire project ---
# It will clean data, train the model, and launch the app, guaranteeing no file errors.

# --- PART 1: DATA CLEANING ---
def clean_data():
    print("--- PART 1: Starting Data Cleaning ---")
    
    # Setup spaCy
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    except OSError:
        print("\n❌ spaCy model 'en_core_web_sm' not found.")
        print("Please run this command in your terminal: python -m spacy download en_core_web_sm")
        return False
    stop_words = nlp.Defaults.stop_words

    # Load raw data
    input_path = "data/sentiment_analysis_for_mental_health.csv.csv"
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Raw data loaded successfully. Found {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Raw data file not found at '{input_path}'.")
        return False

    # Define cleaning function
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = contractions.fix(text)
        text = emoji.replace_emoji(text, replace='')
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Apply cleaning
    print("Applying text cleaning and lemmatization...")
    df['clean_text'] = [clean_text(text) for text in tqdm(df['statement'].tolist(), desc="Cleaning")]
    
    lemmatized_texts = []
    for doc in tqdm(nlp.pipe(df['clean_text'].tolist(), batch_size=50), total=len(df), desc="Lemmatizing"):
        lemmas = [token.lemma_ for token in doc if token.text not in stop_words]
        lemmatized_texts.append(" ".join(lemmas))
    df['clean_text'] = lemmatized_texts

    # Save cleaned data
    output_path = "data/cleaned_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved successfully at: '{output_path}'")
    return True

# --- PART 2: MODEL TRAINING ---
def train_model():
    print("\n--- PART 2: Starting Model Training ---")
    
    # HARD RESET: Delete old models folder to guarantee freshness
    if os.path.exists('models'):
        shutil.rmtree('models')
        print("✅ Deleted old 'models' folder.")
    
    # Load cleaned data
    input_path = "data/cleaned_data.csv"
    try:
        df = pd.read_csv(input_path)
        df = df.dropna(subset=['clean_text', 'status'])
    except FileNotFoundError:
        print(f"❌ ERROR: Cleaned data not found at '{input_path}'.")
        return False

    X = df['clean_text']
    y = df['status']

    # Preprocessing
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train_text, X_val_text, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english', min_df=5)
    X_train = vectorizer.fit_transform(X_train_text)
    print("✅ Vectorizer fitted successfully.")

    # Model training
    model = XGBClassifier(objective='multi:softprob', num_class=len(label_encoder.classes_), eval_metric='mlogloss', random_state=42, use_label_encoder=False, n_jobs=-1)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    print("✅ Model trained successfully.")

    # Save artifacts
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "xgb_model.pkl"))
    joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))
    joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
    print(f"✅ All model files saved successfully in '{save_dir}'.")
    return True

# --- PART 3: LAUNCH STREAMLIT APP ---
def launch_app():
    print("\n--- PART 3: Launching Streamlit App ---")
    print("If the app doesn't open, please manually open this URL in your browser.")
    
    # Find the path to the streamlit executable
    streamlit_path = os.path.join(os.path.dirname(sys.executable), 'streamlit')
    
    # Command to run
    command = [streamlit_path, 'run', 'app.py']
    
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find streamlit at '{streamlit_path}'.")
        print("Please try running the app manually: streamlit run app.py")
    except Exception as e:
        print(f"An error occurred while trying to launch the app: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if clean_data():
        if train_model():
            launch_app()
