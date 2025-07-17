import pandas as pd
import re
import emoji
import contractions
import spacy
from tqdm import tqdm
import os

# --- 1. Setup ---
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("\n❌ spaCy model 'en_core_web_sm' not found.")
    print("Please run this command in your terminal: python -m spacy download en_core_web_sm")
    exit()

stop_words = nlp.Defaults.stop_words

# --- 2. Load Data using the CORRECT Filename ---
input_path = "data/sentiment_analysis_for_mental_health.csv.csv" 
print(f"\nLoading data from '{input_path}'...")
try:
    df = pd.read_csv(input_path)
    print(f"✅ Successfully loaded data. Found {len(df)} rows.")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at '{input_path}'.")
    exit()

# --- 3. Define the Full Cleaning Function ---
def clean_text(text):
    """A robust function to clean text data."""
    if not isinstance(text, str):
        return ""
    
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 4. Apply Cleaning Steps ---
print("\nApplying initial cleaning (regex, lowercase, etc.)...")
# This is the updated, more robust way to show a progress bar.
# It avoids the 'progress_apply' error.
texts_to_clean = df['statement'].tolist()
df['clean_text'] = [clean_text(text) for text in tqdm(texts_to_clean, desc="Initial Cleaning")]


print("\nApplying advanced cleaning (lemmatization & stopword removal)...")
lemmatized_texts = []
texts_to_process = df['clean_text'].tolist()

for doc in tqdm(nlp.pipe(texts_to_process, batch_size=50), total=len(texts_to_process), desc="Lemmatizing"):
    lemmas = [token.lemma_ for token in doc if token.text not in stop_words]
    lemmatized_texts.append(" ".join(lemmas))

df['clean_text'] = lemmatized_texts

# --- 5. Preview and Save Cleaned Data ---
print("\n🔎 Preview of cleaned data:")
print(df[['statement', 'clean_text']].head())

output_path = "data/cleaned_data.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved successfully at: '{output_path}'")
