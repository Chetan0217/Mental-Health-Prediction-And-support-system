import pandas as pd
import numpy as np
import os
import joblib  # Use joblib for saving/loading sklearn objects

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# --- 1. Load Data using RELATIVE Paths ---
# This makes your script portable and work anywhere.
input_path = "data/cleaned_data.csv"
try:
    df = pd.read_csv(input_path)
    # Ensure no lingering null values from the cleaning process
    df = df.dropna(subset=['clean_text', 'status']) 
    print(f"✅ Data loaded successfully from '{input_path}'")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at '{input_path}'. Did you run the data cleaning script first?")
    exit()

# --- 2. Preprocessing and Splitting ---
X = df['clean_text']
y = df['status']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Stratified train-test split
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 3. Feature Engineering (TF-IDF) ---
# Define and fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=5
)

# Fit on training data ONLY to prevent data leakage
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
print(f"✅ Vectorizer fitted. Vocabulary size: {len(vectorizer.vocabulary_)}")

# --- 4. Model Training ---
# Compute class weights for handling imbalanced data
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Initialize XGBoost model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1
)

print("\n🏋️ Training XGBoost model...")
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False # Set to True if you want to see training progress
)
print("✅ Model training complete.")

# --- 5. Save Artifacts using Joblib and a RELATIVE Path ---
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# Use joblib which is more efficient for sklearn objects
joblib.dump(model, os.path.join(save_dir, "xgb_model.pkl"))
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))
joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))

print(f"\n✅ Model, encoder, and vectorizer saved successfully in '{save_dir}' directory.")

# --- 6. Evaluation ---
y_pred = model.predict(X_val)
print("\n🎯 Accuracy:", accuracy_score(y_val, y_pred))
print("\n📋 Classification Report:\n", classification_report(
    y_val, y_pred, target_names=label_encoder.classes_))
