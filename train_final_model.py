import pandas as pd
import numpy as np
import nltk
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load your dataset
df = pd.read_csv(r"C:\Users\mehta\OneDrive\Desktop\Fine tuning\data\cleaned_data.csv")
df = df.dropna(subset=['clean_text', 'status'])

X = df['clean_text']
y = df['status']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ✅ Vectorizer setup
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=5
)

# ✅ Fit vectorizer
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

# ✅ Confirm it’s fitted
print("✅ Vocabulary Size:", len(vectorizer.vocabulary_))
print("✅ Testing transform:", vectorizer.transform(["test sentence"]).shape)

# ✅ Train model
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ✅ Save to correct folder
save_dir = r"C:\Users\mehta\OneDrive\Desktop\Fine tuning\models"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "xgb_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

with open(os.path.join(save_dir, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ All files saved to:", save_dir)

# ✅ Reload & test
print("\n🧪 Final Check: Loading vectorizer again...")
try:
    with open(os.path.join(save_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        test_vec = pickle.load(f)
    _ = test_vec.transform(["this is just a test"])
    print("✅ Vectorizer is fitted and working.")
except Exception as e:
    print("❌ Vectorizer is NOT fitted correctly!")
    print(e)
