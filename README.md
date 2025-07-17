# Mental-Health-Prediction-And-support-system
🧠 AI-Powered Mental Health Prediction System
An intelligent web application that uses Natural Language Processing and Machine Learning to predict potential mental health conditions from user-provided text. This tool serves as a preliminary support system, not a diagnostic tool.

✨ Features
Real-Time Analysis: Instantly analyzes text to predict conditions like Anxiety, Depression, Stress, and more.

NLP-Powered Cleaning: Employs a robust text preprocessing pipeline using spaCy for lemmatization, stopword removal, and more.

XGBoost Model: Uses a powerful and efficient XGBoost classifier for accurate predictions.

Interactive UI: A simple and intuitive user interface built with Streamlit.

Critical Alerts: Provides a special warning and resources for text indicating potential suicidal ideation.

Ethical Disclaimer: Clearly communicates that the tool is not a substitute for professional medical advice.

🛠️ Tech Stack
🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

1. Prerequisites
Python 3.8 or higher

pip package manager

2. Clone the Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

3. Set Up a Virtual Environment (Recommended)
It's best practice to create a virtual environment to keep project dependencies isolated.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
Install all the required packages using the requirements.txt file.

pip install -r requirements.txt

5. Download the spaCy Language Model
The project requires the small English model from spaCy for text processing.

python -m spacy download en_core_web_sm

🏃‍♂️ How to Run the Project
The project is designed to be run in a specific sequence.

Step 1: Clean the Raw Data
First, run the data cleaning script to preprocess the raw dataset.

python data_cleaning.py

This will generate a cleaned_data.csv file inside the /data folder.

Step 2: Train the Machine Learning Model
Next, run the training script. This will use the cleaned data to train the model and save the necessary .pkl files.

python train_model.py

This will create a /models folder containing xgb_model.pkl, tfidf_vectorizer.pkl, and label_encoder.pkl.

Step 3: Launch the Streamlit Web App
Finally, run the main application script.

streamlit run app.py

Your web browser should automatically open with the application running. If not, open the "Local URL" provided in your terminal.

⚠️ Disclaimer
This tool is an educational project and is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing mental health concerns, please consult a qualified healthcare professional. For immediate help in a crisis, please contact a suicide prevention hotline.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
