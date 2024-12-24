from flask import Flask, jsonify
from flask_cors import CORS 
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

DATA_PATH = 'question_answer.csv'
MODEL_DIR = 'models'
VECTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'question_vectorizer.pkl')
QUESTION_VECTORS_PATH = os.path.join(MODEL_DIR, 'question_vectors.pkl')


def train_and_save_question_answer_model():
    # Load dataset
    qa_data = pd.read_csv(DATA_PATH)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize questions
    question_vectors = vectorizer.fit_transform(qa_data['question'])

   
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTOR_MODEL_PATH)
    joblib.dump(question_vectors, QUESTION_VECTORS_PATH)

    print("Model and vectorizer saved.")


def find_closest_answer(user_query):
    # Load vectorizer and question vectors
    vectorizer = joblib.load(VECTOR_MODEL_PATH)
    question_vectors = joblib.load(QUESTION_VECTORS_PATH)

    
    qa_data = pd.read_csv(DATA_PATH)

    
    user_vector = vectorizer.transform([user_query])

    
    similarities = cosine_similarity(user_vector, question_vectors)

    
    closest_idx = np.argmax(similarities)

   
    return qa_data['answer'][closest_idx]

# Route to handle the voice interaction
@app.route('/voice', methods=['GET'])
def voice_input():
    try:
        
        user_query = listen()

       
        answer = find_closest_answer(user_query)
        speak(answer)
        # Return the answer as a JSON response
        return jsonify({'shoe_type': answer})

    except Exception as e:
        return jsonify({'error': str(e)})


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "There was an issue with the speech recognition service."

if __name__ == "__main__":
    # Train model if not already trained
    if not os.path.exists(VECTOR_MODEL_PATH) or not os.path.exists(QUESTION_VECTORS_PATH):
        train_and_save_question_answer_model()

    print("Voice Assistant API is running!")

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
