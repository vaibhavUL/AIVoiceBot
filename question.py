import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def save_question_vectorizer():
    # Example questions for training the vectorizer
    questions = [
        "What is the size of the shoe?",
        "What are the available colors?",
        "Tell me about the shoe material.",
        "What is the price of the shoe?",
        "List all the brands available."
    ]

    # Train a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(questions)

    # Ensure 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the vectorizer
    joblib.dump(vectorizer, 'models/question_vectorizer.pkl')
    print("Question vectorizer saved successfully in 'models' directory.")

if __name__ == "__main__":
    save_question_vectorizer()
