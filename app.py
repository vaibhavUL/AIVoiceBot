import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)


model = joblib.load('models/shoe_type_model.pkl')
vectorizer = joblib.load('models/question_vectorizer.pkl')
qa_data = pd.read_csv('question_answer.csv')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json.get('query')  # User's voice query text
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    
    user_vector = vectorizer.transform([user_input])
    questions_vectorized = vectorizer.transform(qa_data['question'])

    
    similarities = cosine_similarity(user_vector, questions_vectorized)
    best_match_index = similarities.argmax()

    
    response = qa_data.iloc[best_match_index]['answer']
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
