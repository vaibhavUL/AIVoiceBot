import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_and_save_model():
    """
    Train a RandomForest model on the shoe dataset and save it along with encoders.
    """
    
    try:
        data = pd.read_csv('shoe_data.csv')
    except FileNotFoundError:
        print("Error: The file 'data/shoe_data.csv' was not found. Please ensure the file exists.")
        return

    
    encoders = {}
    for column in ['brand', 'color', 'material', 'shoe_type']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le

   
    X = data[['size', 'brand', 'color', 'material', 'price']]
    y = data['shoe_type']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    
    if not os.path.exists('models'):
        os.makedirs('models')

    
    try:
        joblib.dump(model, 'models/shoe_type_model.pkl')
        for col, encoder in encoders.items():
            joblib.dump(encoder, f'models/{col}_encoder.pkl')
        print("Model and encoders saved successfully in the 'models' directory.")
    except Exception as e:
        print(f"Error saving model or encoders: {e}")

if __name__ == "__main__":
    train_and_save_model()
