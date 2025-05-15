import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from neattext.functions import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv('emotion_detection/emotion_data.csv')
    
    # Clean text using NeatText
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x, puncts=True, stopwords=True))
    
    return df

# Step 2: Build and train the model
def train_model(df):
    # Features and labels
    X = df['cleaned_text']
    y = df['emotion']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return pipeline, X_test, y_test

# Step 3: Save the model
def save_model(pipeline, filename='emotion_model.pkl'):
    joblib.dump(pipeline, filename)
    print(f"Model saved as {filename}")

# Step 4: Demo interface for real-time prediction
def predict_emotion(pipeline):
    print("\nEmotion Detection Demo")
    print("Enter 'quit' to exit")
    
    while True:
        user_input = input("\nEnter a sentence: ")
        if user_input.lower() == 'quit':
            break
        
        # Clean the input text
        cleaned_input = clean_text(user_input, puncts=True, stopwords=True)
        
        # Predict emotion
        prediction = pipeline.predict([cleaned_input])[0]
        print(f"Predicted Emotion: {prediction}")

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data_file = 'emotion_data.csv'
    df = load_and_preprocess_data(data_file)
    
    # Train the model
    model, X_test, y_test = train_model(df)
    
    # Save the model
    save_model(model)
    
    # Run the demo
    predict_emotion(model)