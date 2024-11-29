import pickle
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the saved model and vectorizer (adjust names based on what you save later)
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("spam_model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # HTML file for user input

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    message = request.form['message']
    data = [message]

    # Transform the input
    vect_data = vectorizer.transform(data)

    # Predict using the model
    prediction = model.predict(vect_data)[0]

    # Display results
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction_text=f'The email is: {result}')

if __name__ == "__main__":
    app.run(debug=True)