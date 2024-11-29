import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only once)
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the dataset
try:
    print("Loading the dataset...")
    data = pd.read_csv('spam.csv', encoding='latin-1')
    print("Dataset loaded successfully.")
    print(data.head())  # Display first 5 rows of the dataset
except Exception as e:
    print(f"An error occurred: {e}")

# Step 2: Data Cleaning
data = data[['Category', 'Message']]  # Keep only necessary columns
data = data.dropna()  # Handle missing values

# Custom text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ''.join([char for char in text if not char.isdigit()])  # Remove numbers
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize and remove stopwords
    return text

# Apply the cleaning function to the 'Message' column
data['Message'] = data['Message'].apply(clean_text)

# Step 3: Label Encoding
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'], test_size=0.2, random_state=42)

# Step 5: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

model = GridSearchCV(LogisticRegression(max_iter=2000, class_weight='balanced'), param_grid, cv=5)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate performance
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#NEW mmethod from here !!!!!!!!!!
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Grid search to find optimal parameters for SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]  # Only for RBF kernel
}

# Set up the SVM model
svm_model = SVC(probability=True)

# Perform grid search
grid_search = GridSearchCV(svm_model, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# Best parameters from grid search
print("Best Parameters: ", grid_search.best_params_)

# Evaluate the model with the best parameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_tfidf)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Code for Confusion Matrix:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix
y_pred = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix')
plt.show()


#Code for ROC Curve:
from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test_tfidf)[:, 1])

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()  

#HERE AM SAVING THE CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import pickle

# Save the vectorizer and the model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("spam_model.pkl", "wb") as f:
    pickle.dump(best_model, f)