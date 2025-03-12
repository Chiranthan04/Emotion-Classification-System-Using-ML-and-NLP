
import streamlit as st
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

# Load vectorizer and training data
 # Label data

def save_model(model, model_name):
    """Save the updated model."""
    joblib.dump(model, f"models/{model_name}.pkl")



import joblib
from scipy.sparse import vstack
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

def fine_tune_model(raw_text, label, model_name):
    """
    Fine-tune the Logistic Regression model with new text and label.

    Args:
    - raw_text (str): Input text to fine-tune the model.
    - label (str): Correct label for the input text.
    - model_name (str): Name of the model to fine-tune (e.g., "LogisticRegression").

    Returns:
    - None: The fine-tuned model and updated training data are saved to disk.
    """
    # Validate the model name
    if model_name != "LogisticRegression":
        raise ValueError(f"This function only supports LogisticRegression. Got: {model_name}")

    # Load pre-trained components
    vectorizer = joblib.load(r"models/DecisionTreeClassifier.pkl")
    # encoder = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\encoder.pkl")  # Assuming you have a label encoder
    # X_train = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\x_train.pkl")  # Feature data (should be vectorized)
    # y_train = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\y_train.pkl")           # Load training labels
    model = joblib.load(r"models/LogisticRegression.pkl")

    # Vectorize the new input text
    text_vectorized = vectorizer.transform([raw_text])

    # Ensure dimensions match
    if text_vectorized.shape[1] != X_train.shape[1]:
        raise ValueError(f"Dimension mismatch: X_train has {X_train.shape[1]} features, "
                         f"but text_vectorized has {text_vectorized.shape[1]} features.")

    # Encode the label
    label_encoded = encoder.transform([label])[0]

    # Update the training dataset
    X_train = vstack([X_train, text_vectorized])  # Use vstack to concatenate sparse matrices
    y_train = shuffle(y_train, y_train, random_state=42)

    # Fine-tune the Logistic Regression model
    model.fit(X_train, y_train)

    # Save the updated model and training data
    joblib.dump(model, "models/LogisticRegression.pkl")
    joblib.dump(X_train, "models/x_train.pkl")
    joblib.dump(y_train, "models/y_train.pkl")

    print("Logistic Regression model fine-tuned and saved successfully!")



# def fine_tune_model(raw_text, label, model_name):
#     global X_train, y_train
#     st.subheader({raw_text})
#     # Load the vectorizer and vectorize new text
#     # vectorizer = load("models/countvectorizer.pkl")
#     text_vectorized = vectorizer.transform([raw_text])

#     # Check for dimension mismatch
#     if text_vectorized.shape[1] != X_train.shape[1]:
#         raise ValueError("Dimension mismatch: Ensure consistent vectorizer is used.")

#     # Encode label
#     label_encoded = encoder.transform([label])[0]

#     # Add new data
#     X_train = vstack([X_train, text_vectorized])  # Concatenate using sparse matrices
#     y_train = np.hstack([y_train, [label_encoded]])

#     # Shuffle and retrain the model
#     X_train, y_train = shuffle(X_train, y_train, random_state=42)

#     if model_name == "LogisticRegression":
#         model = LogisticRegression(max_iter=1000, n_jobs=-1)
#     elif model_name == "DecisionTreeClassifier":
#         model = DecisionTreeClassifier(max_depth=25)
#     elif model_name == "RandomForestClassifier":
#         model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
#     else:
#         raise ValueError(f"Unknown model name: {model_name}")

#     # Train the updated model
#     model.fit(X_train, y_train)

#     # Save the updated model
#     save_model(model, model_name)


