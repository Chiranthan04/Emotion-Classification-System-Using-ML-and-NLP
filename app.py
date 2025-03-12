# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 

# Track Utils
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details,view_all_prediction_details, add_prediction_details

import streamlit as st
from models import fine_tune_model  # Import the function
import joblib

# # Load vectorizer, encoder, and training data
# vectorizer = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\countvectorizer.pkl")
# encoder = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\encoder.pkl")  # Assuming you have a label encoder
# X_train = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\x_train.pkl")  # Feature data (should be vectorized)
# Y_train = joblib.load(r"C:\Users\Chiranthan\Documents\Mini_pro\App\models\y_train.pkl")  # Label data


# Define your labels
labels = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Bipolar",
    "Stress",
    "Personality disorder"
]

# Function to load the selected model
def load_model(model_name):
    model_path = f"models\\{model_name}.pkl"
    return joblib.load(open(model_path, "rb"))
    # return joblib.load(open("C:\\Users\\Chiranthan\\Documents\\Mini_pro\\App\\models\\LogisticRegression.pkl", "rb"))


# Function to predict emotions
def predict_emotions(model, docx):
    results = model.predict([docx])
    return results[0]

def get_prediction_proba(model, docx):
    results = model.predict_proba([docx])
    return results

# Emoji Dictionary
emotions_emoji_dict = {
    "Anxiety": "😰",
    "Bipolar": "🎭",
    "Depression": "😔",
    "Normal": "😊",
    "Personality disorder": "🧩",
    "Stress": "😩",
    "Suicidal": "💔"
}
# Main Application
def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Dropdown for model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["LogisticRegression", "DecisiontreeClassifier", "RandomForestClassifier"]
    )
   
    # Load the selected model
    model = load_model(model_option)

    if choice == "Home":
        st.subheader(f"Home - Emotion In Text")
        st.subheader(f"Model: {model_option}")
        # Initialize session states for feedback and prediction
        if "feedback_triggered" not in st.session_state:
            st.session_state.feedback_triggered = False
        if "last_prediction" not in st.session_state:
            st.session_state.last_prediction = None
        if "last_raw_text" not in st.session_state:
            st.session_state.last_raw_text = None

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here", value=st.session_state.last_raw_text or "")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            # Save raw text in session state
            st.session_state.last_raw_text = raw_text

            col1, col2 = st.columns(2)

            # Apply Prediction Function
            prediction = predict_emotions(model, raw_text)
            probability = get_prediction_proba(model, raw_text)

            # Save prediction in session state
            st.session_state.last_prediction = {
                "text": raw_text,
                "prediction": prediction,
                "probability": probability
            }

            add_prediction_details(raw_text, prediction, model_option, np.max(probability), datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {np.max(probability)}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=model.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probability', color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)

            # Reset feedback state
            st.session_state.feedback_triggered = False

        # Feedback Section
        if st.session_state.last_prediction:
            st.markdown("---")
            col3, col4 = st.columns(2)

            with col3:
                if st.button("👍 Correct"):
                    st.success("Thank you for your feedback! The prediction is correct.")
                    st.session_state.feedback_triggered = False  # Reset feedback state

            with col4:
                if st.button("👎 Incorrect"):
                    st.warning("Sorry about the incorrect prediction.")
                    st.session_state.feedback_triggered = True  # Trigger feedback logic

        # Display fine-tuning options when feedback is triggered
        if st.session_state.feedback_triggered:
            st.subheader("Fine-Tune the Model")
            selected_model = st.selectbox(
                "Select the model to fine-tune:",
                options=["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]
            )
            selected_labels = st.multiselect(
                "Select the correct label(s):",
                options=list(emotions_emoji_dict.keys()),
                default=None
            )

            if st.button("Submit Feedback"):
                if selected_labels and selected_model:
                    st.success(f"Fine-tuning '{selected_model}' with labels: {', '.join(selected_labels)}")
                    for label in selected_labels:
                        fine_tune_model(st.session_state.last_raw_text,label, selected_model)  # Only passing label and model name
                    st.success(f"The {selected_model} model has been fine-tuned with your feedback!")
                    st.session_state.feedback_triggered = False
                else:
                    st.error("Please select a model and at least one label.")


    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(
                view_all_page_visited_details(),
                columns=['Pagename', 'Time_of_Visit']
            )
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(
                view_all_prediction_details(),
                columns=['Rawtext', 'Prediction','model_used','Probability', 'Time_of_Visit']
            )
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(
                x='Prediction', y='Counts', color='Prediction'
            )
            st.altair_chart(pc, use_container_width=True)

    else:
        st.subheader("About")
        add_page_visited_details("About", datetime.now())
        
        st.write("""
        ### About the Emotion Classification System
        
        This project is designed to classify text-based user inputs into predefined emotion categories:
        **Normal**, **Depression**, **Stress**, **Anxiety**, **Bipolar Disorder**, **Personality Disorder**, and **Suicidal Tendencies**. 
        By utilizing machine learning models like **Logistic Regression**, **Decision Tree**, and **Random Forest**, the system provides reliable 
        and interpretable predictions for emotional states.
        
        #### Key Features:
        - **User-Friendly Interface**: A simple and intuitive design powered by Streamlit for easy interaction.
        - **Multiple Model Options**: Users can choose from three pre-trained machine learning models to compare results.
        - **Performance Visualization**: Displays model accuracy and comparative performance metrics for better transparency.
        - **Real-Time Predictions**: Instant feedback on the input text with detailed insights into emotion classification.
        
        #### Purpose:
        This project aims to bridge the gap between advanced machine learning technologies and practical usability, providing 
        a tool that is accessible to non-technical users. Its applications include:
        - **Mental Health Awareness**: Helping individuals and professionals gain insights into emotional states.
        - **Sentiment Analysis**: Useful for understanding user behavior and feedback.
        - **Educational Value**: Demonstrating the application of machine learning in emotion classification.
        
        #### Technologies Used:
        - **Python** for backend implementation.
        - **Scikit-learn** for training and deploying machine learning models.
        - **Streamlit** for creating an interactive and responsive web interface.
        - **Matplotlib** and **Altair** for data visualization.
        - **Joblib** for saving and loading machine learning models.
        
        This system is a step towards leveraging technology for mental health awareness and sentiment analysis, aiming to make 
        emotion classification accessible and meaningful to a broad audience.
        """)


if __name__ == '__main__':
    main()
