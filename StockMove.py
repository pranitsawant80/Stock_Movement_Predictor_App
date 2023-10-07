import nltk
import streamlit as st
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained logistic regression model
model_path = logistic_regression_model.pkl
lr_clf_loaded = pickle.load(open(model_path, 'rb'))

# Load the fitted TF-IDF vectorizer
vectorizer_path = r'C:\Users\DELL\Desktop\PRTFLIO\Data-Science-Projects\Sentiment_Analysis_for_Dow_Jones_(DJIA_Stock)\tfidf_vectorizer.pkl'  # Replace with the correct file path
tfidf_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Function for cleaning the data
def clean_data(dataset):
     data = pd.Series(dataset).str.replace("[^a-zA-Z]", " ", regex=True)
     return data

# Function to perform lemmatization of the words
def lemmatize_data(data, lemmatizer):
    cleaned_dataset = []
    for i in range(len(data)):
        clean_text = data[i].lower()
        clean_text = clean_text.split()
        clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in stopwords.words('english')]
        cleaned_dataset.append(' '.join(clean_text))
    return cleaned_dataset


# Predictor function
def predictor(newdata):
    # Clean the data
    data1 = clean_data(newdata)

    # Perform lemmatization
    data2 = lemmatize_data(data1, WordNetLemmatizer())

    # Vectorize the new text data using the loaded TfidfVectorizer
    vectorized_data = tfidf_vectorizer.transform(data2)

    # Make a prediction
    predict = lr_clf_loaded.predict(vectorized_data)

    # Return the prediction
    if predict == 1:
        return 'Prediction: The Index Value will remain the SAME or will go UP.'
    else:
        return 'Prediction: The Index Value will go DOWN!'


# Streamlit app header
st.title("DJIA index predictor")

# User input for text data
user_input = st.text_area("Enter 20-30 top world news healines from the reddit site ", "")

# Define a function for prediction
def predict_stock_price_direction(input_text):
    # Check if the input text is not empty
    if input_text:
        # Perform the prediction using your functions
        prediction = predictor([input_text])
        return prediction
    else:
        return "ERROR"
# Button to trigger prediction
if st.button("Predict"):
    prediction = predict_stock_price_direction(user_input)
    st.write(prediction)

