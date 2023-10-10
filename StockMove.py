import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')
nltk.download('popular')

import streamlit as st
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load the trained logistic regression model
model_path = 'logistic_regression_model.pkl'  # Replace with the correct file path
lr_clf_loaded = pickle.load(open(model_path, 'rb'))

# Load the fitted TF-IDF vectorizer
vectorizer_path ='tfidf_vectorizer.pkl'  # Replace with the correct file path
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

    if len(newdata) >= 2500:
        if predict == 1:
            prediction = 'The Index Value will remain the SAME or will go UP.'
        else:
            prediction = 'The Index Value will go DOWN!'
        return prediction
    else:
        return ('Collect more news (>=2000 words should be entered)', f'No of words in the document is :{len(newdata)}')

# Streamlit app header
st.title("DJIA index predictor")

# User input for text data
user_input = st.text_area("Enter 20-30 top world news healines from the reddit site ", "")

# Define a function for prediction
def predict_stock_price_direction(input_text):
    # Check if the input text is not empty
    if input_text:
        # Perform the prediction using your functions
        prediction = predictor(input_text)
        return prediction
    else:
        return "ERROR"
# Button to trigger prediction
if st.button("Predict"):
    prediction = predict_stock_price_direction(user_input)
    st.write(prediction)


# Add a sidebar header
st.sidebar.header("More to explore")
st.sidebar.subheader('"Rate the Performance of the predictor"')
# Create a selectbox for rating
rating = st.sidebar.selectbox("Select your rating:", ["1 (Low)", "2", "3", "4", "5 (High)"])
st.sidebar.write(f"You rated the prediction as: {rating}")

# Add a link to the Reddit website with the header "Reddit Website"
st.sidebar.subheader('Visit Reddit News')
st.sidebar.markdown("[Visit](https://www.reddit.com/r/worldnews/)")

# # URL to the PDF file on Google Drive (replace with your own URL)
st.subheader("Download Section")

# URL to the PDF file on Google Drive (replace with your own URL)
pdf_url = "https://drive.google.com/file/d/13Rc2Et8KRM4HIUyfqPfBvVeWrfrk_4fk/view?usp=sharing"

# Create a download button
if st.button("Download User Guidelines"):
    st.markdown("Downloading User Guidelines... Please wait.")
    # Use 'requests' library to download the file
    import requests

    response = requests.get(pdf_url)
    if response.status_code == 200:
        # Set the content type to PDF
        st.markdown("Download Complete!")
        st.markdown(f"Downloaded file as [User Guidelines.pdf]({pdf_url})")
    else:
        st.markdown("Failed to download User Guidelines. Please try again later.")
