import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained logistic regression model
model_path = r'C:\Users\DELL\Desktop\PRTFLIO\Data-Science-Projects\Sentiment_Analysis_for_Dow_Jones_(DJIA_Stock)\logistic_regression_model.pkl'  # Replace with the correct file path
lr_clf_loaded = pickle.load(open(model_path, 'rb'))

# Load the fitted TF-IDF vectorizer
vectorizer_path = r'C:\Users\DELL\Desktop\PRTFLIO\Data-Science-Projects\Sentiment_Analysis_for_Dow_Jones_(DJIA_Stock)\tfidf_vectorizer.pkl'  # Replace with the correct file path
tfidf_vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# # Load the TfidfVectorizer from the pickle file
# with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
#     loaded_cv = pickle.load(vectorizer_file)

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
