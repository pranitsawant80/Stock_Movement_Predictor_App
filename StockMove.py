import streamlit as st
import predict
# Streamlit app header
st.title("DJIA index predictor")

# User input for text data
user_input = st.text_area("Enter 20-30 top world news healines from the reddit site ", "")

# Define a function for prediction
def predict_stock_price_direction(input_text):
    # Check if the input text is not empty
    # Check if the input text is not empty
    if input_text:
        # Perform the prediction using your functions
        prediction = predict.predictor([input_text])
        return prediction
    else:
        return "ERROR"
# Button to trigger prediction
if st.button("Predict"):
    prediction = predict_stock_price_direction(user_input)
    st.write(prediction)

