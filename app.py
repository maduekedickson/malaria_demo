import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import os

# Load the image at the top of the app
image = Image.open('./image.jpg')  # Replace with your image path

# Display the image
st.image(image, use_column_width=True)

# Load the saved model with error handling
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model_1.pkl')
try:
    loaded_pickle_model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define the function to make predictions
def predict_outcome(age, fever, chills, fatigue, travel_history, mosquito_presence, temperature):
    input_data = pd.DataFrame({
        'age': [age],
        'fever': [fever],
        'chills': [chills],
        'fatigue': [fatigue],
        'travel_history': [travel_history],
        'mosquito_presence': [mosquito_presence],
        'temperature': [temperature]
    })
    prediction = loaded_pickle_model.predict(input_data)
    return prediction[0]

# Create the Streamlit app
def main():
    st.title("Disease Outcome Prediction")
    st.write("""
    This app predicts the likelihood of a disease outcome based on patient data.
    Please fill in the details below and click 'Predict' to see the result.
    """)
    
    # Add input fields for various features
    with st.form("prediction_form"):
        age = st.slider("Age", min_value=0, max_value=100, value=50)
        fever = st.radio("Fever", options=["No", "Yes"])
        chills = st.radio("Chills", options=["No", "Yes"])
        fatigue = st.radio("Fatigue", options=["No", "Yes"])
        travel_history = st.radio("Travel History", options=["No", "Yes"])
        mosquito_presence = st.radio("Mosquito Presence", options=["No", "Yes"])
        temperature = st.slider("Temperature (°C)", min_value=20.0, max_value=40.0, value=30.0)
        
        # Convert categorical inputs to numerical
        fever = 1 if fever == "Yes" else 0
        chills = 1 if chills == "Yes" else 0
        fatigue = 1 if fatigue == "Yes" else 0
        travel_history = 1 if travel_history == "Yes" else 0
        mosquito_presence = 1 if mosquito_presence == "Yes" else 0
        
        # Submit button
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            prediction = predict_outcome(age, fever, chills, fatigue, travel_history, mosquito_presence, temperature)
            
            # Display result with color background based on the prediction
            if prediction == 1:
                st.markdown(
                    '<div style="background-color: red; padding: 10px;">'
                    '<h2 style="color: white; text-align: center;">The patient is predicted to have the disease.</h2>'
                    '</div>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background-color: blue; padding: 10px;">'
                    '<h2 style="color: white; text-align: center;">The patient is predicted to not have the disease.</h2>'
                    '</div>', unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()