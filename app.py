import streamlit as st
import pandas as pd
import numpy as np
import pickle


LABEL_MAPPING = {
    0: "ADHD Negative",
    1: "ADHD Positive"
}

@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model

# load the model
ADHD_model = load_sklearn_models("best_model_rf")


#title of the web page
st.title("ADHD Predicting WebApp")

IMAGE_ADDRESS = "https://media.npr.org/assets/img/2024/05/22/gettyimages-1805884626_custom-12456c773bb5d72837b1b3534ff4e90e4f68a983.jpg?s=1100&c=85&f=jpeg"
# set the image
st.image(IMAGE_ADDRESS)

st.subheader("Please enter the details:")

#input for Gender
Gender=st.selectbox("Choose a gender",("Male","Female"),)
if Gender=="Male":
    G=0

else:
    G=1
#st.write("You selected: ",Gender)

#input for age
Age=st.slider("What is the age", 0, 30, 10)
#st.write("Age is", Age, "years old")

#Input for Handedness
Handedness=st.selectbox("Please select the dominant hand",("Right-Handed","Left-Handed"))
if Handedness=="Right-Handed":
    H=0
else:
    H=1
#st.write("You selected:", "Left-Handed")


# Input for Inattentive score 
Inattentive = st.slider("Rate the level of inattentiveness (0-100)", 0, 100, 50)
#st.write(f"Inattentiveness score: {Inattentive}")



# Input for Impulsive score 
Impulsive = st.slider("Rate the level of impulsiveness (0-100)", 0, 100, 50)
#st.write(f"Impulsiveness score: {Impulsive}")

# IQ Measure input
IQ_Measure=st.slider("Select IQ Measure (1-5)", 1,5,3) 
#st.write(f"Selected IQ Measure: {IQ_Measure}")

# Verbal IQ input
Verbal_IQ = st.slider("Enter the  Verbal IQ", 0, 200, 100)
#st.write(f"Verbal IQ score: {Verbal_IQ}")

# Performance IQ input
Performance_IQ = st.slider("Enter your Performance IQ", 0, 200, 100)
#st.write(f"Performance IQ score: {Performance_IQ}")

# Full4 IQ input
Full4_IQ = st.slider("Enter your Full-Scale IQ (Full4 IQ)", 0, 200, 100)
#st.write(f"Full scale IQ score: {Full4_IQ}")

# Med Status input
Med_Status = st.radio("Are you on medication?", ("Yes", "No"))
if Med_Status=='Yes':
    M=1
else:
    M=2
#st.write(f"Medication Status: {Med_Status}")



#make Predictions

if st.button('Predict ADHD'):
    input_data=[[G,Age,H,Inattentive,Impulsive,IQ_Measure,Verbal_IQ,Performance_IQ,Full4_IQ,M]]
    predictions=ADHD_model.predict(input_data)
    st.spinner(text="In progress...")
    st.subheader("User Condition: {}".format(LABEL_MAPPING[(predictions[0])]))


