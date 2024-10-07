import streamlit as st
import pandas as pd
import numpy as np
import pickle


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model

# load the model
ADHD_model = load_sklearn_models("best_model_rf")

#title of the web page
st.title("ADHD Predicting WebApp")
