import pickle
import numpy as np
import streamlit as st
import re
from nltk.stem import PorterStemmer

model = pickle.load(open('model.pkl', 'rb'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    words = text.split()
    
    # Initialize the stemmer
    stemmer = PorterStemmer()
    
    # Stem the words
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # Join the stemmed words back into a single string
    processed_text = ' '.join(stemmed_words)
    
    return processed_text

st.title("💼 Job Posting Fraud Detection")
st.write("🔍 Enter job details to detect if it's Real or Fraudulent.")

# Input fields
title = st.text_input("Job Title")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")
company_profile = st.text_area("Company Profile")


# When button is clicked
if st.button("Predict"):
    # Combine text features
    combined_text = f"{title} {description} {requirements} {company_profile}"

    # Preprocess
    processed = preprocess_text(combined_text)

    # Vectorize — You need to use the same vectorizer you used in training
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Make sure you saved your TfidfVectorizer
    input_vector = vectorizer.transform([processed])

    # Predict
    result = model.predict(input_vector)[0]
    prob = model.predict_proba(input_vector)[0][result]
    ''
    
    # Output
    if result == 1:
        st.error(f"⚠️ This job posting is likely **Fraudulent** (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ This job posting is **Legit/Real** (Confidence: {prob:.2f})")
    st.write("🔍 **Note:** Always verify job postings through official channels.")
    

