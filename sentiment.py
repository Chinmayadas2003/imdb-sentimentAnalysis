# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import string

# ------------------------------
# Function to Load the Keras Model
# ------------------------------
@st.cache_resource
def load_sentiment_model(model_path):
    model = load_model(model_path)
    return model

# ------------------------------
# Function to Load IMDb Word Index
# ------------------------------
@st.cache_resource
def load_imdb_word_index():
    word_index = imdb.get_word_index()
    # Keras reserves indices for special tokens
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    return word_index

# ------------------------------
# Function to Convert Text to Sequence
# ------------------------------
def text_to_sequence(text, word_index, max_length=200, vocab_size=10000):
    # Clean the text: remove punctuation and make lowercase
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    words = clean_text.split()

    sequence = []
    for word in words:
        index = word_index.get(word, 2)  # 2 is the index for <UNK>
        if index >= vocab_size:
            index = 2  # Map to <UNK> if index exceeds vocab_size
        sequence.append(index)

    # Pad the sequence
    padded = pad_sequences([sequence], maxlen=max_length, padding='post', truncating='post')
    return padded

# ------------------------------
# Function to One-Hot Encode Sequences
# ------------------------------
def oh_representations(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequence in enumerate(sequences):
        # Ensure all indices are within [0, dim-1]
        sequence = [index if index < dim else 2 for index in sequence]
        results[i, sequence] = 1
    return results

# ------------------------------
# Load the Model and Word Index
# ------------------------------
MODEL_PATH = 'sentiment_model.h5'

model = load_sentiment_model(MODEL_PATH)
word_index = load_imdb_word_index()

MAX_SEQUENCE_LENGTH = 200  # Ensure this matches your training
VOCAB_SIZE = 10000

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("📽️ IMDb Movie Review Sentiment Classifier")
st.write("Enter a movie review below, and the model will classify the sentiment as **Positive** or **Negative**.")

# Text input from the user
user_input = st.text_area("📝 Enter your movie review here:", height=200)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review to get the sentiment prediction.")
    else:
        # Preprocess the input
        processed_input = text_to_sequence(user_input, word_index, max_length=MAX_SEQUENCE_LENGTH, vocab_size=VOCAB_SIZE)

        # One-Hot Encode the input using the same function as training
        one_hot_input = oh_representations(processed_input, dim=VOCAB_SIZE)

        # Make prediction
        prediction = model.predict(one_hot_input)
        confidence = prediction[0][0]

        # Determine sentiment
        if confidence >= 0.5:
            sentiment = "✅ **Positive**"
        else:
            sentiment = "❌ **Negative**"

        # Display results
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
