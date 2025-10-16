# app/main.py

import streamlit as st
import sys
import os

# Add the 'src' directory to the Python path to allow importing the Chatbot class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from chat import Chatbot

# --- Configuration ---
MODEL_DIR = "models/lr_experiment_model" # Our champion model!
KEYWORDS_FILE = "data/processed/keywords.txt"

# --- App Title ---
st.title("🤖 Insurance Policy QA Chatbot")
st.markdown("This chatbot is powered by a fine-tuned T5 model and can answer your questions about insurance policies.")

# --- Model Loading ---
# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_chatbot():
    """Loads the chatbot model and caches it."""
    return Chatbot(model_dir=MODEL_DIR, keywords_path=KEYWORDS_FILE)

chatbot = load_chatbot()

# --- Chat Interface ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about insurance..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and display chatbot response
    with st.spinner("Thinking..."):
        response = chatbot.get_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})