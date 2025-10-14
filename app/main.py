import streamlit as st
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_USE_XLA"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.chatbot import Chatbot

from src.chatbot import Chatbot

st.set_page_config(page_title="Kover", page_icon="🛡️")
st.title("Kover — Insurance Policy Chatbot")

@st.cache_resource
def load_bot():
    return Chatbot("results/checkpoints/baseline")

bot = load_bot()
q = st.text_input("Ask about coverage, deductibles, coinsurance, claims:")
max_new = st.slider("Max new tokens", 32, 256, 160, 16)

if st.button("Get Answer"):
    bot.max_new_tokens = max_new
    st.success(bot.get_response(q))
