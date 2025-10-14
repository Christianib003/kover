import os, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.chatbot import Chatbot

st.set_page_config(page_title="Kover", page_icon="🛡️")
st.title("Kover — Insurance Policy Chatbot")

def list_runs(base: str = "results/checkpoints"):
    if not os.path.isdir(base): return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

runs = list_runs()
if not runs:
    st.info("No trained runs found in results/checkpoints/. Train a model first.")
    st.stop()

choice = st.selectbox("Model run", options=runs, index=max(0, len(runs)-1))
max_new = st.slider("Max new tokens", 32, 256, 160, 16)

@st.cache_resource
def load_bot(run_dir: str):
    return Chatbot(os.path.join("results/checkpoints", run_dir))

bot = load_bot(choice)

q = st.text_input("Ask about coverage, deductibles, coinsurance, claims:")
if st.button("Get Answer"):
    bot.max_new_tokens = max_new
    st.success(bot.get_response(q))
