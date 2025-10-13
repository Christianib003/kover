import streamlit as st


st.set_page_config(page_title="Kover", page_icon="🛡️")
st.title("Kover — Insurance Policy Chatbot (Setup)")
q = st.text_input("Ask about coverage, deductibles, claims:")
if st.button("Get Answer"):
    st.info("Model not trained yet. Come back after training.")
