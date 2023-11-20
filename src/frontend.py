import streamlit as st
from backend import get_query_engine, init_index

query_engine = get_query_engine(init_index())

st.set_page_config(
    page_title="Chat with the Streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with MXX Mate")
st.info("Alpha Version", icon="📃")


def get_response(query):
    with st.spinner(text="Thinking ..."):
        st.markdown(query_engine.query(query).response)


with st.form("my_form"):
    text = st.text_area("Enter Your Query:", "Who are you?")
    submitted = st.form_submit_button("Ask")
    if submitted:
        get_response(text)
