import streamlit as st

from backend import get_query_engine, init_index

if "query_engine" not in st.session_state.keys():
    st.session_state.query_engine = get_query_engine(
        indices=[
            # init_index(persist_dir="bitbucket_store"),
            # init_index(persist_dir="confluence_store"),
            init_index(persist_dir="local_store")
        ]
    )


st.set_page_config(
    page_title="Chat with the Streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat with Local Mate")
st.info("Alpha Version", icon="📃")


def get_response(query):
    with st.spinner(text="Thinking ..."):
        st.markdown(st.session_state.query_engine.query(query))


with st.form("my_form"):
    query = st.text_area("Enter Your Query:")
    submitted = st.form_submit_button("Ask")
    if submitted:
        get_response(query)
