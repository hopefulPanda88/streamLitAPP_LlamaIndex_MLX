import streamlit as st

from utils import LocalLLMOnMLX


@st.cache_data
def load_llm(path: str):
    llm = LocalLLMOnMLX(path, path)
    return llm
