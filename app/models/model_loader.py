import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_eval_model():
    return SentenceTransformer("all-MiniLM-L6-v2")