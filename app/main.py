# Stable
# V1 -> RAG pipeline with Hybrid Retrival + Reranking + result evaluation
import streamlit as st
import os
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# -----------------------------
# CONFIG
# -----------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 5
RERANK_TOP_K = 3

# -----------------------------
# LOAD MODELS (CACHED)
# -----------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return embeddings, reranker

# -----------------------------
# PROCESS PDF
# -----------------------------
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    embeddings, _ = load_models()

    # FAISS
    db = FAISS.from_documents(chunks, embeddings)

    # BM25
    tokenized_corpus = [
        doc.page_content.split() for doc in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    return db, bm25, chunks

# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
def hybrid_retrieval(query, db, bm25, chunks):
    # Vector search
    vector_docs = db.similarity_search(query, k=TOP_K)

    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_idx = np.argsort(bm25_scores)[-TOP_K:]
    bm25_docs = [chunks[i] for i in bm25_top_idx]

    # Combine
    combined = vector_docs + bm25_docs

    # Remove duplicates
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs

# -----------------------------
# RERANKING
# -----------------------------
def rerank(query, docs, reranker):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:RERANK_TOP_K]]

# -----------------------------
# EVALUATION (LIGHTWEIGHT)
# -----------------------------
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_eval_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_answer(answer, context):
    model = load_eval_model()

    context_text = " ".join(context)

    similarity = util.cos_sim(
        model.encode(answer),
        model.encode(context_text)
    ).item()

    # Normalize (rough scaling)
    score = min(max(similarity, 0), 1)

    return round(score, 2)

# -----------------------------
# STREAMING RESPONSE
# -----------------------------
def stream_response(llm, prompt):
    response_placeholder = st.empty()
    full_response = ""

    for chunk in llm.stream(prompt):
        full_response += chunk
        response_placeholder.markdown(full_response + "▌")

    response_placeholder.markdown(full_response)
    return full_response

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

st.title("AI PDF Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    db, bm25, chunks = process_pdf("temp.pdf")
    embeddings, reranker = load_models()

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):

            # HYBRID RETRIEVAL
            docs = hybrid_retrieval(query, db, bm25, chunks)

            # RERANK
            docs = rerank(query, docs, reranker)

            # ADD CITATION TAGS
            context = ""
            for i, doc in enumerate(docs):
                context += f"[Source {i+1}]\n{doc.page_content}\n\n"

            prompt = f"""
You are an AI assistant.

Answer ONLY using the context.
Dont add any extra information that the context doesn't contain.
Cite sources using [Chunk X].
If unsure, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

            llm = Ollama(model="phi")

            st.subheader("Answer:")
            answer = stream_response(llm, prompt)

            # EVALUATION
            eval_score = evaluate_answer(
                answer,
                [doc.page_content for doc in docs]
            )

            st.metric("Answer Quality Score", round(eval_score, 2))

            st.session_state.history.append((query, answer))

        # SOURCES
        st.subheader("Sources")
        for i, doc in enumerate(docs):
            st.markdown(f"**[Source {i+1}]**")
            st.write(doc.page_content[:300] + "...")

# HISTORY
if st.session_state.history:
    st.subheader("💬 Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")