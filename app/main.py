import streamlit as st

from langchain_community.llms import Ollama

from app.ingestion.pdf_loader import load_and_split_pdf
from app.models.model_loader import (
    load_embedding_model,
    load_reranker,
    load_eval_model
)
from app.retrieval.hybrid import build_indices, hybrid_retrieval
from app.reranking.cross_encoder import rerank
from app.evaluation.evaluator import evaluate_answer
from app.utils.streaming import stream_response

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("AI PDF Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    chunks = load_and_split_pdf("temp.pdf")

    embeddings = load_embedding_model()
    reranker = load_reranker()
    eval_model = load_eval_model()

    db, bm25 = build_indices(chunks, embeddings)

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):

            docs = hybrid_retrieval(query, db, bm25, chunks)
            docs = rerank(query, docs, reranker)

            context = ""
            for i, doc in enumerate(docs):
                context += f"[Source {i+1}]\n{doc.page_content}\n\n"

            prompt = f"""
You are an AI assistant.

Answer ONLY using the context.
Do not hallucinate.
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

            score = evaluate_answer(
                answer,
                [doc.page_content for doc in docs],
                eval_model
            )

            st.metric("Answer Quality Score", score)

            st.session_state.history.append((query, answer))

        st.subheader("Sources")
        for i, doc in enumerate(docs):
            st.markdown(f"**[Source {i+1}]**")
            st.write(doc.page_content[:300] + "...")

if st.session_state.history:
    st.subheader("Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")