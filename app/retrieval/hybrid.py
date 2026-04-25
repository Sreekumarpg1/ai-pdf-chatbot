import numpy as np
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from app.config import TOP_K


def build_indices(chunks, embeddings):
    # FAISS
    db = FAISS.from_documents(chunks, embeddings)

    # BM25
    tokenized = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)

    return db, bm25


def hybrid_retrieval(query, db, bm25, chunks):
    vector_docs = db.similarity_search(query, k=TOP_K)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_idx = np.argsort(scores)[-TOP_K:]
    bm25_docs = [chunks[i] for i in top_idx]

    combined = vector_docs + bm25_docs
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs