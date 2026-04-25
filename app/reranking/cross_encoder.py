from app.config import RERANK_TOP_K


def rerank(query, docs, reranker):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:RERANK_TOP_K]]