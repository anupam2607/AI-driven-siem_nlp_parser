# src/rag_retriever.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_tfidf = None
_matrix = None
_docs = None

def build_tfidf_index(docs):
    """
    docs: list of text strings
    """
    global _tfidf, _matrix, _docs
    _tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    _matrix = _tfidf.fit_transform(docs)
    _docs = docs

def rag_retrieve(query: str, docs: list, top_k: int = 5):
    """
    Returns list of (score, doc_text, idx)
    """
    global _tfidf, _matrix, _docs
    if _tfidf is None or _matrix is None or _docs is None:
        build_tfidf_index(docs)
    q_vec = _tfidf.transform([query])
    sims = cosine_similarity(q_vec, _matrix).flatten()
    idxs = np.argsort(-sims)[:top_k]
    results = [{"score": float(sims[i]), "text": docs[i], "idx": int(i)} for i in idxs]
    return results
