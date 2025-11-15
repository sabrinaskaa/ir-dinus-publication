import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed docs
def load_processed_docs(processed_dir):
    import os
    docs = []
    doc_ids = []

    files = sorted(os.listdir(processed_dir))
    for fname in files:
        if fname.endswith(".txt"):
            with open(os.path.join(processed_dir, fname), "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append(text)
            doc_ids.append(fname)

    print(f"\n[INFO] Loaded {len(docs)} documents.")
    return docs, doc_ids


# Membuat TF-IDF Matrix
def build_tfidf(docs):
    if not docs:
        raise ValueError("Dokumen belum dimuat.")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    print(f"[INFO] TF-IDF shape: {tfidf_matrix.shape} (docs x terms)")
    return vectorizer, tfidf_matrix


# Query → TF-IDF Vector
def vectorize_query(query, vectorizer):
    if isinstance(query, list):
        query = " ".join(query)
    query = query.lower().strip()
    return vectorizer.transform([query])


# Cosine similarity ranking
def rank(query, docs, doc_ids, vectorizer, tfidf_matrix, k=5):
    q_vec = vectorize_query(query, vectorizer)
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_idx:
        snippet = docs[idx][:120].replace("\n", " ")
        results.append({
            "doc_id": doc_ids[idx],
            "score": float(scores[idx]),
            "snippet": snippet
        })
    return results

# Precision @ k
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    rel = set(relevant)
    hit = sum(1 for d in retrieved_k if d in rel)
    return hit / k

# Average Precision (untuk MAP)
def average_precision(retrieved, relevant, k):
    rel = set(relevant)
    score = 0.0
    hit = 0

    for i in range(min(k, len(retrieved))):
        if retrieved[i] in rel:
            hit += 1
            score += hit / (i + 1)

    if len(relevant) == 0:
        return 0.0

    return score / len(relevant)

# Menghitung nDCG @ k
def ndcg_at_k(retrieved, relevant, k):
    rel = set(relevant)
    dcg = 0.0

    for i in range(min(k, len(retrieved))):
        if retrieved[i] in rel:
            dcg += 1.0 / np.log2(i + 2)

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg