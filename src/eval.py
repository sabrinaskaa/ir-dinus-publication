import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Load processed corpus
def load_corpus_processed(processed_dir):
    filenames = []
    corpus = []
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith(".txt"):
            filenames.append(fname)
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                corpus.append(f.read().split())
    return filenames, corpus

# IDF computation
def compute_df_idf(corpus):
    N = len(corpus)
    df = Counter()
    for doc in corpus:
        for t in set(doc):
            df[t] += 1
    idf = {}
    for t, dfv in df.items():
        idf[t] = math.log((N + 1) / (dfv + 1)) + 1
    return df, idf

# TF-IDF vectors
def build_vocab(corpus):
    vocab = sorted(set(w for doc in corpus for w in doc))
    idx = {w:i for i,w in enumerate(vocab)}
    return vocab, idx

def tfidf_vector(doc_tokens, idf, idx, sublinear=False):
    vec = np.zeros(len(idx), dtype=float)
    tf = Counter(doc_tokens)
    for term, cnt in tf.items():
        if term in idx:
            if sublinear:
                tfv = 1.0 + math.log(cnt)
            else:
                tfv = cnt
            # normalize by doc length (common choice)
            tfv = tfv / len(doc_tokens)
            vec[idx[term]] = tfv * idf.get(term, 0.0)
    return vec

# BM25 Helper
def build_bm25_index(corpus):
    N = len(corpus)
    df = Counter()
    for doc in corpus:
        for t in set(doc):
            df[t] += 1
    avgdl = sum(len(d) for d in corpus) / max(1, N)
    return {"N":N, "df":df, "avgdl":avgdl}

def bm25_score(query_tokens, doc_tokens, bm25_index, k1=1.5, b=0.75):
    N = bm25_index["N"]
    df = bm25_index["df"]
    avgdl = bm25_index["avgdl"]
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    s = 0.0
    for t in query_tokens:
        f = tf.get(t, 0)
        df_t = df.get(t, 0)
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-9)
        denom = f + k1 * (1 - b + b * dl / avgdl)
        if denom > 0:
            s += idf * (f * (k1 + 1)) / denom
    return s

# Similarity
def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

# Ranking wrappers
def rank_tfidf(query_tokens, corpus, idf, idx, sublinear=False):
    qvec = tfidf_vector(query_tokens, idf, idx, sublinear=sublinear)
    doc_vecs = [tfidf_vector(doc, idf, idx, sublinear=sublinear) for doc in corpus]
    scores = [cosine_sim(qvec, dv) for dv in doc_vecs]
    return scores

def rank_bm25(query_tokens, corpus, bm25_index):
    return [bm25_score(query_tokens, doc, bm25_index) for doc in corpus]

# Evaluation metrics
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hit = sum(1 for r in retrieved_k if r in relevant)
    return hit / k

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    hit = sum(1 for r in retrieved[:k] if r in relevant)
    return hit / len(relevant)

def apk(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    score = 0.0
    hits = 0
    for i, d in enumerate(retrieved_k, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0

# Evaluation accross gold queries
def evaluate_all(processed_dir, gold_path, k=5):
    filenames, corpus = load_corpus_processed(processed_dir)
    df, idf = compute_df_idf(corpus)
    vocab, idx = build_vocab(corpus)
    bm25_index = build_bm25_index(corpus)

    # load gold
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    models = ["tfidf", "tfidf_sublinear", "bm25"]
    agg = {m: {"P":[], "R":[], "AP":[]} for m in models}

    for query, relevant in gold.items():
        q_tokens = query.lower().split()
        # TF-IDF
        scores_tfidf = rank_tfidf(q_tokens, corpus, idf, idx, sublinear=False)
        ranked_tfidf = [filenames[i] for i in np.argsort(scores_tfidf)[::-1]]
        scores_sub = rank_tfidf(q_tokens, corpus, idf, idx, sublinear=True)
        ranked_sub = [filenames[i] for i in np.argsort(scores_sub)[::-1]]
        scores_b = rank_bm25(q_tokens, corpus, bm25_index)
        ranked_b = [filenames[i] for i in np.argsort(scores_b)[::-1]]

        for name, ranked in [("tfidf", ranked_tfidf), ("tfidf_sublinear", ranked_sub), ("bm25", ranked_b)]:
            p = precision_at_k(ranked, relevant, k)
            r = recall_at_k(ranked, relevant, k)
            ap = apk(ranked, relevant, k)
            agg[name]["P"].append(p)
            agg[name]["R"].append(r)
            agg[name]["AP"].append(ap)

    results = {}
    for m in models:
        P = float(np.mean(agg[m]["P"])) if agg[m]["P"] else 0.0
        R = float(np.mean(agg[m]["R"])) if agg[m]["R"] else 0.0
        MAP = float(np.mean(agg[m]["AP"])) if agg[m]["AP"] else 0.0
        F1 = (2*P*R/(P+R)) if (P+R)>0 else 0.0
        results[m] = {"Precision@k":P, "Recall@k":R, "F1":F1, "MAP@k":MAP}

    return results

# Plotting helper
def plot_results(results, out_path=None):
    labels = ["Precision@k", "Recall@k", "F1", "MAP@k"]
    models = list(results.keys())
    vals = [[results[m][lab] for lab in labels] for m in models]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9,5))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models)-1)/2)*width, vals[i], width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title("Perbandingan metrik antar skema pembobotan")
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.show()
