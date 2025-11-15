import os
import math
from collections import Counter, defaultdict

# Memilih VSM class jika ada
try:
    from .vsm_ir import VSMRetrieval
    VSM_AVAILABLE = True
except Exception:
    try:
        from vsm_ir import VSMRetrieval
        VSM_AVAILABLE = True
    except Exception:
        VSM_AVAILABLE = False


# Boolean IR utilities
def build_inverted_index(processed_dir: str):
    index: dict[str, set[str]] = defaultdict(set)
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                tokens = f.read().split()
            for t in set(tokens):
                index[t].add(fname)
    return index


def boolean_retrieve(query_tokens, index, op="AND"):
    sets = [index.get(t, set()) for t in query_tokens]
    if not sets:
        return []
    if op == "AND":
        result = set.intersection(*sets)
    else:
        result = set.union(*sets)
    return sorted(result)


# VSM helpers
def explain_top_terms_from_vsm(vsm_obj, doc_index: int, top_n: int = 4):
    try:
        import numpy as np

        vec = vsm_obj.tfidf_matrix[doc_index].toarray().flatten()
        features = vsm_obj.vectorizer.get_feature_names_out()
        top_idx = vec.argsort()[::-1][:top_n]
        return [(features[i], float(vec[i])) for i in top_idx if vec[i] > 0]
    except Exception:
        return []


def run_vsm_cli(processed_dir: str, query: str, k: int = 5, weight: str = "tfidf"):
    # Gunakan VSMRetrieval jika tersedia
    if VSM_AVAILABLE:
        vsm = VSMRetrieval(processed_dir=None)  # VSMRetrieval mengatur path sendiri
        vsm.load_processed_docs()
        vsm.build_tfidf()
        results = vsm.rank(query, k=k)
        explained = []
        for r in results:
            try:
                doc_idx = vsm.doc_ids.index(r["doc_id"])
                top_terms = explain_top_terms_from_vsm(vsm, doc_idx, top_n=4)
            except Exception:
                top_terms = []
            r["top_terms"] = top_terms
            explained.append(r)
        return explained

    # Fallback VSM manual jika VSMRetrieval tidak ada
    corpus = {}
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                corpus[fname] = f.read().split()

    N = len(corpus)
    df = Counter()
    for tokens in corpus.values():
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    q_tokens = query.lower().split()
    scored = []

    import numpy as np

    for fname, tokens in corpus.items():
        tf_q = Counter(q_tokens)
        tf_d = Counter(tokens)
        vocab = set(q_tokens) | set(tokens)
        qvec = []
        dvec = []

        for term in vocab:
            tq = tf_q.get(term, 0)
            td = tf_d.get(term, 0)
            if weight == "tfidf_sublinear":
                tq = 1 + math.log(tq) if tq > 0 else 0
                td = 1 + math.log(td) if td > 0 else 0
            idfv = idf.get(term, math.log((N + 1) / (1 + 1)))
            qvec.append(tq * idfv)
            dvec.append(td * idfv)

        qv = np.array(qvec)
        dv = np.array(dvec)
        denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
        score = float(np.dot(qv, dv) / denom) if denom > 0 else 0.0
        scored.append((fname, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for fname, score in scored[:k]:
        with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
            text = f.read()
        tokens = text.split()
        top_terms = Counter(tokens).most_common(4)
        snippet = " ".join(tokens[:25])
        results.append({
            "doc_id": fname,
            "score": score,
            "snippet": snippet,
            "top_terms": top_terms,
        })
    return results


# Wrapper yang dipanggil dari main.py
def search(processed_dir: str, model: str, query: str, k: int = 5, weight: str = "tfidf"):
    model = model.lower()
    if model == "boolean":
        index = build_inverted_index(processed_dir)
        qtokens = query.lower().split()
        res = boolean_retrieve(qtokens, index, op="AND")
        return [{"doc_id": r, "score": 1.0, "snippet": ""} for r in res[:k]]

    elif model == "bm25":
        # import lokal untuk menghindari dependensi siklik
        try:
            from .eval import build_bm25_index, rank_bm25  # type: ignore
        except Exception:
            from eval import build_bm25_index, rank_bm25  # type: ignore

        filenames = []
        corpus = []
        for fname in sorted(os.listdir(processed_dir)):
            if fname.endswith(".txt"):
                filenames.append(fname)
                with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                    corpus.append(f.read().split())

        bm25_index = build_bm25_index(corpus)
        scores = rank_bm25(query.lower().split(), corpus, bm25_index)
        paired = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for fname, score in paired:
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                tokens = f.read().split()
            snippet = " ".join(tokens[:25])
            results.append({
                "doc_id": fname,
                "score": float(score),
                "snippet": snippet,
                "top_terms": Counter(tokens).most_common(4),
            })
        return results

    else:
        # default: VSM
        return run_vsm_cli(processed_dir, query, k=k, weight=weight)
