import os
import sys
import nltk

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import preprocess_directory, preprocess_text
from src.boolean_ir import build_incidence_matrix, build_inverted_index, parse_boolean_query
from src.vsm_ir import (
    load_processed_docs,
    build_tfidf,
    rank,
    precision_at_k,
    average_precision,
    ndcg_at_k,
)
from src.search_engine import search as engine_search
from src.eval import evaluate_all, plot_results

# NLTK
nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
nltk.download("stopwords", quiet=True)

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ulang Preprocess
def re_preprocess():
    print("\n=== MODE PREPROCESS ULANG ===")
    preprocess_directory(DATA_DIR, PROCESSED_DIR)

# BOOLEAN IR
def demo_boolean():
    print("\n=== MODE BOOLEAN IR ===")
    processed_docs = []
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(PROCESSED_DIR, filename)
            with open(path, encoding="utf-8") as f:
                processed_docs.append(f.read().strip())
    incidence_matrix, vocabulary = build_incidence_matrix(processed_docs)
    inverted_index = build_inverted_index(processed_docs)

    q_bool = input("Masukkan Boolean query (contoh: 'cardiovascular AND knn'): ")
    res_bool = parse_boolean_query(q_bool, inverted_index)
    print("\n=== HASIL BOOLEAN IR ===")
    print("Query  :", q_bool)
    print("Dokumen:", res_bool)

# VSM
def demo_vsm():
    print("\n=== MODE VSM ===")
    docs, doc_ids = load_processed_docs(PROCESSED_DIR)
    vectorizer, tfidf_matrix = build_tfidf(docs)

    q_vsm_raw = input("\nMasukkan query VSM: ")
    q_vsm = preprocess_text(q_vsm_raw)
    top_results = rank(q_vsm, docs, doc_ids, vectorizer, tfidf_matrix, k=5)

    print("\n=== TOP-5 VSM RANKING ===")
    for r in top_results:
        print(f"{r['doc_id']} | {r['score']:.4f} | {r['snippet']}")

    # contoh gold kecil (silakan ganti sesuai tugasmu)
    gold = ["137485529.techno.com.txt", "88974145.techno.com.txt"]
    retrieved = [r["doc_id"] for r in top_results]

    print("\n=== METRIK EVAL ===")
    print("Precision@5:", precision_at_k(retrieved, gold, 5))
    print("MAP@5      :", average_precision(retrieved, gold, 5))
    print("nDCG@5     :", ndcg_at_k(retrieved, gold, 5))

# SEARCH ENGINE (BOOLEAN / VSM / BM25)
def interactive_search():
    print("\n=== MODE SEARCH ENGINE ===")
    print(f"[INFO] Folder processed: {PROCESSED_DIR}")

    while True:
        print("\nPilih model:")
        print("  1. Boolean")
        print("  2. VSM (TF-IDF)")
        print("  3. BM25")
        print("  0. Kembali ke menu utama")
        m = input("Pilihan [0-3]: ").strip()

        if m == "0":
            break
        elif m == "1":
            model = "boolean"
        elif m == "2":
            model = "vsm"
        elif m == "3":
            model = "bm25"
        else:
            print("Pilihan tidak dikenal.")
            continue

        query = input("Masukkan query (kosong untuk kembali): ").strip()
        if not query:
            continue

        try:
            results = engine_search(PROCESSED_DIR, model=model, query=query, k=5, weight="tfidf")
        except Exception as e:
            print(f"[ERROR] Gagal menjalankan search: {e}")
            continue

        print(f"\n🔍 HASIL PENCARIAN (model={model}) untuk: '{query}'\n")
        if not results:
            print("Tidak ada dokumen cocok.")
            continue

        for i, r in enumerate(results, 1):
            print(f"{i}. {r['doc_id']:<35} score={r['score']:.4f}")
            top_terms = r.get("top_terms") or []
            if top_terms:
                parts = []
                for t in top_terms:
                    if isinstance(t, str):
                        parts.append(t)
                    elif isinstance(t, (list, tuple)) and len(t) >= 2:
                        try:
                            parts.append(f"{t[0]}({float(t[1]):.3f})")
                        except Exception:
                            parts.append(str(t[0]))
                    else:
                        parts.append(str(t))
                print("    top_terms:", ", ".join(parts))
            snippet = r.get("snippet") or ""
            if snippet:
                print("    snippet  :", snippet[:160], "...")
            print()

# EVALUATION (Dengan gold.json)
def interactive_eval():
    print("\n=== MODE EVALUATION ===")
    gold_path = os.path.join(DATA_DIR, "gold.json")

    if not os.path.exists(gold_path):
        print(f"[ERROR] gold.json tidak ditemukan di: {gold_path}")
        print("Bikin dulu file gold.json (mapping query -> daftar dokumen relevan).")
        return

    try:
        results = evaluate_all(PROCESSED_DIR, gold_path, k=5)
    except Exception as e:
        print(f"[ERROR] Gagal evaluasi: {e}")
        return

    print("\n=== HASIL EVALUATION (SEMUA MODEL) ===")
    for m, v in results.items():
        print(m, v)

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_img = os.path.join(reports_dir, "metrics_comparison.png")
    try:
        plot_results(results, out_path=out_img)
        print(f"\nPlot disimpan ke: {out_img}")
    except Exception as e:
        print(f"[WARN] Plot gagal dibuat: {e}")


# MAIN MENU
def main():
    while True:
        print("\nSabrina Aska Amalina - A11.2023.15264")
        print("====================================")
        print("       MINI PROJECT STKI UTS       ")
        print("====================================")
        print("1. Ulang Preprocessing")
        print("2. Demo Boolean")
        print("3. VSM")
        print("4. Search Engine (Boolean / VSM / BM25)")
        print("5. Evaluation pakai gold.json")
        print("0. Keluar")
        choice = input("Pilih [0-5]: ").strip()

        if choice == "1":
            re_preprocess()
        elif choice == "2":
            demo_boolean()
        elif choice == "3":
            demo_vsm()
        elif choice == "4":
            interactive_search()
        elif choice == "5":
            interactive_eval()
        elif choice == "0":
            print("Selamat tinggal! 🤗")
            break
        else:
            print("Pilihan tidak dikenal, coba lagi.")


if __name__ == "__main__":
    main()
