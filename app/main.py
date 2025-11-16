import os
import sys
import nltk

# Setup path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import preprocess_directory, preprocess_text
from src.boolean_ir import (build_incidence_matrix,
    build_inverted_index,
    parse_boolean_query,
    explain_boolean_query,
    precision_recall)
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
    # 1. Load dokumen ter-preprocess
    processed_docs = []
    doc_ids = []  # simpan nama file supaya enak dibaca
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(PROCESSED_DIR, filename)
            with open(path, encoding="utf-8") as f:
                processed_docs.append(f.read().strip())
            doc_ids.append(filename)

    if not processed_docs:
        print("[ERROR] Tidak ada dokumen di data/processed/. Jalankan preprocessing dulu.")
        return
    
    # 2. Bangun incidence matrix & inverted index
    incidence_matrix, vocabulary = build_incidence_matrix(processed_docs)
    inverted_index = build_inverted_index(processed_docs)

    print(f"Jumlah dokumen   : {len(processed_docs)}")
    print(f"Ukuran vocabulary: {len(vocabulary)}")
    print("Index sudah terbentuk (incidence matrix & inverted index).")
    print("Masukkan Boolean query (AND / OR / NOT).")
    print("Ketik 'exit', 'quit', atau 'keluar' untuk kembali ke menu.\n")

    # 3. Loop untuk ≥ 3 query (user bisa mencoba berkali-kali)
    while True:
        q_bool = input("Masukkan Boolean query: ").strip()

        # === TAMBAHAN FITUR KELUAR ===
        if q_bool.lower() in ["exit", "quit", "keluar"]:
            print("Keluar dari mode Boolean IR...\n")
            break
        # =================================

        if not q_bool:
            break

        # 4. Jalankan Boolean IR + explanation
        result_docs, steps = explain_boolean_query(q_bool, inverted_index)

        print("\n=== HASIL BOOLEAN IR ===")
        print("Query :", q_bool)

        if not result_docs:
            print("-> Tidak ada dokumen yang cocok.")
        else:
            print("-> Daftar dokumen (id & nama file):")
            for d in sorted(result_docs):
                if 0 <= d < len(doc_ids):
                    print(f"   - doc#{d} : {doc_ids[d]}")
                else:
                    print(f"   - doc#{d}")

        # 5. Tampilkan penjelasan interseksi / union / komplemen
        print("\n=== PENJELASAN LANGKAH BOOLEAN ===")
        for i, stp in enumerate(steps, start=1):
            term = stp["term"]
            op = stp["op"]
            used_not = stp["used_not"]
            raw_docs = stp["raw_docs"]
            docs_after_not = stp["docs_after_not"]
            prev_result = stp["prev_result"]
            new_result = stp["new_result"]

            print(f"Langkah {i}:")
            print(f"  Term      : '{term}'")
            if used_not:
                print(f"  NOT       : yes (komplemen dari {raw_docs} -> {docs_after_not})")
            else:
                print(f"  Postings  : {raw_docs}")
            if op == "and":
                print("  Operasi   : AND (interseksi) dengan result sebelumnya")
            elif op == "or":
                print("  Operasi   : OR  (union) dengan result sebelumnya")
            else:
                print("  Operasi   : INIT (result awal)")
            if prev_result is not None:
                print(f"  Sebelum   : {prev_result}")
            print(f"  Sesudah   : {new_result}")
            print()

        # 6. Mini truth set: precision & recall
        ans = input("Ingin hitung precision/recall untuk query ini? [y/n]: ").strip().lower()
        if ans == "y":
            print("Masukkan gold relevant docs sebagai index dokumen, pisahkan dengan koma.")
            print("Contoh: 0, 3, 5  (lihat nomor doc# di daftar hasil di atas)")
            gold_str = input("Gold relevant docs: ").strip()
            if gold_str:
                try:
                    gold_ids = [int(x.strip()) for x in gold_str.split(",") if x.strip() != ""]
                    p, r = precision_recall(result_docs, gold_ids)
                    print(f"Precision : {p:.3f}")
                    print(f"Recall    : {r:.3f}")
                except ValueError:
                    print("[WARN] Format gold docs tidak valid, lewati perhitungan.")


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
