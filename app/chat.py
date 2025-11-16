import os
import sys
import json
from typing import List, Dict, Any, Tuple

import streamlit as st
import nltk

# ========== PATH & NLTK SETUP ==========

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# NLTK seperti di main.py CLI
nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
nltk.download("stopwords", quiet=True)

# import modul dari src/ (sama seperti main.py)
from src.preprocess import preprocess_directory, preprocess_text  # type: ignore
from src.boolean_ir import (  # type: ignore
    build_incidence_matrix,
    build_inverted_index,
    parse_boolean_query,
    explain_boolean_query,
    precision_recall,
)
from src.vsm_ir import (  # type: ignore
    load_processed_docs,
    build_tfidf,
    rank,
    precision_at_k,
    average_precision,
    ndcg_at_k,
)
from src.search_engine import search as engine_search  # type: ignore
from src.eval import evaluate_all, plot_results  # type: ignore

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GOLD_PATH = os.path.join(DATA_DIR, "gold.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


# ========== HELPER UMUM ==========

def get_doc_counts() -> Tuple[int, int]:
    raw_docs = (
        [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
        if os.path.isdir(DATA_DIR)
        else []
    )
    processed_docs = []
    if os.path.isdir(PROCESSED_DIR):
        processed_docs = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".txt")]
    return len(raw_docs), len(processed_docs)


def get_processed_token_stats(limit: int | None = None):
    if not os.path.isdir(PROCESSED_DIR):
        return []

    stats = []
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(PROCESSED_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            n_tokens = len(text.split())
            stats.append({"file": fname, "tokens": n_tokens})
        except Exception:
            stats.append({"file": fname, "tokens": -1})

    if limit is not None:
        stats = stats[:limit]
    return stats


def load_gold(path: str = GOLD_PATH) -> Dict[str, List[str]]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_original_doc(doc_id: str) -> str:
    raw_path = os.path.join(DATA_DIR, doc_id)
    if not os.path.isfile(raw_path):
        raw_path = os.path.join(PROCESSED_DIR, doc_id)
        if not os.path.isfile(raw_path):
            return "(File dokumen tidak ditemukan)"
    with open(raw_path, encoding="utf-8") as f:
        return f.read()


def run_search(query: str, model_key: str, k: int = 10) -> List[Dict[str, Any]]:
    try:
        results = engine_search(
            PROCESSED_DIR,
            model=model_key,   # "vsm", "bm25", atau "boolean"
            query=query,
            k=k,
            weight="tfidf",
        )
        return results
    except Exception as e:
        st.error(f"Gagal menjalankan pencarian: {e}")
        return []


# ========== CACHE UNTUK STRUKTUR BOOLEAN & VSM ==========

@st.cache_data(show_spinner=False)
def load_processed_texts_for_boolean() -> Tuple[List[str], List[str]]:
    docs: List[str] = []
    doc_ids: List[str] = []
    if not os.path.isdir(PROCESSED_DIR):
        return docs, doc_ids

    for filename in sorted(os.listdir(PROCESSED_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, encoding="utf-8") as f:
            docs.append(f.read().strip())
        doc_ids.append(filename)
    return docs, doc_ids


@st.cache_data(show_spinner=False)
def prepare_boolean_structures():
    docs, doc_ids = load_processed_texts_for_boolean()
    if not docs:
        return doc_ids, docs, None, None, {}
    incidence_matrix, vocabulary = build_incidence_matrix(docs)
    inverted_index = build_inverted_index(docs)
    return doc_ids, docs, incidence_matrix, vocabulary, inverted_index


@st.cache_data(show_spinner=False)
def prepare_vsm_structures():
    """Siapkan TF-IDF untuk VSM (Vector Space Model)."""
    if not os.path.isdir(PROCESSED_DIR):
        return [], [], None, None
    docs, doc_ids = load_processed_docs(PROCESSED_DIR)
    if not docs:
        return [], [], None, None
    vectorizer, tfidf_matrix = build_tfidf(docs)
    return docs, doc_ids, vectorizer, tfidf_matrix


# ========== PAGE: PREPROCESSING (ULANG PREPROCESS) ==========

def page_preprocess():
    st.header("⚙️ Ulang Preprocessing")

    if not os.path.isdir(DATA_DIR):
        st.error("Folder `data/` tidak ditemukan di project.")
        return

    raw_count, processed_count = get_doc_counts()

    col_a, col_b = st.columns(2)
    col_a.metric("Jumlah dokumen mentah (data/*.txt)", raw_count)
    col_b.metric("Jumlah dokumen ter-preprocess (data/processed/*.txt)", processed_count)

    st.write(
        "Preprocessing meliputi: *cleaning* (case folding, hapus angka & tanda baca), "
        "tokenisasi, penghapusan stopword Bahasa Indonesia, dan stemming Sastrawi.\n\n"
        "Hasil preprocessing dipakai oleh **Boolean Retrieval Model**, "
        "**Vector Space Model (VSM & Ranking)**, dan **BM25**."
    )

    if st.button("🔁 Jalankan Ulang Preprocessing (re_preprocess)"):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with st.spinner("Sedang melakukan preprocessing semua dokumen di folder data/..."):
            try:
                preprocess_directory(DATA_DIR, PROCESSED_DIR)
            except Exception as e:
                st.error(f"Terjadi error saat preprocessing: {e}")
                return

        raw_count, processed_count = get_doc_counts()
        st.success(
            f"Preprocessing selesai. Sekarang ada {processed_count} dokumen di `data/processed/`."
        )
        st.cache_data.clear()  # supaya struktur boolean/VSM di-refresh

    # Ringkasan dokumen preprocessed + jumlah tokens (seperti log [OK] ...)
    st.markdown("---")
    st.subheader("Ringkasan Dokumen Ter-preprocess (contoh)")

    stats = get_processed_token_stats(limit=200)
    if not stats:
        st.write("Belum ada dokumen di `data/processed/`.")
    else:
        st.caption(
            "Format: `[OK] nama_file.txt → N tokens` "
            "(menghitung token dari hasil preprocessing)."
        )
        for s in stats:
            tokens = s["tokens"]
            if tokens >= 0:
                st.text(f"[OK] {s['file']} \u2192 {tokens} tokens")
            else:
                st.text(f"[OK] {s['file']} \u2192 (gagal menghitung tokens)")

    st.info(
        "💡 **Tip:** Setelah preprocessing, kamu bisa langsung coba demo "
        "_Boolean IR_, _VSM & Ranking_, atau _Search Engine_ di tab lain."
    )


# ========== PAGE: DEMO BOOLEAN IR (DISAMAKAN DENGAN main.py) ==========

def page_boolean_demo():
    st.header("🧮 Boolean Retrieval Model Demo")

    if not os.path.isdir(PROCESSED_DIR):
        st.error(
            "Folder `data/processed` tidak ditemukan.\n\n"
            "Jalankan dulu preprocessing di tab *Preprocessing*."
        )
        return

    doc_ids, docs, incidence_matrix, vocabulary, inverted_index = prepare_boolean_structures()
    if not docs or inverted_index is None:
        st.error("Tidak ada dokumen yang bisa dipakai untuk Boolean IR.")
        return

    st.markdown(
        f"- Jumlah dokumen ter-preprocess: **{len(docs)}**  \n"
        f"- Ukuran vocabulary (Boolean IR): **{len(vocabulary)}**"
    )

    st.caption(
        "Contoh query Boolean:  \n"
        "`indonesia AND cnn`  \n"
        "`(internet AND teknologi) OR keamanan NOT bisnis`"
    )

    q_bool = st.text_input("Masukkan Boolean query:", value="indonesia AND cnn")

    st.markdown("#### (Opsional) Mini Truth Set untuk Precision/Recall")
    st.write(
        "Pilih dokumen yang menurut kamu **relevan** untuk query ini. "
        "Ini akan dipakai sebagai *gold relevant docs* untuk menghitung precision & recall."
    )
    gold_selected = st.multiselect(
        "Pilih gold relevant docs (berdasarkan nama file dokumen):",
        options=doc_ids,
    )

    run_demo = st.button("Jalankan Boolean Retrieval")

    if run_demo:
        if not q_bool.strip():
            st.warning("Silakan masukkan Boolean query terlebih dahulu.")
            return

        # Sama seperti demo_boolean() di main.py → pakai explain_boolean_query
        try:
            result_docs, steps = explain_boolean_query(q_bool, inverted_index)
        except Exception as e:
            st.error(f"Gagal mem-parsing Boolean query: {e}")
            return

        st.markdown("### Hasil Boolean IR")
        st.write(f"**Query:** `{q_bool}`")

        # result_docs adalah himpunan index dokumen (0-based)
        resolved_ids: List[str] = []
        for d in sorted(result_docs):
            if 0 <= d < len(doc_ids):
                resolved_ids.append(doc_ids[d])

        if not resolved_ids:
            st.info("Tidak ada dokumen yang cocok dengan Boolean query.")
            return

        st.success(f"Jumlah dokumen cocok: **{len(resolved_ids)}**")

        st.markdown("#### Daftar Dokumen Hasil")
        for did in resolved_ids:
            with st.expander(f"📄 {did}"):
                st.code(load_original_doc(did), language="text")

        # ----- Penjelasan langkah Boolean (interseksi / union / komplemen)
        st.markdown("### Penjelasan Langkah Boolean (AND / OR / NOT)")
        if not steps:
            st.info("Tidak ada langkah yang bisa dijelaskan (query kosong?).")
        else:
            for i, stp in enumerate(steps, start=1):
                term = stp["term"]
                op = stp["op"]
                used_not = stp["used_not"]
                raw_docs = stp["raw_docs"]
                docs_after_not = stp["docs_after_not"]
                prev_result = stp["prev_result"]
                new_result = stp["new_result"]

                with st.expander(f"Langkah {i}: term '{term}'"):
                    if used_not:
                        st.write(f"NOT `{term}`: komplemen dari {raw_docs} → {docs_after_not}")
                    else:
                        st.write(f"Postings `{term}`: {raw_docs}")

                    if op == "and":
                        st.write("Operasi: **AND** (interseksi) dengan hasil sebelumnya.")
                    elif op == "or":
                        st.write("Operasi: **OR** (union) dengan hasil sebelumnya.")
                    else:
                        st.write("Operasi: **INIT / result awal** (belum ada kombinasi AND/OR).")

                    if prev_result is not None:
                        st.write(f"Hasil sebelum langkah ini: {prev_result}")
                    st.write(f"Hasil sesudah langkah ini: {new_result}")

        # ----- Precision & Recall seperti di main.demo_boolean()
        if gold_selected:
            gold_indices = [doc_ids.index(name) for name in gold_selected]
            p, r = precision_recall(result_docs, gold_indices)

            st.markdown("### Evaluasi Boolean Result Set (Mini Truth Set)")
            st.write(f"Gold relevant docs (nama file): {', '.join(gold_selected)}")
            c1, c2 = st.columns(2)
            c1.metric("Precision", f"{p:.3f}")
            c2.metric("Recall", f"{r:.3f}")
        else:
            st.info(
                "Jika ingin menghitung precision/recall, pilih beberapa dokumen "
                "sebagai gold relevant docs di atas."
            )


# ========== PAGE: DEMO VSM (VECTOR SPACE MODEL & RANKING) ==========

def page_vsm_demo():
    st.header("📐 Vector Space Model & Ranking Demo (VSM)")

    if not os.path.isdir(PROCESSED_DIR):
        st.error(
            "Folder `data/processed` tidak ditemukan.\n\n"
            "Jalankan dulu preprocessing di tab *Preprocessing*."
        )
        return

    docs, doc_ids, vectorizer, tfidf_matrix = prepare_vsm_structures()
    if not docs or vectorizer is None or tfidf_matrix is None:
        st.error("Dokumen atau struktur VSM belum siap.")
        return

    st.markdown(
        f"- Jumlah dokumen ter-preprocess: **{len(docs)}**  \n"
        "- Model: **VSM dengan bobot TF-IDF & cosine similarity**"
    )

    col_q, col_k = st.columns([3, 1])
    with col_q:
        q_vsm_raw = st.text_input("Masukkan query VSM:", value="data mining knn")
    with col_k:
        k_vsm = st.number_input("k (top-k ranking)", min_value=1, max_value=50, value=5, step=1)

    default_gold = "137485529.techno.com.txt, 88974145.techno.com.txt"
    gold_input = st.text_input(
        "Gold documents (dipakai untuk P@k, MAP, nDCG@k) – pisahkan dengan koma:",
        value=default_gold,
    )

    run_vsm = st.button("Jalankan VSM Demo")

    if run_vsm:
        if not q_vsm_raw.strip():
            st.warning("Silakan masukkan query terlebih dahulu.")
            return

        q_vsm = preprocess_text(q_vsm_raw)
        with st.spinner("Menghitung ranking VSM..."):
            try:
                top_results = rank(
                    q_vsm,
                    docs,
                    doc_ids,
                    vectorizer,
                    tfidf_matrix,
                    k=int(k_vsm),
                )
            except Exception as e:
                st.error(f"Gagal menjalankan ranking VSM: {e}")
                return

        st.markdown("### TOP VSM Ranking")
        if not top_results:
            st.info("Tidak ada hasil ranking.")
        else:
            for r in top_results:
                with st.container():
                    st.markdown(f"**📄 {r['doc_id']}**  |  skor = `{r['score']:.4f}`")
                    snippet = r.get("snippet") or ""
                    if snippet:
                        st.write(snippet)
                    st.markdown("---")

        # Hitung metrik evaluasi seperti di demo_vsm CLI
        gold_docs = [
            g.strip() for g in gold_input.split(",")
            if g.strip()
        ]
        retrieved_ids = [r["doc_id"] for r in top_results]

        if gold_docs:
            p_at = precision_at_k(retrieved_ids, gold_docs, int(k_vsm))
            ap = average_precision(retrieved_ids, gold_docs, int(k_vsm))
            ndcg = ndcg_at_k(retrieved_ids, gold_docs, int(k_vsm))

            st.markdown("### Metrik Evaluasi (berdasarkan gold docs)")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Precision@{int(k_vsm)}", f"{p_at:.3f}")
            c2.metric(f"MAP@{int(k_vsm)}", f"{ap:.3f}")
            c3.metric(f"nDCG@{int(k_vsm)}", f"{ndcg:.3f}")
        else:
            st.info("Gold documents kosong, metrik evaluasi tidak dihitung.")


# ========== PAGE: SEARCH ENGINE (BOOLEAN / VSM / BM25) ==========

def page_search_engine():
    st.header("🔍 Search Engine (Boolean / VSM / BM25)")

    if not os.path.isdir(PROCESSED_DIR):
        st.error(
            "Folder `data/processed` tidak ditemukan.\n\n"
            "Jalankan dulu preprocessing di tab *Preprocessing*."
        )
        return

    st.caption(
        "Mode ini setara dengan fungsi **interactive_search()** di main.py:\n\n"
        "- Pilih model: **Boolean**, **VSM (TF-IDF)**, atau **BM25**  \n"
        "- Masukkan query  \n"
        "- Lihat ranking & top terms"
    )

    model_label = st.selectbox(
        "Pilih model pencarian",
        ["Boolean", "VSM (TF-IDF)", "BM25"],
        index=1,
    )
    model_map = {
        "Boolean": "boolean",
        "VSM (TF-IDF)": "vsm",
        "BM25": "bm25",
    }
    model_key = model_map[model_label]

    k = st.slider("Jumlah dokumen (k)", min_value=1, max_value=20, value=5, step=1)
    query = st.text_input("Masukkan query (kosong = tidak ada pencarian):", value="")

    if st.button("Jalankan Search Engine"):
        if not query.strip():
            st.warning("Silakan masukkan query terlebih dahulu.")
            return

        with st.spinner(f"Menggunakan model **{model_label}** ..."):
            results = run_search(query.strip(), model_key, int(k))

        st.markdown(
            f"### 🔍 Hasil Pencarian (model = `{model_key}`) untuk query: `{query.strip()}`"
        )

        if not results:
            st.info("Tidak ada dokumen cocok.")
            return

        for i, r in enumerate(results, 1):
            doc_id = r.get("doc_id", "(tanpa id)")
            score = r.get("score", 0.0)
            snippet = r.get("snippet") or ""
            top_terms = r.get("top_terms") or []

            with st.container():
                st.markdown(f"#### {i}. {doc_id}")
                st.markdown(f"**Score:** `{score:.4f}`")

                if top_terms:
                    parts = []
                    for t in top_terms:
                        if isinstance(t, str):
                            parts.append(t)
                        elif isinstance(t, (list, tuple)) and len(t) >= 2:
                            try:
                                parts.append(f"{t[0]}({float(t[1]):.3f})")
                            except Exception:
                                parts.append(str(t))
                        else:
                            parts.append(str(t))
                    st.caption("Top terms: " + ", ".join(parts))

                if snippet:
                    st.write(snippet)

                with st.expander("Lihat isi dokumen lengkap"):
                    st.code(load_original_doc(doc_id), language="text")

                st.markdown("---")


# ========== PAGE: EVALUATION (SEMUA MODEL, GOLD.JSON) ==========

def page_eval_all():
    st.header("📊 Evaluation dengan gold.json (Semua Model)")

    if not os.path.isdir(PROCESSED_DIR):
        st.error(
            "Folder `data/processed` tidak ditemukan.\n\n"
            "Jalankan dulu preprocessing dan push folder tersebut ke repo."
        )
        return

    if not os.path.exists(GOLD_PATH):
        st.error(
            f"File `gold.json` tidak ditemukan di: `{GOLD_PATH}`.\n"
            "Buat dulu file gold.json (mapping query -> daftar dokumen relevan)."
        )
        return

    st.caption(
        "Mode ini setara dengan fungsi **interactive_eval()** di main.py:\n\n"
        "- Memanggil `evaluate_all(PROCESSED_DIR, gold.json, k=5)`  \n"
        "- Menampilkan hasil metrik untuk semua model  \n"
        "- Membuat plot perbandingan metrik (disimpan & ditampilkan)"
    )

    k_eval = st.slider("k untuk evaluasi (P@k, MAP@k, nDCG@k)", 1, 20, 5, 1)

    if st.button("Jalankan Evaluation"):
        with st.spinner("Menghitung evaluasi untuk semua model..."):
            try:
                results = evaluate_all(PROCESSED_DIR, GOLD_PATH, k=int(k_eval))
            except Exception as e:
                st.error(f"Gagal evaluasi: {e}")
                return

        st.markdown("### Hasil Evaluation (Semua Model)")
        # results biasanya dict: {model_name: {metric_name: value, ...}, ...}
        rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name}
            if isinstance(metrics, dict):
                for mk, mv in metrics.items():
                    row[str(mk)] = mv
            rows.append(row)

        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Tidak ada hasil evaluasi yang bisa ditampilkan.")

        # Plot hasil seperti di interactive_eval
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out_img = os.path.join(REPORTS_DIR, "metrics_comparison.png")
        try:
            plot_results(results, out_path=out_img)
            if os.path.exists(out_img):
                st.markdown("### Plot Perbandingan Metrik")
                st.image(out_img, caption=f"Perbandingan metrik (disimpan di {out_img})")
            else:
                st.warning("Plot tidak ditemukan setelah pemanggilan plot_results.")
        except Exception as e:
            st.warning(f"Plot gagal dibuat: {e}")


# ========== MAIN APP ==========

def main():
    st.set_page_config(
        page_title="Mini Project STKI - IR System",
        layout="wide",
    )

    st.title("📚 Mini Project STKI - Sistem Temu Kembali Informasi")
    st.caption("Porting fitur `main.py` ke dalam UI Streamlit")

    tab_pre, tab_bool, tab_vsm, tab_search, tab_eval = st.tabs(
        [
            "1️⃣ Preprocessing",
            "2️⃣ Boolean IR",
            "3️⃣ VSM & Ranking",
            "4️⃣ Search Engine",
            "5️⃣ Evaluation (gold.json)",
        ]
    )

    with tab_pre:
        page_preprocess()

    with tab_bool:
        page_boolean_demo()

    with tab_vsm:
        page_vsm_demo()

    with tab_search:
        page_search_engine()

    with tab_eval:
        page_eval_all()


if __name__ == "__main__":
    main()
