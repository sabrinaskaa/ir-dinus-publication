import os
import sys

import streamlit as st

# ==== Setup path supaya bisa import src/ ====
# Lokasi file ini (app/main.py)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = parent dari folder app
PROJECT_ROOT = os.path.dirname(FILE_DIR)

# Tambahkan project root ke sys.path (kalau belum ada)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Sekarang import dari src/ akan bekerja
from src.search_engine import search as engine_search  # type: ignore

# Lokasi data yang sudah dipreproses
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def run_search(query: str, model_key: str, k: int = 10):
    """Wrapper pemanggilan search_engine dengan penanganan error sederhana."""
    try:
        # search(processed_dir, model, query, k, weight)
        results = engine_search(PROCESSED_DIR, model=model_key, query=query, k=k, weight="tfidf")
        return results
    except Exception as e:
        st.error(f"Gagal menjalankan pencarian: {e}")
        return []


def main():
    st.set_page_config(
        page_title="Sistem Temu Kembali Informasi",
        layout="wide",
    )

    st.title("🔍 Sistem Temu Kembali Informasi")
    st.markdown(
        "Aplikasi ini menggunakan koleksi dokumen di folder `data/processed` "
        "dan modul IR di folder `src/`."
    )

    # Cek apakah folder processed ada
    if not os.path.isdir(PROCESSED_DIR):
        st.error(
            "Folder `data/processed` tidak ditemukan.\n\n"
            "Jalankan dulu skrip preprocessing di lokal untuk membuat folder tersebut, "
            "lalu pastikan folder itu ikut dipush ke repo."
        )
        st.stop()

    # --- Sidebar: pengaturan ---
    with st.sidebar:
        st.header("Pengaturan Pencarian")

        model_label = st.selectbox(
            "Model pencarian",
            ["VSM (TF-IDF)", "BM25", "Boolean"],
            index=0,
        )

        model_map = {
            "VSM (TF-IDF)": "vsm",      # default di search_engine -> VSM
            "BM25": "bm25",
            "Boolean": "boolean",
        }
        model_key = model_map[model_label]

        k = st.slider("Jumlah dokumen yang ditampilkan (k)", min_value=5, max_value=50, value=10, step=5)

    # --- Main area: input query & hasil ---
    query = st.text_input("Masukkan query pencarian:", value="", max_chars=200)
    search_clicked = st.button("Cari")

    if search_clicked:
        if not query.strip():
            st.warning("Silakan masukkan query terlebih dahulu.")
            st.stop()

        with st.spinner("Sedang mencari dokumen..."):
            results = run_search(query.strip(), model_key, k)

        st.markdown("---")
        st.subheader(f"Hasil pencarian untuk: `{query}`")

        if not results:
            st.warning("Tidak ada dokumen yang cocok.")
        else:
            for i, r in enumerate(results, start=1):
                doc_id = r.get("doc_id", "(tanpa id)")
                score = r.get("score", 0.0)
                snippet = r.get("snippet", "")
                top_terms = r.get("top_terms", [])

                with st.container():
                    st.markdown(f"### {i}. {doc_id}")
                    st.markdown(f"**Skor:** `{score:.4f}`")

                    if snippet:
                        st.write(snippet)

                    if top_terms:
                        # top_terms: list [(term, bobot/freq), ...]
                        pretty_terms = []
                        for t, w in top_terms:
                            try:
                                pretty_terms.append(f"{t} ({float(w):.3f})")
                            except Exception:
                                pretty_terms.append(str(t))
                        st.caption("Top terms: " + ", ".join(pretty_terms))

                    st.markdown("---")


if __name__ == "__main__":
    main()
