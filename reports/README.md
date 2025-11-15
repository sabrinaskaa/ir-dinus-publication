# Mini Project STKI – IR Dinus Publication

Proyek ini adalah Information Retrieval untuk koleksi publikasi UDINUS Techno.com, dengan beberapa model:

- **Boolean Retrieval Model**
- **Vector Space Model (VSM) dengan TF-IDF & cosine similarity**
- **BM25**
- **Search Engine gabungan (Boolean / VSM / BM25)**
- **Evaluasi dengan `gold.json`** (Precision@k, MAP, nDCG@k)
- Antarmuka **CLI (main.py)** dan **Web UI (Streamlit)**

## 1. Struktur Proyek

Struktur direktori (ringkas):

```text
.
├─ app/
│  ├─ streamlit_deploy.py      # Aplikasi Streamlit
|  └─ main.py                  # CLI (menu di terminal)
├─ src/
│  ├─ preprocess.py            # Preprocessing & tokenisasi
│  ├─ boolean_ir.py            # Boolean IR (incidence matrix, inverted index, parser)
│  ├─ vsm_ir.py                # VSM, TF-IDF, ranking, metrik evaluasi
│  ├─ search_engine.py         # Wrapper search untuk Boolean / VSM / BM25
│  └─ eval.py                  # evaluate_all + plot_results
├─ data/
│  ├─ *.txt                    # Dokumen mentah
│  └─ processed/               # Dokumen hasil preprocessing (.txt)
├─ reports/
│  └─ metrics_comparison.png   # Plot evaluasi (otomatis dibuat)
├─ requirements.txt
└─ README.md
```

## 2. Instalasi Dependency

Dapat melakukan instalasi sebelum menjalankan project dengan `pip install -r requirements.txt`

## 3. Cara Menjalankan (Mode CLI – main.py)

Jalankan di terminal `python app/main.py demo`

## 4. Website

Dapat mengakses website juga di [https://sabrinaskaa-ir-uts.streamlit.app/](https://sabrinaskaa-ir-uts.streamlit.app/)
