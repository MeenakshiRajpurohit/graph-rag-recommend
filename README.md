# X-KGRank: A Graph RAG Framework for Explainable Recommendations

A knowledge-graph retrieval-augmented framework that unifies structural
collaborative filtering with LLM-based explanation. X-KGRank constructs a
heterogeneous knowledge graph from MovieLens-1M, trains a LightGCN ranker with
content-aware SBERT initialisation, mines structural and community-level graph
features, retrieves candidate-specific KG paths at inference time, and routes
each candidate through one of two LLM explanation prompts based on its
popularity bucket.

On MovieLens-1M (99-sample protocol), X-KGRank reaches **NDCG@10 = 0.2956** and
**Recall@10 = 0.5371**, improving over a strong popularity baseline by **17.1%**
on both metrics.

> Paper: *X-KGRank: A Graph RAG Framework for Explainable Recommendations via
> Pattern Mining and LLM Re-Ranking.* San José State University.

## Authors

- Meenakshi Rajpurohit — meenakshi.rajpurohit@sjsu.edu
- Jainish Patel — jainish.patel@sjsu.edu
- Erick Vazquez — erick.vazquez@sjsu.edu

## Repository layout

```
graph-rag-recommend/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── FINAL_PROJECT_CODE.ipynb     # full runnable Colab notebook (canonical)
└── src/                             # notebook exported stage-by-stage
    ├── 00_setup_and_data.py         # data + KG loading, popularity split
    ├── 01_sbert_encoder.py          # SBERT content-aware item init
    ├── 02_lightgcn_bpr.py           # LightGCN ranker, rating-weighted BPR
    ├── 03_graph_mining.py           # node2vec + community detection
    ├── 04_kg_projector.py           # MLP projector (node2vec -> T5 space)
    ├── 05_llm_kg_retrieval.py       # Flan-T5 + 4-tier KG path retrieval
    ├── 06_pipeline.py               # full X-KGRank pipeline + routing
    ├── 07_popularity_baseline.py    # baseline + ranking metrics
    ├── 08_rag_demo.py               # NL RAG query demo + LoRA utilities
    ├── 09_llm_comparison.py         # Flan-T5 vs Qwen2.5-1.5B vs Mistral-7B
    └── 10_visualizations.py         # paper figures, dashboards, Cypher
```

The `src/` files are exported directly from the notebook. They share global
state (`df_train`, `KG`, the trained LightGCN, etc.), so the notebook is the
canonical way to run everything end-to-end; the `.py` files mirror the notebook
stage-by-stage for review and version control.

## Pipeline overview

1. **Knowledge graph** — 9,762 nodes (6,040 users, 3,704 movies, 18 genres) and
   999,264 edges (RATED, HAS_GENRE, CO_RATED), persisted in Neo4j and projected
   to NetworkX for traversal.
2. **Structural learning** — LightGCN (3 layers, 128-d, ~1.3M params) warm-started
   from SBERT title+genre embeddings.
3. **Graph mining** — node2vec (64-d) + greedy modularity communities.
4. **Popularity-selective routing** — median split (1,855 cold / 1,849 warm);
   cold items get KG-grounded prompts, warm items get open prompts. Cuts
   KG-augmented generations by ~50%.
5. **Grounded explanation** — four-tier KG path fallback feeds a two-sentence
   prompt to the LLM; ranking stays fixed by LightGCN scores.

## Data & checkpoints

The preprocessed data bundle (`xkgrank_save.zip`: id maps, train/valid/test
CSVs, `edge_index.pt`, `kg_graph.pkl`) and the trained checkpoints
(`lightgcn_best.pt`, `kg_projector.pt`, LoRA adapters) are **too large for git**
and are hosted on Google Drive:

**Google Drive (data + checkpoints):** `<PASTE_YOUR_DRIVE_SHARE_LINK_HERE>`

See the "How to upload to Google Drive" section below to create this link.

## Running

The project was developed on **Google Colab Pro** with a single NVIDIA A100 GPU.

1. Open `notebook/FINAL_PROJECT_CODE.ipynb` in Colab.
2. Run the first cell — it mounts Google Drive and prompts you to upload
   `xkgrank_save.zip` (or reads it from your Drive).
3. Run the cells in order (Stage 0 → Stage 10). End-to-end training across all
   stages takes roughly three hours.

Local runs are possible with the dependencies in `requirements.txt`, but a CUDA
GPU is strongly recommended for the LightGCN, node2vec, and LLM stages.

## How to upload data/checkpoints to Google Drive

1. Go to <https://drive.google.com> and sign in.
2. Click **+ New → New folder**, name it `xkgrank_release`.
3. Open the folder, click **+ New → File upload**, and upload `xkgrank_save.zip`
   and your checkpoint files.
4. Right-click the folder → **Share → Share**.
5. Under *General access*, switch from "Restricted" to
   **"Anyone with the link"**, set the role to **Viewer**, then **Copy link**.
6. Paste that link into the "Data & checkpoints" section above (replace the
   `<PASTE_YOUR_DRIVE_SHARE_LINK_HERE>` placeholder) and commit the change.

## Citation

```bibtex
@misc{rajpurohit2025xkgrank,
  title  = {X-KGRank: A Graph RAG Framework for Explainable Recommendations
            via Pattern Mining and LLM Re-Ranking},
  author = {Rajpurohit, Meenakshi and Patel, Jainish and Vazquez, Erick},
  year   = {2025},
  note   = {San Jose State University}
}
```
