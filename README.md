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
