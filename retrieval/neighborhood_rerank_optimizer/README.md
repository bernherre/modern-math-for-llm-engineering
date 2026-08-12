# Neighborhood Rerank Optimizer

## AI process

Allocate reranking compute in a RAG/retrieval pipeline.

A conventional two-stage retriever often applies the expensive reranker to a fixed number of candidates for every query. Easy queries therefore pay the same second-stage cost as ambiguous queries.

## Baseline

1. retrieve a fixed top-N list with pooled dense embeddings;
2. run the expensive reranker over all N candidates.

The controlled lab uses token-level MaxSim as the expensive stage so the compute budget is directly countable without requiring another generative LLM.

## Mathematical intervention

Treat the retrieved candidate list as a **local geometric neighborhood** around the query and measure its shape before spending reranker compute.

The lightweight gate uses:

- the margin between the top two first-stage scores;
- entropy of the local similarity distribution;
- effective rank of the centered candidate embeddings;
- average angular dispersion of the neighborhood.

Those statistics produce a training-free difficulty score. Easy neighborhoods skip reranking; ambiguous neighborhoods receive progressively larger budgets.

Conceptually:

```text
cheap retriever
    -> candidate neighborhood
    -> local geometry
    -> budget 0 / small / medium / full
    -> expensive reranker only where needed
```

## Hypothesis

**The geometry of the retrieved neighborhood contains enough query-difficulty information to reduce reranker work while preserving the quality of a fixed-depth reranking pipeline.**

The specific combination of margin, entropy, effective rank and dispersion used in this lab is an experimental proposal from this repository. The papers below motivate adaptive computation, adaptive retrieval depth and local-neighborhood difficulty; they do not claim this exact gate.

## Related papers

1. **AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking** — Yoon et al., 2025. Dynamically allocates reranking computation instead of using a fixed amount for every query. The paper uses ranking uncertainty; this lab asks whether candidate-space geometry can be a cheaper control signal.  
   https://arxiv.org/abs/2505.18512

2. **Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications** — Xu et al., 2025. Uses clustering patterns in ordered query-document distances to adapt retrieval depth and reports lower token/latency cost at preserved relevance. It is the closest direct precedent for using the shape of retrieval scores as a RAG control signal.  
   https://arxiv.org/abs/2511.14769

3. **The Role of Local Intrinsic Dimensionality in Benchmarking Nearest Neighbor Search** — Aumuller & Ceccarello, 2019. Shows that local intrinsic dimensionality is informative about nearest-neighbor query difficulty. The lab uses effective rank as a lightweight spectral proxy first, with LID planned as a stronger alternative.  
   https://arxiv.org/abs/1907.07387

4. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** — Khattab & Zaharia, SIGIR 2020. Provides the late-interaction/MaxSim pattern used here as a deliberately more expensive second stage than pooled cosine.  
   https://arxiv.org/abs/2004.12832

5. **Spectral Retrieval: Multi-Scale Sinc Convolution over Token Embeddings for Localized Retrieval in LLM Multi-Agent Systems** — Morandi et al., 2026. A recent example of a token-level second-stage reranker applied only to a first-stage candidate pool, reinforcing the value of controlling expensive localized scoring rather than applying it corpus-wide.  
   https://arxiv.org/abs/2605.24764

## Minimal experiment

`example.py` creates frozen token representations with four controlled query regimes. Candidate generation uses cheap mean-pooled cosine. The expensive score is token MaxSim.

Run:

```bash
python retrieval/neighborhood_rerank_optimizer/example.py
```

The adaptive policy is compared against:

- no reranking;
- fixed top-12 MaxSim reranking;
- neighborhood-adaptive MaxSim reranking.

## Observed lightweight result

The deterministic v0 experiment currently produces:

```text
no rerank top-1 accuracy : 0.792
fixed top-12 MaxSim      : 1.000
adaptive MaxSim accuracy : 1.000
fixed rerank calls/query : 12.00
adaptive calls/query     : 2.00
rerank work saved        : 83.3%
```

This is a deliberately constructed mechanism test, **not** evidence that an 83% saving transfers to BEIR, MS MARCO or production RAG.

## Real sentence-embedding experiment

`example_hf.py` uses one frozen SentenceTransformer model for both levels:

- sentence-level normalized embeddings for cheap candidate generation;
- contextual token embeddings for MaxSim reranking.

The neighborhood gate only decides how many candidates receive token-level scoring. The encoder itself is not fine-tuned.

Install:

```bash
python -m pip install -e ".[hf]"
```

Run:

```bash
python retrieval/neighborhood_rerank_optimizer/example_hf.py
```

Embedding reference:

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** — Reimers & Gurevych, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
- `sentence-transformers/all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Metrics

- first-stage top-1 accuracy;
- fixed-rerank top-1 accuracy;
- adaptive-rerank top-1 accuracy;
- reranked candidates per query;
- fraction of reranker work avoided;
- later: nDCG@10, Recall@k, p50/p95 latency, token/FLOP cost and final RAG answer quality.

## Next experiment

Use a public retrieval benchmark and a real cross-encoder or LLM reranker. Compare:

1. fixed top-20 rerank;
2. fixed top-10 rerank;
3. AcuRank-style adaptive compute;
4. score-gap-only gating;
5. neighborhood geometry gating;
6. geometry gating plus [matroid-constrained candidate selection](../../combinatorics/matroid_selection/) so the rerank budget is spent on structurally independent evidence.

The most important target is not a higher standalone reranker score. It is the same or better end-to-end RAG quality with materially less second-stage compute.
