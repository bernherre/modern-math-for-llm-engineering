# Metric RAG

## AI process

Dense retrieval inside a RAG pipeline.

Typical dense RAG ranks chunks with a fixed similarity such as dot product or cosine similarity. The encoder may already contain useful information, but a single global similarity can treat every embedding direction as equally meaningful for every domain.

## Baseline

Frozen query/chunk embeddings + cosine similarity.

## Mathematical intervention

Keep embeddings frozen and replace the fixed similarity with a positive-definite Mahalanobis metric:

\[
d_M(q,d)^2=(q-d)^T M(q-d).
\]

The matrix `M` acts as a task/domain-specific geometric lens. In a production version it could be learned from relevance pairs while leaving the embedding model untouched.

## Hypothesis

A lightweight metric can alter retrieval neighborhoods and improve ranking for a domain without retraining the encoder.

## Related papers

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., NeurIPS 2020. Establishes the RAG architecture with parametric generation plus non-parametric retrieval.  
   https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

2. **Dense Passage Retrieval for Open-Domain Question Answering** — Karpukhin et al., EMNLP 2020. Core dense-retrieval baseline using learned query/passage representations and vector similarity.  
   https://aclanthology.org/2020.emnlp-main.550/

3. **Efficient Learning of Mahalanobis Metrics for Ranking** — Lim & Lanckriet, ICML 2014. Learns Mahalanobis metrics directly for ranking objectives, the mathematical intervention tested here in miniature.  
   https://proceedings.mlr.press/v32/lim14.html

4. **A Statistical Framework for Data-dependent Retrieval-Augmented Models** — Basu, Rawat & Zaheer, ICML 2024. Formalizes retrieval-augmented models with a data-dependent retrieval metric.  
   https://proceedings.mlr.press/v235/basu24a.html

## Experiment

`example.py` simulates a frozen dense retriever. It evaluates:

- cosine top-k retrieval;
- a domain metric applied to the same frozen vectors;
- Recall@k against known relevant chunks.

No embedding is updated.

```bash
python example.py
```

## Metrics

- Recall@3
- top-3 ranking

## What this lab proves

Only that changing the metric can change retrieval behavior while embeddings remain frozen. It does **not** establish that the chosen metric generalizes to real RAG corpora.

## Next experiment

Learn `M` from relevance pairs on a small BEIR-style dataset and compare cosine vs metric-conditioned retrieval under identical embeddings and index budget.

## Observed v0.1 result

With the frozen toy embeddings, cosine retrieval reaches **Recall@3 = 0.50**, while the domain Mahalanobis metric reaches **Recall@3 = 1.00**. The purpose is mechanism validation: geometry alone changed the retrieved neighborhood.
