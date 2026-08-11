# Optimal-Transport Reranking

## AI process

Reranking retrieved text chunks before they enter the LLM context.

A common dense retriever compresses a query/chunk into a single vector. This can hide token-level structure: two texts can have similar means even when their semantic components do not align.

## Baseline

Mean-pool token embeddings and rank by cosine similarity.

## Mathematical intervention

Represent query and chunk as discrete distributions over frozen token embeddings and compute an entropically regularized optimal-transport cost.

The transport plan seeks a low-cost coupling between token sets rather than comparing only their means.

## Hypothesis

A training-free OT reranker can correct some false positives produced by mean-pooled similarity when token-level semantic alignment matters.

## Related papers

1. **From Word Embeddings To Document Distances** — Kusner et al., ICML 2015. Introduces Word Mover's Distance, casting text distance as an optimal-transport / Earth Mover problem over word embeddings.  
   https://proceedings.mlr.press/v37/kusnerb15.html

2. **Re-evaluating Word Mover's Distance** — Sato, Yamada & Kashima, ICML 2022. Reassesses WMD carefully and shows why strong baselines/preprocessing are essential; included to avoid treating OT as automatically superior.  
   https://proceedings.mlr.press/v162/sato22b.html

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., NeurIPS 2020. RAG baseline whose candidate list this lab conceptually reranks.  
   https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

4. **CR-Refiner: An Object-Centric Optimal Transport Reranker for Edit-Conditioned 3D Scene Retrieval** — Wu et al., 2026. Recent example of a training-free OT reranker wrapped around a base retriever; not an LLM RAG paper, but directly relevant to the reranking pattern.  
   https://arxiv.org/abs/2607.19115

## Experiment

The query and each document contain two frozen token embeddings. A decoy document has the same mean direction as the query, so mean-cosine ranks it first. OT sees that the individual token geometry is wrong and prefers the truly aligned document.

```bash
python example.py
```

## Metrics

- cosine ranking over mean embeddings
- Sinkhorn OT cost ranking
- rank of the known relevant chunk

## Next experiment

Use frozen transformer token embeddings for a real retrieval benchmark. Apply OT only to the top-N candidates so cost remains practical, and compare nDCG/Recall/latency against a cross-encoder reranker.

## Observed v0.1 result

Mean-pooled cosine ranks the decoy first and the relevant chunk second. The Sinkhorn OT cost ranks the relevant chunk first (**0.05385**) and the mean-decoy second (**0.71063**).
