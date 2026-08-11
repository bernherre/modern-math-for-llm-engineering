# Sheaf-inspired RAG Context Consistency

## AI process

Selecting a compact, mutually compatible context after RAG retrieval and before LLM generation.

A conventional retriever can return individually relevant chunks that disagree with one another. Passing all of them to the generator can increase ambiguity or contradiction.

## Baseline

Select the top-k chunks by relevance score alone.

## Mathematical intervention

Use a **sheaf-inspired local-to-global consistency energy**. Each chunk carries a local state and pairwise restriction maps encode how states should agree when two chunks are used together.

For a 0-cochain `x`, the toy consistency energy is:

\[
E(x)=\|\delta x\|^2.
\]

The lab then trades off retrieval relevance against compatibility.

Important: this v0.1 implementation is deliberately a small sheaf-inspired selection layer, not a full learned cellular-sheaf RAG system.

## Hypothesis

A structured consistency term can reject a highly relevant but contradictory chunk and produce a more coherent context without changing the retriever or LLM.

## Related papers

1. **Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding** — Gebhart, Hansen & Schrater, AISTATS 2023. Treats knowledge-graph embeddings as approximate global sections with consistency constraints and supports composite-relation reasoning.  
   https://proceedings.mlr.press/v206/gebhart23a.html

2. **Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs** — Bodnar et al., NeurIPS 2022. Shows how non-trivial cellular sheaves alter information propagation via the sheaf Laplacian.  
   https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract.html

3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., NeurIPS 2020. Provides the baseline retrieval-then-generation process that this lab augments.  
   https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

## Experiment

The retriever returns five chunks with relevance scores. Four local claims are compatible; one highly relevant chunk is contradictory.

The baseline takes top-3 relevance. The structured selector exhaustively searches top-3 subsets and maximizes:

\[
\text{score}(S)=\sum_{i\in S} r_i-\lambda E(S).
\]

```bash
python example.py
```

## Metrics

- sum of retrieval relevance
- consistency energy
- selected context

## Next experiment

Replace hand-defined local states with frozen sentence embeddings / NLI logits and learn pairwise restriction maps. Evaluate factual QA accuracy and contradiction rate on multi-source RAG.

## Observed v0.1 result

Top-k relevance selects a contradictory context with consistency energy **6.9962**. The structured selector gives up only 0.04 relevance-score units while reducing the energy to **0.0056**.
