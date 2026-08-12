# Matroid Selection for RAG Context and LLM Training Data

## AI processes

This lab applies the same combinatorial object to two expensive selection problems:

1. **RAG context selection** - choose a small set of reranked chunks that is relevant but not dominated by redundant evidence;
2. **LLM training-data selection** - spend a fixed fine-tuning/pretraining budget on high-value samples while preserving coverage across semantic/domain partitions.

The scorer is not replaced. Relevance, quality or influence is computed first. The matroid controls **which already-scored items are allowed to consume the scarce budget**.

## Baselines

### RAG

Take the top-k chunks by reranker/retriever score.

This can waste context on several highly ranked chunks that all support essentially the same facet of the answer.

### Training

Take the top-k samples by quality/influence score.

This can concentrate the training budget on one domain or one dense family of near-duplicate examples.

## Mathematical intervention

Use a **truncated partition matroid**.

The candidate set is partitioned into semantic groups. Each group has a capacity:

```text
compression : at most 1
reranking   : at most 1
filtering   : at most 1
operations  : at most 1
```

For training the partitions can instead be domains, skills, sources, languages, difficulty bands or other constraints that represent desirable coverage.

An independent set satisfies every capacity constraint and the global budget. With non-negative additive item scores, sorting by score and greedily accepting an item whenever independence is preserved gives an optimal maximum-weight independent set for the matroid objective.

The practical interpretation is useful:

```text
reranker / quality scorer
        |
        +-- low score ----------------------> unrelated / discard
        |
        +-- high score + independent -------> selected budget
        |
        +-- high score + dependent ---------> supporting / redundant pool
```

"Dependent" does **not** mean irrelevant. It means that, under the selected independence rule, the item does not add a new admissible slot to the current subset.

## Hypotheses

### RAG hypothesis

**A matroid constraint can preserve high reranker relevance while increasing evidence/facet coverage and reducing redundant context sent to the LLM.**

The natural integration point is after candidate scoring and before the final context budget is consumed. It can also sit before an expensive reranker, coupled to `retrieval/neighborhood_rerank_optimizer`, so that reranking budget is spent on structurally distinct candidates.

### Training-data hypothesis

**A matroid-constrained subset can use fewer training samples more effectively than pure top-score selection by preventing high-scoring but redundant domains/sources from monopolizing the training budget.**

The same idea can operate at dataset, shard or minibatch level.

## Related papers

### Matroid-constrained data selection

1. **Minibatch Selection via Partition Matroid Constrained Gradient Matching** - Agrawal et al., 2026. This is the closest direct precedent for the training side of this lab: it selects cross-domain LLM fine-tuning minibatches with per-domain budgets encoded as a partition-matroid constraint and a gradient-matching utility. Experiments include Qwen2.5 and Llama-3 fine-tuning.  
   https://arxiv.org/abs/2606.07954

2. **Less data is more: Selecting informative and diverse subsets with balancing constraints** - Ramalingam et al., ICLR 2022. Introduces matroid-based balancing constraints for selecting informative and diverse training subsets and reports competitive deep-model performance with less data.  
   https://openreview.net/forum?id=6PlIkYUK9As

3. **Fairness in Streaming Submodular Maximization over a Matroid Constraint** - El Halabi et al., ICML 2023. Studies representative subset selection under matroid constraints in streaming settings, useful when the candidate/training pool is too large to materialize globally.  
   https://proceedings.mlr.press/v202/el-halabi23a.html

### RAG diversity and non-redundant context

4. **DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation** - Khan et al., Findings of EACL 2026. Shows that relevance-only retrieval can introduce redundant context and that dynamically controlling diversity can improve reasoning-intensive RAG. DF-RAG uses MMR rather than matroids; this lab tests a hard combinatorial independence constraint as a different mechanism.  
   https://aclanthology.org/2026.findings-eacl.150/

### LLM training-data quality plus diversity

5. **The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph** - Wu et al., ICML 2025. GraphFilter jointly targets quality and instruction diversity for LLM data selection and reports better performance/efficiency than multiple baselines.  
   https://proceedings.mlr.press/v267/wu25ac.html

6. **D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning** - Zhang et al., IJCAI 2025. Selects instruction-tuning data using diversity, difficulty and dependability and reports competitive or better instruction-following with less than 10% of the full data in its experiments.  
   https://www.ijcai.org/proceedings/2025/928

7. **Harnessing Diversity for Important Data Selection in Pretraining Large Language Models** - Zhang et al., ICLR 2025. `Quad` combines influence-based quality with diversity-aware cluster selection for LLM pretraining, reinforcing the importance of avoiding high-score redundancy.  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/b588d9b67932b459ea66ff6e2804c6b3-Abstract-Conference.html

### Classical subset-selection connection

8. **Submodularity in Data Subset Selection and Active Learning** - Wei, Iyer & Bilmes, ICML 2015. Establishes a broad connection between subset selection, information/diversity objectives and efficient constrained optimization. A future version of this lab should combine submodular marginal utility with matroid feasibility rather than use only additive scores.  
   https://proceedings.mlr.press/v37/wei15.html

## Minimal experiment

Run:

```bash
python combinatorics/matroid_selection/example.py
```

### RAG mode

The reranker scores are fixed. The baseline takes top-k. The intervention applies one slot per evidence facet.

The output separates candidates into:

- selected independent context;
- relevant but dependent/supporting evidence;
- unrelated evidence filtered by the scorer.

### Training mode

The upstream quality scores are also fixed. The baseline spends its budget on the globally highest scores. The matroid imposes one slot per domain/skill partition.

This is a mechanism test. `macro-domain utility` is a deliberately transparent proxy for a balanced downstream evaluation suite; it is not a trained-model benchmark.

## Real sentence-embedding experiment

`example_hf.py` uses frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings.

For RAG:

- cosine similarity to the query supplies the relevance score;
- frozen semantic anchor embeddings assign chunks to evidence partitions;
- the reranker/relevance threshold first rejects unrelated chunks;
- the partition matroid then decides which relevant chunks consume context budget.

For training data:

- the sample text is embedded without fine-tuning;
- semantic anchors assign examples to skill/domain partitions;
- externally supplied quality scores remain unchanged;
- the matroid changes only the selected subset.

Install:

```bash
python -m pip install -e ".[hf]"
```

Run:

```bash
python combinatorics/matroid_selection/example_hf.py
```

Embedding references:

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** - Reimers & Gurevych, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
- `sentence-transformers/all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Metrics

### RAG

- reranker relevance sum;
- unique evidence/facet coverage;
- redundant/dependent chunks excluded from prompt budget;
- downstream answer accuracy/F1;
- context tokens;
- reranker calls when combined with neighborhood gating.

### Training

- mean selected quality/influence;
- domain/skill/source coverage;
- duplicate or gradient-conflict rate;
- training tokens/FLOPs;
- downstream macro benchmark score;
- catastrophic forgetting / retention;
- gradient matching error for a PartitionSel-style extension.

## Observed lightweight result

The deterministic `example.py` intentionally creates a case where pure top-k is dominated by redundant high-scoring items.

Current output:

```text
RAG
  baseline relevance sum : 3.820
  matroid relevance sum  : 3.580
  baseline facet coverage: 0.500
  matroid facet coverage : 1.000

Training selection
  baseline mean quality        : 0.960
  matroid mean quality         : 0.904
  baseline domain coverage     : 0.400
  matroid domain coverage      : 1.000
  baseline macro-domain utility: 0.386
  matroid macro-domain utility : 0.904
```

This is a deliberately constructed mechanism test, not evidence that these gains transfer to a production RAG pipeline or an actual LLM training run. The point is that **independence can be made an explicit optimization constraint instead of an informal diversity penalty**.

## Connection to the neighborhood rerank optimizer

A natural combined pipeline is:

```text
TurboVec / ANN
    -> cheap candidate neighborhood
    -> neighborhood geometry decides rerank budget
    -> matroid selects structurally independent candidates for that budget
    -> expensive reranker
    -> selected context + dependent supporting evidence
    -> LLM
```

This separates two decisions:

- **geometry:** how much compute should this query receive?
- **matroid:** which candidates deserve that compute/context budget?

## Next experiments

1. Compare top-k, MMR, DF-RAG-style diversity and partition-matroid selection on a public RAG benchmark.
2. Infer partitions from clusters rather than fixed anchors and test sensitivity to cluster errors.
3. Replace additive score with a submodular information-gain objective under a matroid constraint.
4. Use source/tenant/language caps for enterprise RAG where evidence provenance matters.
5. Apply partition-matroid minibatch selection to a small open LLM with real LoRA fine-tuning.
6. Compare quality-only, random, clustering, D3/GraphFilter-inspired and matroid-constrained training subsets under the same token budget.
7. Add gradient matching so the training branch can be compared directly with PartitionSel.
