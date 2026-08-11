# Noncommutative Adapter Composition

## AI process

Combining independently trained LoRA/adapters at inference time.

Adapter libraries are attractive because a frozen base model can acquire multiple skills without full retraining. But composition is not generally order-independent: applying adapter `A` then `B` can differ from applying `B` then `A`.

## Baseline

Naive additive merge of two low-rank adapter operators.

## Mathematical intervention

Treat adapters as linear operators and explicitly measure their **commutator**:

\[
[A,B]=AB-BA.
\]

If the commutator is large, composition order matters and a naive commutative merge assumption is risky.

## Hypothesis

The commutator norm can serve as a cheap diagnostic for when adapter order/composition deserves explicit optimization instead of simple averaging/addition.

## Related papers

1. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022. Provides the adapter-like low-rank update mechanism used by the toy operators.  
   https://openreview.net/forum?id=nZeVKeeFYf9

2. **LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models** — Hu et al., 2023. Studies adapter-based PEFT across LLM reasoning tasks.  
   https://arxiv.org/abs/2304.01933

3. **Beyond Adapter Retrieval: Latent Geometry-Preserving Composition via Sparse Task Projection** — Jin et al., 2025. Formulates adapter composition as geometry-aware sparse reconstruction rather than simple retrieval/averaging.  
   https://arxiv.org/abs/2410.09908

4. **Task-Aware LoRA Adapter Composition via Similarity Retrieval in Vector Databases** — Adsul et al., 2026. Studies dynamic retrieval and merging of task-specific LoRA adapters with frozen embeddings.  
   https://arxiv.org/abs/2602.21222

## Experiment

Two rank-1 adapters are applied to frozen hidden states. The target behavior is `A` followed by `B`. Compare:

- `A -> B`;
- `B -> A`;
- naive additive merge;
- commutator norm.

```bash
python example.py
```

## Metrics

- MSE against the target composed behavior
- Frobenius norm of `[A, B]`

## Next experiment

Run the same diagnostic on real LoRA delta matrices. Test whether commutator magnitude predicts quality loss for merge orders, TIES-like merges, linear merges, or routed adapter stacks.

## Observed v0.1 result

The two rank-1 adapter operators have commutator norm **0.07766**. Reversing the intended order produces MSE **0.00142657**, and a naive additive merge produces MSE **0.00140181**, while the intended ordered composition is exact in this controlled example.
