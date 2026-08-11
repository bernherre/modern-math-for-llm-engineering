# Information Geometry for Fine-tuning

## AI process

Fine-tuning a small trainable head/adapter on top of frozen representations.

Ordinary gradient descent treats the parameter coordinates as Euclidean. Information geometry instead equips the model family with the Fisher information metric and takes a steepest-descent step in distribution space.

## Baseline

Gradient descent on a logistic head over frozen features.

## Mathematical intervention

Natural gradient:

\[
\theta_{t+1}=\theta_t-\eta F(\theta_t)^{-1}\nabla_\theta L,
\]

where `F` is the empirical Fisher information matrix.

## Hypothesis

When frozen features are highly anisotropic/correlated, a Fisher-preconditioned update can make more useful progress per optimization step than a coordinate-wise Euclidean update.

## Related papers

1. **Fisher Information and Natural Gradient Learning in Random Deep Networks** — Amari, Karakida & Oizumi, AISTATS 2019. Studies the Fisher metric of deep networks and tractable approximations to natural-gradient updates.  
   https://proceedings.mlr.press/v89/amari19a.html

2. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022. Establishes a widely used PEFT baseline in which pretrained weights remain frozen while small low-rank updates are trained.  
   https://openreview.net/forum?id=nZeVKeeFYf9

## Experiment

A frozen feature matrix is intentionally badly conditioned. Only a two-parameter logistic head is tuned. Under the same number of optimization steps, compare:

- Euclidean gradient descent;
- Fisher-preconditioned natural gradient.

```bash
python example.py
```

## Real sentence-embedding experiment

The real-sentence version freezes the encoder and tunes only a tiny evidence-quality logistic head, comparing ordinary gradient descent with a Fisher-preconditioned natural-gradient update under the same number of steps.

The encoder is **frozen**. The purpose is to move from hand-written/random arrays to actual language representations without introducing model training as a confounder.

Install the optional dependency once from the repository root:

```bash
python -m pip install -e ".[hf]"
```

Run:

```bash
python geometry/information_geometry_finetuning/example_hf.py
```

Embedding reference:

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** - Reimers & Gurevych, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
- `sentence-transformers/all-MiniLM-L6-v2` model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Metrics

- initial/final binary cross-entropy
- training accuracy
- number of trainable parameters (identical)
- optimization steps (identical)

## What this lab does not claim

This is not a claim that exact natural gradient is practical for full LLM fine-tuning. The useful research question is whether structured approximations (block Fisher, K-FAC-like, diagonal/low-rank Fisher, adapter-local Fisher) improve PEFT under a fixed compute budget.

## Next experiment

Apply a Fisher approximation only to LoRA parameters of a small open transformer and compare against AdamW using equal tokens, steps, trainable parameters and wall-clock budget.

## Observed v0.1 result

Under 35 updates with the same two trainable parameters, the Euclidean update finishes at loss **0.27981 / accuracy 0.8900**, while the Fisher-preconditioned update finishes at loss **0.17822 / accuracy 0.9133** on the toy tuning set.
