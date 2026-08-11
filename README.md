# Modern Mathematics for LLM Engineering

Small, reproducible experiments that apply modern mathematics to **practical AI/LLM processes**.

The mathematics is not the destination. Each lab starts from a concrete process such as retrieval, RAG context selection, parameter-efficient fine-tuning, adapter composition, or reranking, then asks whether a mathematical intervention can improve that process under a controlled budget.

## Scope rule

A topic belongs in this repository only when it is connected to a measurable AI/LLM process.

Every lab follows the same chain:

> **AI process -> baseline -> mathematical intervention -> hypothesis -> papers -> Python experiment -> metrics -> result**

Whenever possible, the base representation/model remains frozen. This makes it possible to test whether structure around the model can change behavior without expensive full retraining.

## Labs in v0.1.1

| Area | Lab | AI process | Mathematical intervention | Status |
|---|---|---|---|---|
| Geometry | [metric_rag](geometry/metric_rag/) | dense RAG retrieval | Mahalanobis / task-conditioned metric | runnable |
| Geometry | [information_geometry_finetuning](geometry/information_geometry_finetuning/) | fine-tuning | Fisher metric / natural gradient | runnable |
| Topology | [sheaf_rag](topology/sheaf_rag/) | RAG context selection | local-to-global consistency energy | runnable |
| Algebra | [lie_algebra_finetuning](algebra/lie_algebra_finetuning/) | PEFT-style adaptation | Lie algebra constrained update | runnable |
| Algebra | [noncommutative_adapter_composition](algebra/noncommutative_adapter_composition/) | LoRA/adapter composition | commutator / order-sensitive operators | runnable |
| Optimal transport | [optimal_transport_reranking](transport/optimal_transport_reranking/) | retrieval reranking | entropic optimal transport | runnable |
| High-dimensional probability | [high_dimensional_quantization_rag](probability/high_dimensional_quantization_rag/) | RAG vector-index compression / online ingest | random rotation + data-oblivious low-bit quantization | runnable |

## Run

```bash
python -m pip install -e .
python run_all.py
```

Each lab can also be run independently:

```bash
python geometry/metric_rag/example.py
```

## Design principles

1. **Real AI operation first.** No standalone demonstrations of mathematical definitions.
2. **Baseline required.** Every intervention must be compared with a conventional method.
3. **Papers in every README.** Each lab identifies the papers that motivate the baseline and the mathematical intervention.
4. **Small before large.** v0.1 uses deterministic, synthetic frozen representations so the mechanism is visible and fast to test.
5. **Scale only after mechanism validation.** The next stage replaces synthetic arrays with frozen sentence/LLM embeddings, then open LLMs where warranted.
6. **Do not confuse demonstration with evidence.** The current examples validate implementation ideas, not production-level gains.

## Planned directions

- high-dimensional probability for training-free / online embedding quantization
- spectral context selection for RAG
- persistent-homology diagnostics for fine-tuning / representation collapse
- Clifford/geometric-algebra interventions for multimodal representations
- Hamiltonian and constrained dynamics for stable adaptation
- noncommutative geometry for operator-defined latent distances
- geometry-aware adapter routing
- topology + geometry combinations for context selection

See [ROADMAP.md](ROADMAP.md).
