# Modern Math for LLM Engineering

Practical, paper-backed experiments that apply modern mathematics to real AI and LLM engineering problems.

The goal is not to demonstrate mathematics in isolation. Each lab starts from an existing AI process, introduces one mathematical intervention, and measures whether it improves the process under a controlled budget.

> Improve RAG, retrieval, fine-tuning, adapter composition, compression, routing, or inference without defaulting to full-model retraining.

## Project rule

A topic belongs in this repository only when it changes or diagnoses a measurable AI/LLM process.

Every lab follows the same chain:

```text
AI process
  -> baseline
  -> mathematical intervention
  -> hypothesis
  -> related papers
  -> Python experiment
  -> metrics
  -> result
```

Whenever possible, the underlying model or embeddings remain frozen. This isolates the effect of the mathematical intervention and makes compute cost part of the comparison.

## Two experiment levels

Each current lab has two executable layers:

```text
example.py     -> minimal deterministic mechanism using only NumPy
example_hf.py  -> the same idea over frozen embeddings from simple real sentences
```

The Hugging Face companion examples use `sentence-transformers/all-MiniLM-L6-v2` so one downloaded model can be reused across the repository. The encoder remains frozen; the experiment changes only the mathematical intervention around the representation.

Install the optional embedding stack:

```bash
python -m pip install -e ".[hf]"
```

Run all real-sentence experiments:

```bash
python run_hf.py
```

The default CI runs the lightweight NumPy labs and syntax-checks the HF companions; it does not download model weights on every commit.

The model maps sentences and paragraphs to a 384-dimensional dense space; see its Hugging Face model card and the Sentence-BERT paper referenced in each lab README.

## Current labs

### Geometry

- [Metric RAG](geometry/metric_rag/) - Replace fixed cosine similarity with a task-conditioned Mahalanobis metric for dense retrieval.
- [Information Geometry Fine-Tuning](geometry/information_geometry_finetuning/) - Compare ordinary parameter updates with Fisher-aware / natural-gradient updates.

### Topology

- [Sheaf RAG](topology/sheaf_rag/) - Use local-to-global consistency to select a less contradictory RAG context.

### Algebra

- [Lie Algebra Fine-Tuning](algebra/lie_algebra_finetuning/) - Constrain parameter-efficient updates to a structured Lie-algebra family.
- [Noncommutative Adapter Composition](algebra/noncommutative_adapter_composition/) - Measure when adapter order matters through commutators and order-sensitive operators.

### Optimal Transport

- [Optimal Transport Reranking](transport/optimal_transport_reranking/) - Rerank retrieval candidates using token-level distributions instead of only pooled vectors.

### High-Dimensional Probability

- [High-Dimensional Quantization for RAG](probability/high_dimensional_quantization_rag/) - Test random rotations and data-oblivious low-bit quantization for frozen embedding indexes, inspired by TurboQuant-style methods.

## Run the labs

Install the project:

```bash
python -m pip install -e .
```

Run every lab:

```bash
python run_all.py
```

Run one lab directly:

```bash
python geometry/metric_rag/example.py
```

## What every lab must contain

At minimum:

```text
<lab>/
  README.md
  example.py
  example_hf.py   # when the mechanism operates on embeddings/text representations
```

The lab README must document:

1. AI/LLM process being improved.
2. Conventional baseline.
3. Mathematical intervention.
4. Testable hypothesis.
5. Related papers and primary sources.
6. Python experiment.
7. Metrics.
8. Observed result.
9. Limitations and next experiment.
10. Real-sentence companion experiment when applicable.

Papers are part of the experiment, not decorative references. Each lab should make clear what comes from prior work, what is reproduced, and what is an extension proposed in this repository.

## Design principles

- **AI process first.** No standalone demonstrations of mathematical definitions.
- **Baseline required.** An intervention has to compete with a conventional method.
- **Paper-backed.** Every lab README includes the work that motivates the experiment.
- **Frozen when possible.** Prefer changing the mathematical structure around a model before paying for full retraining.
- **Measure compute too.** Quality without memory, latency, parameter count, or training-cost context is incomplete.
- **Small before large.** Validate the mechanism with a deterministic experiment before scaling it.
- **Do not confuse a toy result with production evidence.** A runnable mechanism is the start of an experiment, not the conclusion.

## Research directions

Planned directions include:

- subspace and Grassmann-style retrieval for RAG;
- spectral context selection;
- persistent-homology diagnostics for representation collapse and fine-tuning;
- geometry-aware adapter routing;
- optimal-transport alignment for retrieval and domain adaptation;
- high-dimensional probability for training-free quantization and compression;
- noncommutative methods for adapter and operator composition;
- constrained and Hamiltonian dynamics for stable adaptation;
- topology plus geometry for context selection and consistency.

See [ROADMAP.md](ROADMAP.md) for the evolving experiment plan.

## Contributing

New labs should follow [LAB_TEMPLATE.md](LAB_TEMPLATE.md) and the rules in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
