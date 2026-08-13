# <Lab name>

## AI process
Describe the concrete LLM/AI process being modified.

## Baseline
State the conventional baseline.

## Mathematical intervention
Define the mathematical object/operator and where it enters the pipeline.

## Hypothesis
State one falsifiable claim.

## Related papers
For every paper include: title, authors, venue/year, stable URL, and exactly what idea this lab uses from it. Separate published claims from repository extensions.

## Experiment
Explain `example.py`, controls, frozen components, and variable components.

## Metrics
List quantitative metrics and compute/memory constraints.

## Observed result
Record output from the current implementation.

## Interpretation
Say what the result supports and what it does not support.

## Next experiment
Define the smallest step toward a real LLM/RAG benchmark.

## Real sentence-embedding companion

When the mechanism acts on text embeddings or latent representations, add:

```text
example_hf.py
```

Requirements:

- use a frozen public encoder (default: `sentence-transformers/all-MiniLM-L6-v2`);
- use short, inspectable phrases rather than random arrays;
- preserve the same baseline/intervention comparison as `example.py`;
- do not silently introduce model fine-tuning;
- document the embedding model/paper in the README;
- keep the heavy dependency optional via `python -m pip install -e ".[hf]"`.


## Portable implementation path

Only add this section after profiling shows a real bottleneck.

Document:

- the bottleneck observed in the Python/HF experiment;
- the portable Rust implementation, when added;
- the wgpu/WGSL GPU implementation, when added;
- any specialized CUDA/Triton, Metal or Vulkan fast path;
- numerical tolerance against the reference implementation;
- AI metric before/after optimization;
- latency, memory, throughput and other relevant systems metrics;
- supported desktop/mobile platforms and fallback behavior.

Do not add a backend only to demonstrate the technology. The backend must improve a measured AI/LLM engineering process.
