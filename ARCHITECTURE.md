# Portable Implementation Architecture

This repository starts from a mathematical hypothesis about an AI/LLM process and only optimizes the implementation after the mechanism has measurable value.

The preferred implementation ladder is:

```text
paper / mathematical idea
  -> example.py              reference mechanism
  -> example_hf.py           frozen real embeddings / model states
  -> portable Rust core      reusable CPU implementation
  -> wgpu + WGSL             portable GPU implementation
  -> specialized backend     only when benchmarks justify it
```

The goal is to keep the algorithm independent from the hardware backend while preserving a path from research code to desktop, server and mobile systems.

## Portability target

The portable path should be able to support, when the algorithm and runtime allow it:

| Layer | Windows | Linux | macOS | Android | iOS |
|---|---|---|---|---|---|
| Python reference | yes | yes | yes | limited | limited |
| Rust CPU | yes | yes | yes | yes | yes |
| wgpu / WGSL | yes | yes | yes | yes | yes |
| CUDA / Triton | optional | optional | no | no | no |
| Metal specialized | no | no | optional | no | optional |
| Vulkan specialized | optional | optional | no | optional | no |

`wgpu` is the preferred portable GPU layer because it can target the native graphics/compute APIs available on the host platform. Specialized backends are accelerators, not the source of truth.

## Backend rule

The mathematical algorithm must not depend on a specific backend.

A lab that grows beyond Python should expose the same conceptual operation through interchangeable implementations, for example:

```text
Reference implementation
  -> CPU implementation
  -> portable GPU implementation
  -> optional vendor-specific fast path
```

The portable implementation should define the expected numerical behavior. Specialized kernels may change precision, tiling or memory layout, but they must be checked against the reference implementation within documented tolerances.

## When to add a systems implementation

Do not add Rust, WGSL, Triton, CUDA, Metal or Vulkan merely because a lab can be ported.

Add a lower-level implementation only when the experiment identifies a measurable bottleneck such as:

- embedding/index memory;
- candidate-scoring throughput;
- reranker latency;
- token-level similarity cost;
- matrix/sketch construction cost;
- KV-cache memory;
- host/device transfer overhead;
- repeated allocations;
- energy or mobile inference constraints.

The optimization must preserve the AI-level metric that motivated the lab.

## Required measurements

When a lab adds a systems implementation, compare it with the Python/reference path using both AI and systems metrics.

AI metrics may include:

- Recall@k / nDCG / MRR;
- final RAG answer quality;
- fine-tuning loss or task score;
- forgetting / retention;
- routing or selection accuracy.

Systems metrics may include:

- latency p50 / p95;
- throughput;
- RAM / VRAM;
- index size;
- allocations;
- host-device transfer volume;
- energy when practical;
- build/index/update time.

A faster kernel that changes the final AI behavior beyond the accepted tolerance is not automatically an improvement.

## Suggested lab layout after optimization

Labs remain simple until an optimized implementation is justified.

```text
<lab>/
  README.md
  example.py
  example_hf.py
  implementations/
    rust_cpu/
    wgpu/
    triton/        # optional
    metal/         # optional
    vulkan/        # optional
```

Not every lab should have every backend.

## Mobile and desktop principle

Firmware, BIOS/UEFI and device-specific driver experiments may be useful for architecture research, but they are not the portability layer of this repository.

For reusable AI/LLM algorithms, prefer:

```text
Rust core
  + CPU fallback
  + wgpu/WGSL portable GPU path
  + vendor-specific acceleration only when useful
```

This keeps the research portable across desktop and mobile operating systems while still allowing CUDA/Triton, Metal or Vulkan fast paths where they produce a measurable gain.
