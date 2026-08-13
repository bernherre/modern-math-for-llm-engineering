# Roadmap

The repository grows by **AI process**, while keeping mathematical families visible.

## v0.1 - Mechanism tests

- [x] metric-conditioned RAG retrieval
- [x] information-geometric fine-tuning
- [x] sheaf-inspired RAG consistency selection
- [x] Lie-algebra constrained adaptation
- [x] noncommutative adapter composition diagnostics
- [x] optimal-transport reranking


## v0.1.1 - High-dimensional retrieval compression

- [x] TurboQuant-inspired random-rotation quantization mechanism test for RAG embeddings
- [x] TurboVec optional smoke test
- [x] include RaBitQ, DRIVE/EDEN and 2026 critical comparisons in the lab bibliography

## v0.2 - Frozen real embeddings

Replace synthetic embeddings with frozen public encoders while keeping the same mathematical intervention and baseline visible.

- [x] add `example_hf.py` to every current lab using one frozen SentenceTransformer model
- [x] use short real phrases for retrieval, consistency, tuning-head, composition, OT and quantization experiments
- [x] keep Hugging Face dependencies optional so the NumPy mechanism tests remain lightweight
- [ ] add hidden states from a small open transformer for tuning diagnostics
- [ ] replace adapter-like semantic operators with real LoRA deltas for composition experiments
- [ ] run the real-sentence suite in a network-enabled CI/cache job


## v0.2.1 - Retrieval compute optimization

- [x] expand TurboVec integration beyond smoke testing
- [x] add optional low-bit ANN recall/latency/size benchmark
- [x] add TurboVec metadata/tenant allowlist experiment
- [x] add online-ingest and incremental-persistence experiment
- [x] add low-bit candidate generation plus exact float32 reranking experiment
- [x] add neighborhood-adaptive rerank optimizer with local geometric difficulty signals
- [x] add a frozen sentence/token embedding companion for adaptive reranking


## v0.2.2 - Combinatorial budget selection

- [x] add matroid-constrained RAG context selection
- [x] separate relevant-independent, relevant-dependent and unrelated candidates
- [x] reuse the same partition-matroid mechanism for LLM training-data budgets
- [x] add a frozen sentence-embedding companion with semantic partitions
- [x] connect the lab to neighborhood-adaptive reranking
- [ ] compare against MMR / DF-RAG on a public retrieval benchmark
- [ ] add a real LoRA fine-tuning experiment with partition-matroid minibatch selection



## v0.2.3 - Portable implementation architecture

- [x] define Python -> Rust -> wgpu/WGSL as the preferred implementation ladder
- [x] define CPU fallback and portable GPU paths for desktop and mobile
- [x] keep CUDA/Triton, Metal and Vulkan-specific kernels as optional measured fast paths
- [x] require AI-quality metrics and systems metrics for optimized implementations
- [ ] port the first proven retrieval bottleneck to a portable Rust core
- [ ] add the first wgpu/WGSL compute implementation
- [ ] compare portable GPU performance with a specialized backend on the same lab

## v0.3 - End-to-end small LLM tests

Compare interventions on compact open models under fixed compute budgets.

Metrics should include task score, latency, memory, trainable parameter count, forgetting, retrieval recall/nDCG, and robustness.

## Candidate labs

### High-dimensional probability / compression
- high-dimensional probability / data-oblivious quantization for changing RAG corpora
- quantized Johnson-Lindenstrauss inner-product correction
- random matrix / sketching methods for embedding indexes and KV caches

### Retrieval compute allocation
- learn/calibrate neighborhood difficulty thresholds on public retrieval benchmarks
- compare effective-rank proxies with local intrinsic dimensionality estimators
- combine neighborhood gating with diverse cluster representatives before a cross-encoder/LLM reranker

### Combinatorial optimization
- matroid-constrained context selection after reranking
- matroid-constrained minibatch/data selection for fine-tuning
- submodular information gain under matroid feasibility constraints
- source/language/tenant-aware independence constraints for RAG

### Spectral methods
- spectral diversification of retrieved context
- graph-Laplacian context compression
- attention-spectrum diagnostics

### Topology
- persistent homology as a detector of representation collapse during tuning
- cohomological inconsistency filters for multi-source RAG
- learned sheaf maps for context propagation

### Algebra
- Clifford-valued multimodal adapters
- structured operator algebras for compositional reasoning
- adapter commutator as a merge-risk predictor

### Geometry / dynamics
- local Riemannian metrics for query-dependent retrieval
- natural-gradient approximations for LoRA
- constrained or Hamiltonian updates for retention-aware tuning

### Noncommutative geometry
Only add a lab when it modifies an actual AI process. Candidate: define an operator-induced distance on frozen latent states and test whether changing the Dirac-like operator improves routing or retrieval without changing the encoder.
