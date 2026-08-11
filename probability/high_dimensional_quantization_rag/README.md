# High-Dimensional Quantization for RAG

## AI process

Compress and search the frozen embedding index used by a RAG retriever, with special interest in **online ingest** where a corpus changes continuously and retraining a quantizer/codebook is undesirable.

A second application from the same mathematical family is LLM KV-cache compression, but the runnable lab here focuses on retrieval so it stays comparable with the other RAG experiments in this repository.

## Baseline

The controlled baseline is cosine retrieval over normalized `float32` vectors. The low-bit baseline directly scalar-quantizes the same vectors in their original coordinate basis.

In a production benchmark this should be extended to trained Product Quantization (for example FAISS PQ/FastScan), RaBitQ and other strong ANN baselines at matched memory budgets.

## Mathematical intervention

The lab isolates a key idea behind TurboQuant:

1. normalize each embedding onto the unit hypersphere;
2. apply one shared random orthogonal rotation;
3. exploit high-dimensional concentration, where rotated coordinates have a predictable scale around `1/sqrt(d)`;
4. quantize those coordinates with a fixed low-bit scalar quantizer.

The important AI-engineering idea is that the quantizer can be **data-oblivious**: the mathematical transformation makes coordinate statistics predictable enough that the indexing procedure does not need a corpus-specific codebook-training phase.

TurboQuant itself is more sophisticated than `example.py`: it derives distribution-aware Lloyd-Max codebooks and, for inner products, adds a residual Quantized Johnson-Lindenstrauss correction. TurboVec then turns the algorithm into a Rust/Python vector index with bit packing, SIMD kernels, persistence and filtered search.

## Hypothesis

For anisotropic frozen embeddings, a random orthogonal basis can spread coordinate energy enough that a fixed low-bit quantizer preserves retrieval neighborhoods substantially better than applying the same bit budget naively in the original basis.

This lab does **not** claim that TurboQuant is universally superior to PQ, RaBitQ or EDEN. The 2026 literature contains direct challenges to both its novelty framing and some reported empirical advantages, so those comparisons are part of the next benchmark rather than assumptions baked into this repository.

## Related papers and implementations

1. **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** — Amir Zandieh, Majid Daliri, Majid Hadian & Vahab Mirrokni, ICLR 2026 / OpenReview. The main intervention: random rotations make coordinates follow a predictable high-dimensional distribution, enabling data-oblivious scalar quantization; the inner-product variant adds a one-bit QJL residual correction.  
   https://openreview.net/forum?id=tO3ASKZlok

2. **TurboVec** — RyanCodrai, open-source Rust vector index with Python bindings implementing TurboQuant for vector search. This is the practical implementation that motivated adding the lab. Its published README reports online ingest without a separate training step, low-bit storage, SIMD kernels and filtered search. Those performance numbers are project claims until independently reproduced here.  
   https://github.com/RyanCodrai/turbovec

3. **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search** — Jianyang Gao & Cheng Long, SIGMOD 2024. Important strong baseline based on random transformations and theoretically controlled low-bit ANN distance estimation.  
   https://arxiv.org/abs/2405.12497

4. **DRIVE: One-bit Distributed Mean Estimation** — Vargaftik et al., NeurIPS 2021. Earlier use of random rotations plus aggressive scalar/sign quantization in a machine-learning communication problem. It matters historically because later 2026 discussion argues that TurboQuant overlaps materially with the DRIVE/EDEN line.  
   https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html

5. **EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning** — Vargaftik et al., ICML 2022. Generalizes the rotation/quantization framework to flexible bit budgets and is a key antecedent for evaluating what is genuinely new in TurboQuant.  
   https://proceedings.mlr.press/v162/vargaftik22a.html

6. **Revisiting RaBitQ and TurboQuant: A Symmetric Comparison of Methods, Theory, and Experiments** — Gao et al., 2026 technical note. Provides a direct critical comparison and reports that TurboQuant does not consistently outperform RaBitQ under matched settings, while also documenting reproduction issues for some TurboQuant results.  
   https://arxiv.org/abs/2604.19528

7. **A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work** — Ben-Basat et al., 2026 technical note. Argues that TurboQuant's MSE construction is closely related to the earlier DRIVE/EDEN framework and reports stronger EDEN results in the authors' comparisons. This is included so the lab tests the algorithm rather than inheriting a novelty claim.  
   https://arxiv.org/abs/2604.18555

8. **TurboVec: A Case Study in Cost-Efficient Private Retrieval for Enterprise RAG via Codebook-Oblivious Quantization** — Shukla, Pandey & Tiwari, 2026. Recent RAG-specific case study of TurboVec, including memory/recall/latency and filtered-retrieval experiments. Useful as a target for later reproduction, but currently limited to a narrow evaluation setting.  
   https://arxiv.org/abs/2607.16973

## Experiment

`example.py` creates a deterministic frozen retrieval corpus with anisotropic topic directions. It compares:

- exact `float32` cosine retrieval;
- direct 2-bit scalar quantization in the original basis;
- a **TurboQuant-inspired** 2-bit path using one random orthogonal rotation and a tighter distribution-informed coordinate range.

Run:

```bash
python probability/high_dimensional_quantization_rag/example.py
```

There is also an optional smoke test against the actual public TurboVec package:

```bash
pip install turbovec
python probability/high_dimensional_quantization_rag/turbovec_smoke.py
```

The optional script is not part of `run_all.py`, because TurboVec ships compiled Rust/Python bindings and is intentionally not a mandatory dependency of the repository.

## Real sentence-embedding experiment

The real-sentence version builds a small multi-topic corpus, embeds every sentence once, and compares float32 retrieval, naive 2-bit scalar quantization, and random-rotation 2-bit quantization.

The encoder is **frozen**. The purpose is to move from hand-written/random arrays to actual language representations without introducing model training as a confounder.

Install the optional dependency once from the repository root:

```bash
python -m pip install -e ".[hf]"
```

Run:

```bash
python probability/high_dimensional_quantization_rag/example_hf.py
```

Embedding reference:

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** - Reimers & Gurevych, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
- `sentence-transformers/all-MiniLM-L6-v2` model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Metrics

- Recall@1 / Recall@5 / Recall@10 against known topic labels
- reconstruction MSE
- theoretical vector payload at `float32` versus packed 2-bit codes plus one stored norm
- later: index build/training time, query latency, RSS, nDCG, RAG answer quality and incremental-ingest cost

## Observed result

The deterministic mechanism test currently gives:

```text
Recall@1   float32=1.000  naive-2bit=0.210  rotated-2bit=1.000
Recall@5   float32=1.000  naive-2bit=0.510  rotated-2bit=1.000
Recall@10  float32=1.000  naive-2bit=0.710  rotated-2bit=1.000

reconstruction MSE naive   ~= 0.00546
reconstruction MSE rotated ~= 0.00206
```

The theoretical packed payload in this toy drops from 1000 KiB of float32 vector data to about 70.3 KiB for 2-bit coordinate codes plus one float norm per vector, about **14.2x** smaller. The Python toy does not actually bit-pack its NumPy arrays; the payload metric represents what such a packed index would store.

## Interpretation

The result supports only the local mechanism: when embeddings are strongly anisotropic, changing to a random orthogonal basis can make a very low-bit, data-oblivious scalar quantizer far less destructive to retrieval neighborhoods.

It is **not evidence that this simplified quantizer matches TurboQuant**, and it does not establish superiority over PQ, RaBitQ, EDEN or production ANN indexes. Those require matched-bit, matched-latency benchmarks on real embeddings.

## Next experiment

Use one frozen public embedding model and one public retrieval dataset, then compare at equal memory budgets:

1. exact float32 cosine;
2. FAISS Product Quantization / FastScan;
3. RaBitQ;
4. TurboVec 2-bit and 4-bit;
5. optionally EDEN-style quantization where a suitable ANN implementation is available.

Record Recall@k, nDCG, index-training/build time, online insertion cost, wall-clock query latency, RSS and final RAG answer accuracy. The purpose is specifically to test whether the **mathematical data-obliviousness** provides a practical advantage in changing RAG corpora.
