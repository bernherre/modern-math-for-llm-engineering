"""TurboQuant-inspired toy for RAG vector-index compression.

This is intentionally NOT a reimplementation of TurboQuant or TurboVec.
It isolates one mechanism from the paper:

    normalize -> random orthogonal rotation -> low-bit scalar quantization

and compares it with direct low-bit scalar quantization on the same frozen
retrieval embeddings.

The synthetic corpus is deliberately anisotropic/sparse so that the effect of
spreading energy across coordinates is visible with a small deterministic test.
"""

from __future__ import annotations

import numpy as np

SEED = 42
D = 128
N_TOPICS = 100
DOCS_PER_TOPIC = 20
BITS = 2
LEVELS = 2**BITS


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_corpus(rng: np.random.Generator):
    """Create frozen, clustered embeddings with anisotropic topic directions."""
    centers = np.zeros((N_TOPICS, D), dtype=np.float64)

    for topic in range(N_TOPICS):
        idx = rng.choice(D, size=4, replace=False)
        centers[topic, idx] = rng.normal(size=4)

    centers = normalize(centers)

    docs = []
    labels = []
    for topic, center in enumerate(centers):
        for _ in range(DOCS_PER_TOPIC):
            docs.append(normalize(center + 0.08 * rng.normal(size=D)))
            labels.append(topic)

    queries = []
    query_labels = []
    for topic, center in enumerate(centers):
        queries.append(normalize(center + 0.10 * rng.normal(size=D)))
        query_labels.append(topic)

    return (
        np.asarray(docs),
        np.asarray(labels),
        np.asarray(queries),
        np.asarray(query_labels),
    )


def quantize_uniform(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Dataset-oblivious fixed scalar quantizer."""
    grid = np.linspace(low, high, LEVELS, dtype=np.float64)
    indices = np.argmin(np.abs(x[..., None] - grid), axis=-1)
    return grid[indices]


def recall_at_k(
    queries: np.ndarray,
    docs: np.ndarray,
    doc_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> float:
    scores = queries @ docs.T
    topk = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(
        np.mean(
            [
                np.any(doc_labels[topk[i]] == query_labels[i])
                for i in range(len(queries))
            ]
        )
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    docs, doc_labels, queries, query_labels = make_corpus(rng)

    # Full-precision retrieval baseline.
    exact_docs = normalize(docs)
    exact_queries = normalize(queries)

    # Naive 2-bit scalar quantization in the original coordinate basis.
    naive_docs = normalize(quantize_uniform(exact_docs, -1.0, 1.0))

    # TurboQuant-inspired intervention:
    # use a fixed random orthogonal basis, then exploit concentration around
    # O(1/sqrt(d)) with a tighter data-oblivious scalar range.
    gaussian = rng.normal(size=(D, D))
    rotation, _ = np.linalg.qr(gaussian)

    rotated_docs = exact_docs @ rotation
    rotated_queries = exact_queries @ rotation

    clip = 3.0 / np.sqrt(D)
    rotated_quantized = normalize(
        quantize_uniform(rotated_docs, -clip, clip)
    )

    # Rotate reconstructed vectors back only for an interpretable MSE metric.
    rotated_reconstructed = rotated_quantized @ rotation.T

    full_precision_bytes = docs.shape[0] * D * 4
    packed_2bit_bytes = docs.shape[0] * D * BITS // 8
    norm_bytes = docs.shape[0] * 4

    print("High-dimensional quantization for RAG retrieval")
    print(f"documents={len(docs)}, dim={D}, bits={BITS}")
    print()

    for k in (1, 5, 10):
        exact = recall_at_k(
            exact_queries, exact_docs, doc_labels, query_labels, k
        )
        naive = recall_at_k(
            exact_queries, naive_docs, doc_labels, query_labels, k
        )
        rotated = recall_at_k(
            rotated_queries,
            rotated_quantized,
            doc_labels,
            query_labels,
            k,
        )

        print(
            f"Recall@{k:<2}  float32={exact:.3f}  "
            f"naive-{BITS}bit={naive:.3f}  rotated-{BITS}bit={rotated:.3f}"
        )

    naive_mse = float(np.mean((exact_docs - naive_docs) ** 2))
    rotated_mse = float(
        np.mean((exact_docs - rotated_reconstructed) ** 2)
    )

    print()
    print(f"reconstruction MSE naive   : {naive_mse:.6f}")
    print(f"reconstruction MSE rotated : {rotated_mse:.6f}")
    print()
    print(f"theoretical float32 payload : {full_precision_bytes / 1024:.1f} KiB")
    print(
        "theoretical 2-bit payload  : "
        f"{(packed_2bit_bytes + norm_bytes) / 1024:.1f} KiB "
        "(packed codes + one float norm/vector)"
    )
    print(
        "compression ratio           : "
        f"{full_precision_bytes / (packed_2bit_bytes + norm_bytes):.2f}x"
    )
    print()
    print(
        "NOTE: this toy demonstrates the rotation/concentration mechanism only. "
        "TurboQuant adds distribution-optimized codebooks and an inner-product "
        "correction; TurboVec adds a production Rust/SIMD index."
    )


if __name__ == "__main__":
    main()
