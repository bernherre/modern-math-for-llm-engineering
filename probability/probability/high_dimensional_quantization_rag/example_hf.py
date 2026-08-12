"""TurboQuant-inspired quantization over real frozen sentence embeddings.

The corpus consists of simple English phrases encoded by a frozen SentenceTransformer.
We compare full precision, naive 2-bit scalar quantization, and a data-oblivious random
orthogonal rotation followed by a tighter 2-bit quantizer.

This is not an implementation or benchmark reproduction of TurboQuant/TurboVec.
"""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        'Install the optional HF dependencies first: python -m pip install -e ".[hf]"'
    ) from exc

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
BITS = 2
LEVELS = 2**BITS

TOPICS = {
    "rag": [
        "RAG retrieves passages before a language model generates an answer.",
        "Retrieved context supplies external facts to the generator.",
        "A retriever searches a corpus for passages relevant to a user query.",
        "RAG combines information retrieval with language model generation.",
        "Context retrieval can improve factual answers when the corpus contains evidence.",
    ],
    "quantization": [
        "Low-bit quantization compresses embedding vectors to reduce memory.",
        "Vector quantization stores approximate coordinates with fewer bits.",
        "Quantized indexes trade small numerical error for lower memory use.",
        "Random rotations can spread vector energy before scalar quantization.",
        "Embedding compression can make nearest-neighbor indexes smaller.",
    ],
    "finetuning": [
        "Fine-tuning adapts model parameters to a downstream task.",
        "LoRA learns low-rank parameter updates for a frozen language model.",
        "Parameter-efficient tuning changes a small fraction of model parameters.",
        "Adapters specialize a pretrained model without full retraining.",
        "A downstream dataset can be used to adapt a foundation model.",
    ],
    "databases": [
        "A vector database performs approximate nearest-neighbor search.",
        "Database indexes accelerate search over large collections.",
        "Semantic search retrieves items using embedding similarity.",
        "An ANN index returns vectors close to a query vector.",
        "Vector stores keep embeddings together with document metadata.",
    ],
    "optics": [
        "A laser cavity uses mirrors to sustain optical resonance.",
        "A lens focuses light according to its curvature and refractive index.",
        "Interference occurs when coherent optical waves overlap.",
        "A diffraction grating separates light by wavelength.",
        "Photon detectors convert incoming light into electrical signals.",
    ],
    "thermodynamics": [
        "Entropy measures the number of accessible microscopic configurations.",
        "Heat flows spontaneously from hotter systems to colder systems.",
        "Free energy determines whether a thermodynamic process is favorable.",
        "Temperature is related to the distribution of microscopic energy.",
        "An isolated system approaches thermal equilibrium over time.",
    ],
}

QUERIES = {
    "rag": "How does retrieval augmented generation provide evidence to a language model?",
    "quantization": "How can low-bit quantization shrink an embedding index?",
    "finetuning": "How can a pretrained language model be adapted with only a few parameters?",
    "databases": "How does a vector database search for semantically similar items?",
    "optics": "How does an optical cavity maintain laser resonance?",
    "thermodynamics": "Why does heat move toward thermal equilibrium?",
}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def quantize_uniform(x: np.ndarray, low: float, high: float) -> np.ndarray:
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
    hits = [np.any(doc_labels[topk[i]] == query_labels[i]) for i in range(len(queries))]
    return float(np.mean(hits))


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    rng = np.random.default_rng(SEED)

    docs = []
    doc_labels = []
    for topic, sentences in TOPICS.items():
        docs.extend(sentences)
        doc_labels.extend([topic] * len(sentences))

    query_texts = [QUERIES[topic] for topic in TOPICS]
    query_labels = np.array(list(TOPICS))
    doc_labels_arr = np.array(doc_labels)

    doc_vectors = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    query_vectors = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
    dim = doc_vectors.shape[1]

    exact_docs = normalize(doc_vectors.astype(np.float64))
    exact_queries = normalize(query_vectors.astype(np.float64))

    naive_docs = normalize(quantize_uniform(exact_docs, -1.0, 1.0))

    gaussian = rng.normal(size=(dim, dim))
    rotation, _ = np.linalg.qr(gaussian)
    rotated_docs = exact_docs @ rotation
    rotated_queries = exact_queries @ rotation
    clip = 3.0 / np.sqrt(dim)
    rotated_quantized = normalize(quantize_uniform(rotated_docs, -clip, clip))
    reconstructed = rotated_quantized @ rotation.T

    print(f"model: {MODEL_NAME}")
    print(f"documents={len(docs)}, queries={len(query_texts)}, dim={dim}, bits={BITS}\n")
    for k in (1, 3, 5):
        exact = recall_at_k(exact_queries, exact_docs, doc_labels_arr, query_labels, k)
        naive = recall_at_k(exact_queries, naive_docs, doc_labels_arr, query_labels, k)
        rotated = recall_at_k(
            rotated_queries,
            rotated_quantized,
            doc_labels_arr,
            query_labels,
            k,
        )
        print(
            f"Recall@{k:<2} float32={exact:.3f} "
            f"naive-{BITS}bit={naive:.3f} rotated-{BITS}bit={rotated:.3f}"
        )

    naive_mse = float(np.mean((exact_docs - naive_docs) ** 2))
    rotated_mse = float(np.mean((exact_docs - reconstructed) ** 2))
    float_bytes = len(docs) * dim * 4
    packed_bytes = len(docs) * dim * BITS // 8

    print(f"\nMSE naive   = {naive_mse:.6f}")
    print(f"MSE rotated = {rotated_mse:.6f}")
    print(f"theoretical vector payload: {float_bytes / 1024:.1f} KiB -> {packed_bytes / 1024:.1f} KiB")
    print(f"code-only compression ratio: {float_bytes / packed_bytes:.1f}x")
    print(
        "\nNOTE: the embeddings and phrases are real; the quantizer is intentionally "
        "simplified and should not be reported as TurboQuant/TurboVec performance."
    )


if __name__ == "__main__":
    main()
