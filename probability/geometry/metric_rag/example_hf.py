"""Metric-conditioned RAG retrieval over real frozen sentence embeddings.

This is the language-data companion to example.py.  It keeps the encoder frozen
and compares ordinary cosine retrieval with a low-rank Mahalanobis metric built
from two task anchors.
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


def normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def metric_distance(
    a: np.ndarray,
    b: np.ndarray,
    axes: list[np.ndarray],
    weights: list[float],
    isotropic_weight: float = 0.15,
) -> float:
    """Low-rank Mahalanobis distance without materializing a 384x384 matrix."""
    delta = a - b
    value = isotropic_weight * float(delta @ delta)
    for axis, weight in zip(axes, weights, strict=True):
        value += weight * float(delta @ axis) ** 2
    return float(np.sqrt(value))


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranking[:k]) & relevant) / len(relevant)


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    query = (
        "How can low-bit vector quantization reduce RAG index memory while "
        "preserving nearest-neighbor retrieval quality?"
    )
    chunks = {
        "rotation_mechanism": (
            "Random orthogonal rotations spread embedding energy across coordinates "
            "before low-bit quantization, helping preserve similarity search."
        ),
        "memory_effect": (
            "Compressing frozen document embeddings to a few bits reduces vector-index "
            "memory while aiming to keep nearest-neighbor recall high."
        ),
        "generic_quantization": (
            "Quantization maps continuous numerical values to a smaller set of discrete levels."
        ),
        "generic_rag": (
            "Retrieval-augmented generation retrieves documents and adds them to the "
            "language model context."
        ),
        "unrelated": (
            "A laser cavity uses mirrors to sustain optical resonance at selected wavelengths."
        ),
    }
    relevant = {"rotation_mechanism", "memory_effect"}

    texts = [query, *chunks.values()]
    embedded = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    q = embedded[0]
    doc_vectors = dict(zip(chunks, embedded[1:], strict=True))

    # Task anchors define directions that should matter more for this retrieval task.
    anchors = model.encode(
        [
            "random rotations preserve vector similarity during low-bit quantization",
            "reduce vector index memory while preserving retrieval recall",
            "a generic definition with no implementation or operational consequence",
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    mechanism_axis = normalize(anchors[0] - anchors[2])
    operational_axis = normalize(anchors[1] - anchors[2])

    cosine_ranking = sorted(
        chunks,
        key=lambda name: cosine(q, doc_vectors[name]),
        reverse=True,
    )
    metric_ranking = sorted(
        chunks,
        key=lambda name: metric_distance(
            q,
            doc_vectors[name],
            [mechanism_axis, operational_axis],
            [5.0, 5.0],
        ),
    )

    print(f"model: {MODEL_NAME}")
    print(f"query: {query}\n")
    print("Cosine ranking")
    for name in cosine_ranking:
        print(f"  {name:22s} cosine={cosine(q, doc_vectors[name]):.4f}")

    print("\nTask-conditioned metric ranking")
    for name in metric_ranking:
        distance = metric_distance(
            q,
            doc_vectors[name],
            [mechanism_axis, operational_axis],
            [5.0, 5.0],
        )
        print(f"  {name:22s} distance={distance:.4f}")

    print("\nRecall")
    for k in (2, 3):
        print(
            f"  Recall@{k}: cosine={recall_at_k(cosine_ranking, relevant, k):.2f} "
            f"metric={recall_at_k(metric_ranking, relevant, k):.2f}"
        )


if __name__ == "__main__":
    main()
