"""Order-sensitive adapter-like operators derived from sentence embeddings.

Two rank-1 transformations are constructed from frozen semantic directions:
A moves technical statements toward concise phrasing; B moves them toward cautious
phrasing.  Their commutator measures whether composition order can be ignored.
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

BASE_CONCISE_PAIRS = [
    (
        "The vector database stores a large collection of document embeddings for semantic retrieval.",
        "The vector database stores embeddings for semantic retrieval.",
    ),
    (
        "The retriever obtains a collection of candidate passages before the language model generates an answer.",
        "The retriever gets candidate passages before generation.",
    ),
    (
        "Low-bit quantization can reduce the amount of memory required by the embedding index.",
        "Low-bit quantization can reduce index memory.",
    ),
]

BASE_CAUTIOUS_PAIRS = [
    (
        "The reranker improves retrieval quality.",
        "The reranker may improve retrieval quality depending on the data.",
    ),
    (
        "Quantization preserves nearest-neighbor recall.",
        "Quantization can preserve nearest-neighbor recall under suitable settings.",
    ),
    (
        "The adapter improves the model.",
        "The adapter may improve the model for some downstream tasks.",
    ),
]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x), 1e-12, None)


def build_rank1_operator(
    source: np.ndarray,
    target: np.ndarray,
    strength: float = 0.55,
) -> np.ndarray:
    """Create I + alpha*u*v^T from an empirical semantic shift."""
    direction = normalize(np.mean(target - source, axis=0))
    activation = normalize(np.mean(source, axis=0))
    return np.eye(source.shape[1]) + strength * np.outer(direction, activation)


def nearest_label(vector: np.ndarray, candidates: dict[str, np.ndarray]) -> tuple[str, float]:
    vector = normalize(vector)
    scored = [(name, float(vector @ normalize(candidate))) for name, candidate in candidates.items()]
    return max(scored, key=lambda row: row[1])


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    a_source = model.encode([x for x, _ in BASE_CONCISE_PAIRS], normalize_embeddings=True)
    a_target = model.encode([y for _, y in BASE_CONCISE_PAIRS], normalize_embeddings=True)
    b_source = model.encode([x for x, _ in BASE_CAUTIOUS_PAIRS], normalize_embeddings=True)
    b_target = model.encode([y for _, y in BASE_CAUTIOUS_PAIRS], normalize_embeddings=True)

    a = build_rank1_operator(np.asarray(a_source), np.asarray(a_target))
    b = build_rank1_operator(np.asarray(b_source), np.asarray(b_target))

    sentence = "The new retrieval method reduces memory and improves answer quality."
    candidate_texts = {
        "base": sentence,
        "concise": "The retrieval method reduces memory and improves quality.",
        "cautious": "The retrieval method may reduce memory and may improve answer quality.",
        "concise_cautious": "The retrieval method may reduce memory and improve quality in some settings.",
    }
    vectors = model.encode([sentence, *candidate_texts.values()], normalize_embeddings=True)
    x = np.asarray(vectors[0])
    candidates = dict(zip(candidate_texts, vectors[1:], strict=True))

    # Row-vector convention: x @ A.T applies A.
    ab = x @ a.T @ b.T
    ba = x @ b.T @ a.T
    commutator = a @ b - b @ a

    ab_label, ab_score = nearest_label(ab, candidates)
    ba_label, ba_score = nearest_label(ba, candidates)

    print(f"model: {MODEL_NAME}")
    print(f"embedding dim: {len(x)}")
    print(f"input: {sentence}\n")
    print(f"||AB - BA||_F = {np.linalg.norm(commutator):.6f}")
    print(f"cosine(A->B, B->A) = {normalize(ab) @ normalize(ba):.6f}")
    print("\nNearest phrase after composition")
    print(f"  A -> B: {ab_label:18s} cosine={ab_score:.4f}")
    print(f"  B -> A: {ba_label:18s} cosine={ba_score:.4f}")
    print(
        "\nNOTE: these are adapter-like rank-1 operators inferred from frozen semantic "
        "directions, not trained LoRA modules. The lab isolates order sensitivity."
    )


if __name__ == "__main__":
    main()
