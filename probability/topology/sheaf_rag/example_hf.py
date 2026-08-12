"""Sheaf-inspired RAG consistency over frozen sentence embeddings.

Real phrases are embedded with a frozen SentenceTransformer.  Relevance comes from
query/chunk cosine similarity; a one-dimensional local state is derived from
positive/negative semantic anchors, and local disagreement is penalized before the
context reaches a generator.
"""

from __future__ import annotations

from itertools import combinations
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        'Install the optional HF dependencies first: python -m pip install -e ".[hf]"'
    ) from exc

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 3
LAMBDA = 0.85


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def consistency_energy(names: tuple[str, ...] | list[str], states: dict[str, float]) -> float:
    values = np.array([states[name] for name in names], dtype=float)
    energy = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            energy += float((values[i] - values[j]) ** 2)
    return energy


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    query = "Can the device be used without an internet connection?"
    chunks = {
        "manual_A": "The device supports offline mode after the required files are downloaded.",
        "manual_B": "Downloaded projects can be opened and edited without an internet connection.",
        "contradiction": "The device cannot be used offline and requires a permanent internet connection.",
        "manual_D": "Cached content remains available when the network connection is unavailable.",
        "unrelated": "The device includes a high-resolution display and two USB ports.",
    }

    all_texts = [query, *chunks.values()]
    vectors = model.encode(all_texts, normalize_embeddings=True, convert_to_numpy=True)
    q = vectors[0]
    doc_vectors = dict(zip(chunks, vectors[1:], strict=True))

    positive_anchor, negative_anchor = model.encode(
        [
            "The product works without an internet connection.",
            "The product requires an internet connection at all times.",
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    relevance = {name: max(0.0, cosine(q, vector)) for name, vector in doc_vectors.items()}
    states = {
        name: cosine(vector, positive_anchor) - cosine(vector, negative_anchor)
        for name, vector in doc_vectors.items()
    }

    baseline = sorted(chunks, key=lambda name: relevance[name], reverse=True)[:K]

    candidates = []
    for subset in combinations(chunks.keys(), K):
        rel = sum(relevance[name] for name in subset)
        energy = consistency_energy(subset, states)
        objective = rel - LAMBDA * energy
        candidates.append((objective, subset, rel, energy))
    structured = max(candidates, key=lambda row: row[0])

    print(f"model: {MODEL_NAME}")
    print(f"query: {query}\n")
    print("Retrieved chunks")
    for name in sorted(chunks, key=lambda n: relevance[n], reverse=True):
        print(
            f"  {name:14s} relevance={relevance[name]:.4f} "
            f"local_state={states[name]:+.4f}"
        )
        print(f"    {chunks[name]}")

    print("\nBaseline top-k")
    print("  context   =", baseline)
    print("  relevance =", round(sum(relevance[n] for n in baseline), 4))
    print("  energy    =", round(consistency_energy(baseline, states), 4))

    print("\nSheaf-inspired selection")
    print("  context   =", list(structured[1]))
    print("  relevance =", round(structured[2], 4))
    print("  energy    =", round(structured[3], 4))
    print("  objective =", round(structured[0], 4))


if __name__ == "__main__":
    main()
