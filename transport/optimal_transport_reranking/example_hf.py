"""Optimal-transport reranking over real frozen token embeddings.

The baseline mean-pools contextual token embeddings and ranks with cosine.  The
intervention keeps each sentence as a token distribution and reranks with an
entropically regularized OT cost.
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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x), 1e-12, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalize(a) @ normalize(b))


def sinkhorn_cost(x: np.ndarray, y: np.ndarray, epsilon: float = 0.09, iterations: int = 120) -> float:
    """Balanced entropic OT using cosine distance between contextual token vectors."""
    x = normalize_rows(x)
    y = normalize_rows(y)
    n, m = len(x), len(y)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    cost = 1.0 - np.clip(x @ y.T, -1.0, 1.0)
    kernel = np.exp(-cost / epsilon) + 1e-12
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(iterations):
        u = a / np.clip(kernel @ v, 1e-12, None)
        v = b / np.clip(kernel.T @ u, 1e-12, None)
    plan = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(plan * cost))


def token_embeddings(model: SentenceTransformer, texts: list[str]) -> list[np.ndarray]:
    encoded = model.encode(
        texts,
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    result = []
    for tokens in encoded:
        if hasattr(tokens, "detach"):
            arr = tokens.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            arr = np.asarray(tokens, dtype=np.float64)
        # MiniLM sequences include special boundary tokens; remove them when possible.
        if len(arr) > 2:
            arr = arr[1:-1]
        result.append(arr)
    return result


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    query = "How does low-bit vector quantization reduce memory in a RAG embedding index?"
    documents = {
        "relevant": (
            "Low-bit quantization compresses document embeddings, reducing the memory "
            "used by a RAG vector index while attempting to preserve retrieval recall."
        ),
        "semantic_decoy": (
            "RAG systems use vector databases and semantic embeddings to retrieve context, "
            "while numerical quantization is also used in machine learning."
        ),
        "model_quantization": (
            "Quantizing neural network weights can reduce the memory required to run a language model."
        ),
        "unrelated": "Optical resonators use mirrors to confine laser light inside a cavity.",
    }

    texts = [query, *documents.values()]
    token_sets = token_embeddings(model, texts)
    q_tokens = token_sets[0]
    doc_tokens = dict(zip(documents, token_sets[1:], strict=True))

    q_mean = q_tokens.mean(axis=0)
    cosine_ranking = sorted(
        documents,
        key=lambda name: cosine(q_mean, doc_tokens[name].mean(axis=0)),
        reverse=True,
    )
    ot_ranking = sorted(documents, key=lambda name: sinkhorn_cost(q_tokens, doc_tokens[name]))

    print(f"model: {MODEL_NAME}")
    print(f"query tokens used: {len(q_tokens)}\n")
    print("Mean-token cosine ranking")
    for name in cosine_ranking:
        score = cosine(q_mean, doc_tokens[name].mean(axis=0))
        print(f"  {name:16s} cosine={score:.4f}")

    print("\nOptimal-transport ranking")
    for name in ot_ranking:
        score = sinkhorn_cost(q_tokens, doc_tokens[name])
        print(f"  {name:16s} OT cost={score:.4f}")

    print("\nKnown relevant rank")
    print("  cosine:", cosine_ranking.index("relevant") + 1)
    print("  OT    :", ot_ranking.index("relevant") + 1)


if __name__ == "__main__":
    main()
