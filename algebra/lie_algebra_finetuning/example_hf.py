"""Lie-algebra constrained adaptation over real frozen sentence embeddings.

Sentence embeddings are projected to a 2-D semantic plane.  A known small rotation
acts as the downstream target transformation; the experiment compares an unrestricted
2x2 adapter with a one-parameter SO(2) adapter on real language representations.
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
RNG = np.random.default_rng(3)

TRAIN_SENTENCES = [
    "RAG retrieves relevant documents for a language model.",
    "LoRA adapts a frozen language model with low-rank updates.",
    "Vector quantization reduces the memory used by embeddings.",
    "Attention combines information from multiple token positions.",
    "Adapters modify model behavior without full retraining.",
    "Embeddings place semantically related sentences near each other.",
    "A reranker reorders retrieval candidates before generation.",
    "Fine-tuning changes model parameters for a downstream task.",
]

VAL_SENTENCES = [
    "A vector database supports semantic nearest-neighbor search.",
    "Low-bit weights reduce model memory requirements.",
    "Retrieved context can contain contradictory evidence.",
    "Natural gradient uses the geometry of a statistical model.",
    "Prompt routing can send a request to a specialized expert.",
    "A frozen encoder can still feed a trainable classification head.",
]


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def semantic_plane(train_emb: np.ndarray, val_emb: np.ndarray):
    mean = train_emb.mean(axis=0, keepdims=True)
    centered = train_emb - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return centered @ basis, (val_emb - mean) @ basis


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def norm_distortion(before: np.ndarray, after: np.ndarray) -> float:
    return float(
        np.mean(np.abs(np.linalg.norm(after, axis=1) - np.linalg.norm(before, axis=1)))
    )


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    train_emb = model.encode(TRAIN_SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
    val_emb = model.encode(VAL_SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
    train_x, val_x = semantic_plane(train_emb, val_emb)

    true_angle = np.deg2rad(28.0)
    r_true = rotation(true_angle)
    train_y = train_x @ r_true.T + RNG.normal(scale=0.025, size=train_x.shape)
    val_y = val_x @ r_true.T

    # Baseline: unrestricted 2x2 adapter.
    w = np.eye(2)
    for _ in range(140):
        pred = train_x @ w.T
        grad_w = 2.0 * (pred - train_y).T @ train_x / len(train_x)
        w -= 0.12 * grad_w

    # Structured adapter: one parameter in so(2).
    theta = 0.0
    j = np.array([[0.0, -1.0], [1.0, 0.0]])
    for _ in range(140):
        r = rotation(theta)
        pred = train_x @ r.T
        dr = j @ r
        dpred = train_x @ dr.T
        grad_theta = 2.0 * np.mean(np.sum((pred - train_y) * dpred, axis=1))
        theta -= 0.25 * grad_theta

    val_free = val_x @ w.T
    r_lie = rotation(theta)
    val_lie = val_x @ r_lie.T

    print(f"model: {MODEL_NAME}")
    print(f"frozen sentence embedding dim: {train_emb.shape[1]}")
    print("PCA semantic plane dim: 2\n")

    print("Unconstrained adapter (4 params)")
    print(f"  validation MSE  = {mse(val_free, val_y):.8f}")
    print(f"  norm distortion = {norm_distortion(val_x, val_free):.8f}")

    print("\nLie adapter SO(2) (1 param)")
    print(f"  learned angle   = {np.rad2deg(theta):.4f} deg")
    print(f"  validation MSE  = {mse(val_lie, val_y):.8f}")
    print(f"  norm distortion = {norm_distortion(val_x, val_lie):.12f}")

    print(
        "\nNOTE: the language representations are real and frozen; the target rotation "
        "is synthetic so the lab isolates the structured-update mechanism."
    )


if __name__ == "__main__":
    main()
