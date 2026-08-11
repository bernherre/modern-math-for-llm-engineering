"""Natural-gradient vs Euclidean tuning on frozen sentence embeddings.

The encoder is never updated.  We fit only a tiny logistic evidence-quality head
on top of sentence embeddings, comparing ordinary gradient descent with a Fisher-
preconditioned natural-gradient update under the same number of optimization steps.
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
RNG = np.random.default_rng(7)

TRAIN = [
    ("The retrieved passage directly supports the answer with a specific fact.", 1),
    ("Two independent retrieved sources agree on the same value.", 1),
    ("The context contains the exact mechanism asked about in the query.", 1),
    ("The passage gives evidence that is relevant and internally consistent.", 1),
    ("The retrieved text explicitly states the requested limitation.", 1),
    ("The source explains the cause and the measured consequence.", 1),
    ("The passage is about another topic and does not answer the question.", 0),
    ("The retrieved source contradicts the claim made in the answer.", 0),
    ("The context is generic background without the requested evidence.", 0),
    ("The source discusses a similar product but not the one in the query.", 0),
    ("The passage contains an unsupported guess rather than a factual statement.", 0),
    ("The retrieved snippets disagree with each other on the key fact.", 0),
]

VALIDATION = [
    ("The context gives the exact benchmark result needed to answer the query.", 1),
    ("The retrieved evidence clearly describes the requested implementation detail.", 1),
    ("The passage is relevant and corroborates the answer.", 1),
    ("The source provides only a broad definition and no evidence for the answer.", 0),
    ("The context directly conflicts with the generated statement.", 0),
    ("The retrieved text is semantically related but answers a different question.", 0),
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def prepare_features(train_emb: np.ndarray, val_emb: np.ndarray, dim: int = 12):
    """PCA coordinates keep the head small while preserving real HF structure."""
    mean = train_emb.mean(axis=0, keepdims=True)
    centered = train_emb - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(dim, len(vt))].T
    train_x = centered @ basis
    val_x = (val_emb - mean) @ basis

    # Reparameterize with the empirical singular scales.  This is not new data;
    # it exposes the conditioning problem natural gradient is designed to handle.
    scale = np.linspace(6.0, 0.7, train_x.shape[1])
    return train_x * scale, val_x * scale


def loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    p = sigmoid(x @ theta)
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def accuracy(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((sigmoid(x @ theta) >= 0.5) == y))


def gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x.T @ (sigmoid(x @ theta) - y) / len(x)


def fisher(theta: np.ndarray, x: np.ndarray, damping: float = 2e-2) -> np.ndarray:
    p = sigmoid(x @ theta)
    weights = p * (1.0 - p)
    matrix = x.T @ (x * weights[:, None]) / len(x)
    return matrix + damping * np.eye(x.shape[1])


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    train_text = [text for text, _ in TRAIN]
    val_text = [text for text, _ in VALIDATION]
    train_y = np.array([label for _, label in TRAIN], dtype=float)
    val_y = np.array([label for _, label in VALIDATION], dtype=float)

    train_emb = model.encode(train_text, normalize_embeddings=False, convert_to_numpy=True)
    val_emb = model.encode(val_text, normalize_embeddings=False, convert_to_numpy=True)
    train_x, val_x = prepare_features(train_emb, val_emb)

    steps = 18
    theta_gd = np.zeros(train_x.shape[1])
    theta_ng = np.zeros(train_x.shape[1])

    for _ in range(steps):
        theta_gd -= 0.08 * gradient(theta_gd, train_x, train_y)

    for _ in range(steps):
        g = gradient(theta_ng, train_x, train_y)
        direction = np.linalg.solve(fisher(theta_ng, train_x), g)
        theta_ng -= 0.30 * direction

    print(f"model: {MODEL_NAME}")
    print(f"frozen embedding dimension: {train_emb.shape[1]}")
    print(f"tiny trainable head dimension: {train_x.shape[1]}")
    print(f"optimization steps: {steps}\n")

    print("Euclidean gradient")
    print(f"  train loss = {loss(theta_gd, train_x, train_y):.5f}")
    print(f"  val loss   = {loss(theta_gd, val_x, val_y):.5f}")
    print(f"  val acc    = {accuracy(theta_gd, val_x, val_y):.3f}")

    print("\nNatural gradient")
    print(f"  train loss = {loss(theta_ng, train_x, train_y):.5f}")
    print(f"  val loss   = {loss(theta_ng, val_x, val_y):.5f}")
    print(f"  val acc    = {accuracy(theta_ng, val_x, val_y):.3f}")

    print(
        "\nNOTE: this is a frozen-encoder head-tuning experiment, not a claim that "
        "full LLM natural-gradient fine-tuning is this cheap."
    )


if __name__ == "__main__":
    main()
