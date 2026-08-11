"""Training-free optimal-transport reranker over frozen token embeddings."""

import numpy as np


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sinkhorn_cost(x, y, epsilon=0.08, iterations=200):
    """Balanced entropic OT with uniform token masses."""
    n, m = len(x), len(y)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    K = np.exp(-C / epsilon) + 1e-12
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(iterations):
        u = a / (K @ v)
        v = b / (K.T @ u)
    P = (u[:, None] * K) * v[None, :]
    return float(np.sum(P * C))


query = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])

documents = {
    # Correct token-level alignment.
    "relevant": np.array([[1.05, 0.02], [0.02, 0.95]]),
    # Mean vector is almost perfect, but both tokens collapse to the middle.
    "mean_decoy": np.array([[0.55, 0.55], [0.55, 0.55]]),
    "unrelated": np.array([[-0.7, 0.1], [0.0, -0.8]]),
}

q_mean = query.mean(axis=0)

cosine_ranking = sorted(
    documents,
    key=lambda n: cosine(q_mean, documents[n].mean(axis=0)),
    reverse=True,
)

ot_ranking = sorted(
    documents,
    key=lambda n: sinkhorn_cost(query, documents[n]),
)

print("Mean-pooled cosine ranking")
for name in cosine_ranking:
    print(f"  {name:12s} cosine={cosine(q_mean, documents[name].mean(axis=0)):.5f}")

print("\nOptimal-transport ranking")
for name in ot_ranking:
    print(f"  {name:12s} OT cost={sinkhorn_cost(query, documents[name]):.5f}")

print("\nRelevant rank")
print("  cosine:", cosine_ranking.index("relevant") + 1)
print("  OT    :", ot_ranking.index("relevant") + 1)
