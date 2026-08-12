"""Optional TurboVec candidate generation followed by exact float32 reranking.

The experiment asks whether very aggressive low-bit ANN can be sufficient as a
candidate generator even when its direct top-1 ranking is imperfect.

Install separately:
    pip install turbovec
"""

from __future__ import annotations

import numpy as np

try:
    from turbovec import TurboQuantIndex
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("TurboVec is optional. Install it with: pip install turbovec") from exc


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    rng = np.random.default_rng(29)
    n = 8000
    dim = 128
    n_queries = 120

    # Anisotropic clustered vectors make the low-bit ranking nontrivial.
    centers = normalize(rng.normal(size=(48, dim)).astype(np.float32))
    labels = rng.integers(0, len(centers), size=n)
    vectors = centers[labels] + 0.22 * rng.normal(size=(n, dim)).astype(np.float32)
    vectors = normalize(vectors).astype(np.float32)

    anchors = rng.choice(n, size=n_queries, replace=False)
    queries = vectors[anchors] + 0.045 * rng.normal(size=(n_queries, dim)).astype(np.float32)
    queries = normalize(queries).astype(np.float32)

    exact_scores = queries @ vectors.T
    exact_top1 = np.argmax(exact_scores, axis=1)

    index = TurboQuantIndex(dim=dim, bit_width=2)
    index.add(vectors)

    _, direct = index.search(queries, k=1)
    direct = np.asarray(direct, dtype=np.int64)[:, 0]
    direct_acc = float(np.mean(direct == exact_top1))

    print("candidate-k  gt-in-candidates  exact-rerank-top1")
    print("-----------  ----------------  -----------------")
    print(f"direct top1  {'-':>16s}  {direct_acc:17.3f}")

    for candidate_k in (4, 8, 16, 32):
        _, candidates = index.search(queries, k=candidate_k)
        candidates = np.asarray(candidates, dtype=np.int64)
        contains = np.array([
            exact_top1[i] in set(map(int, candidates[i]))
            for i in range(n_queries)
        ])

        reranked = []
        for i in range(n_queries):
            ids = candidates[i]
            scores = vectors[ids] @ queries[i]
            reranked.append(int(ids[np.argmax(scores)]))
        reranked = np.array(reranked)

        print(
            f"{candidate_k:11d}  {contains.mean():16.3f}  "
            f"{np.mean(reranked == exact_top1):17.3f}"
        )


if __name__ == "__main__":
    main()
