"""Optional TurboVec ANN benchmark against exact float32 cosine.

This is intentionally small enough for a laptop.  It measures candidate recall,
search latency, build time, and serialized index size for 2-bit and 4-bit
TurboVec indexes.  Use --calibrate to exercise TQ+ when the installed API
supports calibration.

Install separately:
    pip install turbovec
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

try:
    from turbovec import TurboQuantIndex
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("TurboVec is optional. Install it with: pip install turbovec") from exc


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def build_dataset(n: int, dim: int, n_queries: int, seed: int = 41):
    rng = np.random.default_rng(seed)
    n_topics = 32
    topics = normalize(rng.normal(size=(n_topics, dim)).astype(np.float32))
    labels = rng.integers(0, n_topics, size=n)
    vectors = topics[labels] + 0.18 * rng.normal(size=(n, dim)).astype(np.float32)
    vectors = normalize(vectors).astype(np.float32)

    anchor_ids = rng.choice(n, size=n_queries, replace=False)
    queries = vectors[anchor_ids] + 0.025 * rng.normal(size=(n_queries, dim)).astype(np.float32)
    queries = normalize(queries).astype(np.float32)
    return vectors, queries


def exact_topk(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    scores = queries @ vectors.T
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(queries))[:, None]
    order = np.argsort(-scores[row, part], axis=1)
    return part[row, order]


def recall_at_k(exact: np.ndarray, approx: np.ndarray) -> float:
    values = []
    for truth, pred in zip(exact, approx, strict=True):
        values.append(len(set(map(int, truth)) & set(map(int, pred))) / len(truth))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    vectors, queries = build_dataset(args.n, args.dim, args.queries)
    exact = exact_topk(vectors, queries, args.k)

    print(f"corpus={args.n} dim={args.dim} queries={args.queries} k={args.k}")
    print(f"calibration requested: {args.calibrate}\n")
    print("bits  recall@k  top1-match  build-ms  search-ms/query  file-MiB")
    print("----  --------  ----------  --------  ---------------  --------")

    for bits in (2, 4):
        index = TurboQuantIndex(dim=args.dim, bit_width=bits)
        if args.calibrate and hasattr(index, "calibrate"):
            sample = vectors[: min(1024, len(vectors))]
            index.calibrate(sample)

        start = perf_counter()
        index.add(vectors)
        build_ms = (perf_counter() - start) * 1000.0

        # One warm search before timing.
        index.search(queries[:1], k=args.k)
        start = perf_counter()
        _, approx = index.search(queries, k=args.k)
        search_ms = (perf_counter() - start) * 1000.0 / len(queries)
        approx = np.asarray(approx, dtype=np.int64)

        recall = recall_at_k(exact, approx)
        top1 = float(np.mean(exact[:, 0] == approx[:, 0]))

        with tempfile.TemporaryDirectory(prefix="turbovec-bench-") as tmp:
            path = Path(tmp) / f"index-{bits}bit.tv"
            index.write(str(path))
            mib = path.stat().st_size / (1024.0**2)

        print(
            f"{bits:>4d}  {recall:8.3f}  {top1:10.3f}  "
            f"{build_ms:8.1f}  {search_ms:15.3f}  {mib:8.3f}"
        )

    print("\nExact float32 is the relevance oracle here; this is not an end-to-end RAG benchmark.")


if __name__ == "__main__":
    main()
