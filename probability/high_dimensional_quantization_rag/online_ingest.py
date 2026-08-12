"""Optional TurboVec online-ingest and incremental-persistence microbenchmark.

Adds several batches to an IdMapIndex with no codebook-training phase, checks
that a just-added vector can be retrieved, and measures full write / incremental
sync latency when the installed TurboVec build supports sync().

Install separately:
    pip install turbovec
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

try:
    from turbovec import IdMapIndex
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("TurboVec is optional. Install it with: pip install turbovec") from exc


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    rng = np.random.default_rng(23)
    dim = 128
    batch_size = 750
    batches = 4
    index = IdMapIndex(dim=dim, bit_width=4)

    next_id = 1_000_000
    with tempfile.TemporaryDirectory(prefix="turbovec-ingest-") as tmp:
        path = Path(tmp) / "online.tvim"

        print("batch  total  add-ms/vector  persist-ms  self-match")
        print("-----  -----  -------------  ----------  ----------")

        for batch in range(batches):
            vectors = normalize(rng.normal(size=(batch_size, dim)).astype(np.float32)).astype(np.float32)
            ids = np.arange(next_id, next_id + batch_size, dtype=np.uint64)
            next_id += batch_size

            start = perf_counter()
            index.add_with_ids(vectors, ids)
            add_ms_per_vector = (perf_counter() - start) * 1000.0 / batch_size

            start = perf_counter()
            if batch == 0 or not hasattr(index, "sync"):
                index.write(str(path))
            else:
                index.sync(str(path))
            persist_ms = (perf_counter() - start) * 1000.0

            _, found = index.search(vectors[[-1]], k=1)
            found_id = int(np.asarray(found)[0][0])
            self_match = found_id == int(ids[-1])
            total = (batch + 1) * batch_size
            print(
                f"{batch + 1:5d}  {total:5d}  {add_ms_per_vector:13.4f}  "
                f"{persist_ms:10.2f}  {str(self_match):>10s}"
            )

        loaded = IdMapIndex.load(str(path))
        _, found = loaded.search(vectors[[-1]], k=1)
        assert int(np.asarray(found)[0][0]) == int(ids[-1])
        print("\nReload after final persist: passed")


if __name__ == "__main__":
    main()
