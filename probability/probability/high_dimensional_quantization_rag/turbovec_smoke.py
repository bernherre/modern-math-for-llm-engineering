"""Optional integration smoke test for the public TurboVec Python package.

Covers stable ids, filtered search, persistence, reload, removal, and incremental
sync when supported by the installed TurboVec build.

Install separately:
    pip install turbovec
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

try:
    from turbovec import IdMapIndex
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("TurboVec is optional. Install it with: pip install turbovec") from exc


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def first_row(x: np.ndarray) -> list[int]:
    return np.asarray(x)[0].astype(np.uint64).tolist()


def main() -> None:
    rng = np.random.default_rng(7)
    dim = 128
    vectors = normalize(rng.normal(size=(8, dim)).astype(np.float32)).astype(np.float32)
    ids = np.arange(1001, 1009, dtype=np.uint64)
    query = vectors[[2]].copy()

    index = IdMapIndex(dim=dim, bit_width=4)
    index.add_with_ids(vectors, ids)

    scores, found = index.search(query, k=4)
    print("initial top ids:", first_row(found))
    print("initial scores :", np.round(np.asarray(scores)[0], 4).tolist())
    assert int(np.asarray(found)[0][0]) == 1003, "self match should rank first"

    allowed = np.array([1001, 1003, 1008], dtype=np.uint64)
    _, filtered = index.search(query, k=3, allowlist=allowed)
    filtered_ids = first_row(filtered)
    print("allowlist ids  :", filtered_ids)
    assert set(filtered_ids).issubset(set(allowed.tolist()))

    with tempfile.TemporaryDirectory(prefix="turbovec-smoke-") as tmp:
        path = Path(tmp) / "index.tvim"
        index.write(str(path))
        loaded = IdMapIndex.load(str(path))

        _, loaded_ids = loaded.search(query, k=4)
        print("reloaded top   :", first_row(loaded_ids))
        assert int(np.asarray(loaded_ids)[0][0]) == 1003

        loaded.remove(1003)
        _, after_remove = loaded.search(query, k=4)
        after_ids = first_row(after_remove)
        print("after remove   :", after_ids)
        assert 1003 not in after_ids

        extra = normalize(rng.normal(size=(1, dim)).astype(np.float32)).astype(np.float32)
        loaded.add_with_ids(extra, np.array([2001], dtype=np.uint64))
        if hasattr(loaded, "sync"):
            loaded.sync(str(path))
            print("incremental sync: supported")
        else:  # older package versions
            loaded.write(str(path))
            print("incremental sync: unavailable; used full write")

        resumed = IdMapIndex.load(str(path))
        _, resumed_ids = resumed.search(extra, k=1)
        assert int(np.asarray(resumed_ids)[0][0]) == 2001
        print("resume self id :", int(np.asarray(resumed_ids)[0][0]))

    print("TurboVec smoke test passed.")


if __name__ == "__main__":
    main()
