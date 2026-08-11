"""Optional smoke test using the public TurboVec Python package.

Install separately:
    pip install turbovec

This file is not executed by run_all.py because TurboVec has compiled
Rust/Python bindings and is intentionally kept as an optional dependency.
"""

from __future__ import annotations

import numpy as np

try:
    from turbovec import TurboQuantIndex
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "TurboVec is optional. Install it with: pip install turbovec"
    ) from exc

rng = np.random.default_rng(7)
dim = 128
vectors = rng.normal(size=(1000, dim)).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
query = vectors[[123]].copy()

index = TurboQuantIndex(dim=dim, bit_width=4)
index.add(vectors)
scores, indices = index.search(query, k=5)

print("TurboVec top-5 indices:", indices[0].tolist())
print("TurboVec top-5 scores :", np.round(scores[0], 4).tolist())
print("Expected self-match id 123 to appear at/near rank 1.")
