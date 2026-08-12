"""Optional TurboVec filtered-search experiment for tenant/metadata RAG.

Compares global retrieval followed by post-filtering with TurboVec's in-kernel
allowlist search.  The goal is to show the failure mode where post-filtering
returns fewer than k valid candidates under selective metadata constraints.

Install separately:
    pip install turbovec
"""

from __future__ import annotations

import numpy as np

try:
    from turbovec import IdMapIndex
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("TurboVec is optional. Install it with: pip install turbovec") from exc


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main() -> None:
    rng = np.random.default_rng(19)
    dim = 128
    tenants = 20
    docs_per_tenant = 200
    n_topics = 12
    n = tenants * docs_per_tenant

    topic_centers = normalize(rng.normal(size=(n_topics, dim)).astype(np.float32))
    tenant_ids = np.repeat(np.arange(tenants), docs_per_tenant)
    topic_ids = np.tile(np.arange(docs_per_tenant) % n_topics, tenants)
    vectors = topic_centers[topic_ids] + 0.12 * rng.normal(size=(n, dim)).astype(np.float32)
    vectors = normalize(vectors).astype(np.float32)
    ids = np.arange(10_000, 10_000 + n, dtype=np.uint64)

    index = IdMapIndex(dim=dim, bit_width=4)
    index.add_with_ids(vectors, ids)

    target_tenant = 7
    target_topic = 3
    q = topic_centers[[target_topic]].astype(np.float32)
    k = 5

    # Naive global retrieval + post-filter.
    _, global_ids = index.search(q, k=25)
    global_ids = np.asarray(global_ids)[0].astype(np.uint64)
    id_to_slot = {int(doc_id): slot for slot, doc_id in enumerate(ids)}
    postfiltered = [
        int(doc_id)
        for doc_id in global_ids
        if tenant_ids[id_to_slot[int(doc_id)]] == target_tenant
    ][:k]

    allowed = ids[tenant_ids == target_tenant]
    _, filtered_ids = index.search(q, k=k, allowlist=allowed)
    filtered_ids = np.asarray(filtered_ids)[0].astype(np.uint64).tolist()

    print(f"target tenant: {target_tenant}; topic: {target_topic}; requested k={k}")
    print("post-filter after global top-25:", postfiltered)
    print("in-kernel allowlist top-5      :", filtered_ids)
    print("post-filter returned           :", len(postfiltered))
    print("allowlist returned             :", len(filtered_ids))

    assert len(filtered_ids) == k
    assert all(tenant_ids[id_to_slot[int(doc_id)]] == target_tenant for doc_id in filtered_ids)


if __name__ == "__main__":
    main()
