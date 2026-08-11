"""Metric-conditioned retrieval over frozen embeddings.

The arrays stand in for embeddings produced by a frozen encoder.  The experiment
compares conventional cosine retrieval with a Mahalanobis distance that emphasizes
domain-relevant directions.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def mahalanobis_distance(a, b, metric):
    delta = a - b
    return float(np.sqrt(delta @ metric @ delta))


def recall_at_k(ranking, relevant, k):
    return len(set(ranking[:k]) & set(relevant)) / len(relevant)


# Frozen query/chunk embeddings. Dimensions are intentionally interpretable:
# [generic semantic similarity, domain mechanism, operational consequence].
query = np.array([1.0, 0.75, 0.70])

chunks = {
    "generic_overview": np.array([1.00, 0.55, 0.50]),
    "domain_mechanism": np.array([0.70, 0.78, 0.69]),
    "operational_effect": np.array([0.72, 0.68, 0.76]),
    "generic_definition": np.array([1.00, 0.60, 0.56]),
    "unrelated": np.array([-0.20, 0.10, 0.00]),
}

# In this toy task, these two chunks contain the evidence required by the answer.
relevant = {"domain_mechanism", "operational_effect"}

cosine_ranking = sorted(
    chunks,
    key=lambda name: cosine(query, chunks[name]),
    reverse=True,
)

# Downweight the generic dimension and emphasize the two domain dimensions.
M_domain = np.diag([0.10, 4.0, 4.0])
metric_ranking = sorted(
    chunks,
    key=lambda name: mahalanobis_distance(query, chunks[name], M_domain),
)

print("Frozen query:", query)
print("\nCosine ranking:")
for name in cosine_ranking:
    print(f"  {name:20s} score={cosine(query, chunks[name]):.4f}")

print("\nDomain-metric ranking:")
for name in metric_ranking:
    print(f"  {name:20s} distance={mahalanobis_distance(query, chunks[name], M_domain):.4f}")

print("\nRecall@3")
print("  cosine      :", recall_at_k(cosine_ranking, relevant, 3))
print("  domain metric:", recall_at_k(metric_ranking, relevant, 3))
