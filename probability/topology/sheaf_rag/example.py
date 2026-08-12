"""Sheaf-inspired context selection for RAG.

A retrieval score alone can select mutually inconsistent evidence.  This lab adds a
local-to-global compatibility energy before the context is passed to the generator.
"""

from itertools import combinations
import numpy as np

# A scalar local state represents the normalized value of the same underlying
# quantity extracted from each chunk. Compatible sources should map to similar
# values after normalization.
chunks = {
    "source_A": {"relevance": 0.95, "state": 1.00},
    "source_B": {"relevance": 0.93, "state": 1.04},
    "source_C": {"relevance": 0.90, "state": -0.85},  # contradictory
    "source_D": {"relevance": 0.86, "state": 0.98},
    "source_E": {"relevance": 0.72, "state": 1.08},
}

K = 3
LAMBDA = 0.60


def consistency_energy(names):
    """0-cochain disagreement energy on the complete local compatibility graph."""
    states = np.array([chunks[n]["state"] for n in names], dtype=float)
    energy = 0.0
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            energy += (states[i] - states[j]) ** 2
    return float(energy)


def relevance(names):
    return sum(chunks[n]["relevance"] for n in names)


baseline = sorted(chunks, key=lambda n: chunks[n]["relevance"], reverse=True)[:K]

candidates = []
for subset in combinations(chunks.keys(), K):
    rel = relevance(subset)
    energy = consistency_energy(subset)
    score = rel - LAMBDA * energy
    candidates.append((score, subset, rel, energy))

structured = max(candidates, key=lambda row: row[0])

print("Baseline top-k by relevance:")
print("  context   =", baseline)
print("  relevance =", round(relevance(baseline), 4))
print("  energy    =", round(consistency_energy(baseline), 4))

print("\nSheaf-inspired context selection:")
print("  context   =", list(structured[1]))
print("  relevance =", round(structured[2], 4))
print("  energy    =", round(structured[3], 4))
print("  objective =", round(structured[0], 4))
