"""Neighborhood-adaptive rerank depth over frozen synthetic token embeddings.

The first stage uses cheap mean-pooled cosine.  The expensive stage uses token
MaxSim.  A small set of geometric statistics from the candidate neighborhood
chooses how many candidates deserve the expensive score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), EPS, None)


def cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    return normalize(docs) @ normalize(query)


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT-style late interaction score over already frozen token vectors."""
    q = normalize(query_tokens)
    d = normalize(doc_tokens)
    return float(np.max(q @ d.T, axis=1).mean())


def normalized_entropy(scores: np.ndarray, temperature: float = 0.05) -> float:
    shifted = (scores - scores.max()) / temperature
    probs = np.exp(np.clip(shifted, -60.0, 0.0))
    probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, None))))
    return entropy / np.log(len(scores))


def effective_rank(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 2:
        return 1.0
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if energy.sum() <= EPS:
        return 1.0
    probs = energy / energy.sum()
    return float(np.exp(-np.sum(probs * np.log(np.clip(probs, EPS, None)))))


def neighborhood_features(scores: np.ndarray, points: np.ndarray) -> dict[str, float]:
    margin = float(scores[0] - scores[1]) if len(scores) > 1 else 1.0
    entropy = normalized_entropy(scores)
    rank = effective_rank(points)
    rank_norm = min(1.0, (rank - 1.0) / max(1.0, len(points) - 2.0))

    p = normalize(points)
    pairwise = p @ p.T
    upper = pairwise[np.triu_indices(len(p), k=1)]
    dispersion = float(1.0 - upper.mean()) if len(upper) else 0.0
    dispersion = float(np.clip(dispersion, 0.0, 1.0))

    margin_uncertainty = 1.0 - float(np.clip(margin / 0.15, 0.0, 1.0))
    difficulty = (
        0.45 * entropy
        + 0.30 * margin_uncertainty
        + 0.15 * rank_norm
        + 0.10 * dispersion
    )

    return {
        "margin": margin,
        "entropy": entropy,
        "effective_rank": rank,
        "dispersion": dispersion,
        "difficulty": float(difficulty),
    }


def choose_budget(features: dict[str, float], max_budget: int) -> int:
    """Training-free gate: spend more only when the local neighborhood is ambiguous."""
    d = features["difficulty"]
    margin = features["margin"]
    if d < 0.43 and margin > 0.08:
        return 0
    if d < 0.58:
        return min(4, max_budget)
    if d < 0.72:
        return min(8, max_budget)
    return max_budget


@dataclass
class Case:
    name: str
    query_tokens: np.ndarray
    docs: list[np.ndarray]
    relevant: int


def make_case(kind: str, variation: int, dim: int = 12) -> Case:
    kind_seed = {"easy": 11, "ambiguous": 23, "broad": 37, "noisy": 53}[kind]
    rng = np.random.default_rng(1000 + 37 * variation + kind_seed)
    basis = np.eye(dim)

    if kind == "easy":
        facets = [0]
    elif kind == "ambiguous":
        facets = [0, 1]
    elif kind == "broad":
        facets = [0, 1, 2]
    elif kind == "noisy":
        facets = [0, 1, 2]
    else:  # pragma: no cover - local construction guard
        raise ValueError(kind)

    query_tokens = np.stack([basis[i] for i in facets])
    query_tokens = normalize(query_tokens + rng.normal(scale=0.015, size=query_tokens.shape))

    relevant = normalize(
        np.vstack([basis[i] + rng.normal(scale=0.025, size=dim) for i in facets])
    )
    docs: list[np.ndarray] = []

    if kind == "easy":
        docs.append(relevant)
        for j in range(1, 12):
            idx = (j + 2) % 8 + 3
            token = basis[idx] + rng.normal(scale=0.12, size=dim)
            docs.append(normalize(np.vstack([token, token + rng.normal(scale=0.08, size=dim)])))
        relevant_idx = 0

    elif kind == "ambiguous":
        # Mean-pooled decoys look excellent but do not contain the query facets separately.
        mean_dir = normalize((basis[0] + basis[1])[None, :])[0]
        for scale in (0.01, 0.025):
            token = mean_dir + rng.normal(scale=scale, size=dim)
            docs.append(normalize(np.vstack([token, token + rng.normal(scale=scale, size=dim)])))
        docs.append(relevant)
        for j in range(9):
            idx = 2 + (j % 6)
            docs.append(normalize(np.vstack([
                basis[idx] + rng.normal(scale=0.10, size=dim),
                basis[(idx + 1) % dim] + rng.normal(scale=0.10, size=dim),
            ])))
        relevant_idx = 2

    elif kind == "broad":
        # Several partial matches crowd the top of pooled retrieval.
        partials = [(0, 1), (1, 2), (0, 2), (0,), (1,), (2,)]
        for subset in partials:
            tokens = [basis[i] + rng.normal(scale=0.035, size=dim) for i in subset]
            docs.append(normalize(np.vstack(tokens)))
        docs.append(relevant)
        for j in range(5):
            idx = 3 + j
            docs.append(normalize(np.vstack([
                basis[idx] + rng.normal(scale=0.10, size=dim),
                basis[(idx + 1) % dim] + rng.normal(scale=0.10, size=dim),
            ])))
        relevant_idx = 6

    else:  # noisy
        # The relevant document is still in the candidate pool but the neighborhood is diffuse.
        mixed = [
            (0, 1), (1, 2), (0, 2), (0, 3), (1, 4), (2, 5), (0,), (1,), (2,)
        ]
        for subset in mixed:
            tokens = [basis[i] + rng.normal(scale=0.07, size=dim) for i in subset]
            docs.append(normalize(np.vstack(tokens)))
        docs.append(relevant)
        for j in range(2):
            random_tokens = normalize(rng.normal(size=(3, dim)))
            docs.append(random_tokens)
        relevant_idx = 9

    return Case(
        name=f"{kind}-{variation}",
        query_tokens=query_tokens,
        docs=docs,
        relevant=relevant_idx,
    )


def pooled_doc(doc_tokens: np.ndarray) -> np.ndarray:
    return normalize(doc_tokens.mean(axis=0, keepdims=True))[0]


def rerank_prefix(case: Case, candidate_ids: np.ndarray, budget: int) -> int:
    """Return predicted document id after reranking only the selected prefix."""
    if budget == 0:
        return int(candidate_ids[0])
    chosen = candidate_ids[:budget]
    scores = np.array([maxsim(case.query_tokens, case.docs[int(i)]) for i in chosen])
    return int(chosen[np.argmax(scores)])


def evaluate() -> None:
    cases = [
        make_case(kind, variation)
        for kind in ("easy", "ambiguous", "broad", "noisy")
        for variation in range(6)
    ]

    fixed_budget = 12
    fixed_correct = 0
    adaptive_correct = 0
    no_rerank_correct = 0
    adaptive_calls = 0
    budgets: list[int] = []

    print("case            kind        margin  entropy  eff-rank  difficulty  budget  hit")
    print("-" * 82)

    for case in cases:
        q_pool = normalize(case.query_tokens.mean(axis=0, keepdims=True))[0]
        doc_pools = np.stack([pooled_doc(doc) for doc in case.docs])
        stage1_scores = cosine_scores(q_pool, doc_pools)
        candidate_ids = np.argsort(-stage1_scores)[:fixed_budget]
        candidate_scores = stage1_scores[candidate_ids]
        candidate_points = doc_pools[candidate_ids]

        features = neighborhood_features(candidate_scores, candidate_points)
        budget = choose_budget(features, fixed_budget)
        budgets.append(budget)
        adaptive_calls += budget

        pred_no = int(candidate_ids[0])
        pred_fixed = rerank_prefix(case, candidate_ids, fixed_budget)
        pred_adaptive = rerank_prefix(case, candidate_ids, budget)

        no_rerank_correct += int(pred_no == case.relevant)
        fixed_correct += int(pred_fixed == case.relevant)
        adaptive_correct += int(pred_adaptive == case.relevant)

        kind = case.name.split("-")[0]
        print(
            f"{case.name:15s} {kind:10s} "
            f"{features['margin']:.3f}   {features['entropy']:.3f}    "
            f"{features['effective_rank']:.2f}      {features['difficulty']:.3f}      "
            f"{budget:2d}     {'yes' if pred_adaptive == case.relevant else 'no'}"
        )

    n = len(cases)
    print("\nSummary")
    print(f"  no rerank top-1 accuracy : {no_rerank_correct / n:.3f}")
    print(f"  fixed top-{fixed_budget} MaxSim   : {fixed_correct / n:.3f}")
    print(f"  adaptive MaxSim accuracy : {adaptive_correct / n:.3f}")
    print(f"  fixed rerank calls/query : {fixed_budget:.2f}")
    print(f"  adaptive calls/query     : {adaptive_calls / n:.2f}")
    print(f"  rerank work saved        : {1.0 - adaptive_calls / (n * fixed_budget):.1%}")
    print(f"  budgets used             : {sorted(set(budgets))}")


if __name__ == "__main__":
    evaluate()
