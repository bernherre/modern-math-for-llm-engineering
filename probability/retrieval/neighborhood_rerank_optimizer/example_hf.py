"""Neighborhood-adaptive reranking over frozen sentence/token embeddings.

Stage 1: pooled sentence embeddings + cosine.
Stage 2: token-level MaxSim, computed only for the budget selected from the
geometry of the first-stage neighborhood.
"""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        'Install the optional HF dependencies first: python -m pip install -e ".[hf]"'
    ) from exc

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), EPS, None)


def normalized_entropy(scores: np.ndarray, temperature: float = 0.04) -> float:
    shifted = (scores - scores.max()) / temperature
    probs = np.exp(np.clip(shifted, -60.0, 0.0))
    probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log(np.clip(probs, EPS, None))))
    return entropy / np.log(len(scores))


def effective_rank(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0, keepdims=True)
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
    dispersion = float(np.clip(1.0 - upper.mean(), 0.0, 1.0)) if len(upper) else 0.0
    margin_uncertainty = 1.0 - float(np.clip(margin / 0.12, 0.0, 1.0))
    difficulty = 0.45 * entropy + 0.30 * margin_uncertainty + 0.15 * rank_norm + 0.10 * dispersion
    return {
        "margin": margin,
        "entropy": entropy,
        "effective_rank": rank,
        "dispersion": dispersion,
        "difficulty": float(difficulty),
    }


def choose_budget(features: dict[str, float], max_budget: int) -> int:
    d = features["difficulty"]
    margin = features["margin"]
    if d < 0.42 and margin > 0.07:
        return 0
    if d < 0.57:
        return min(3, max_budget)
    if d < 0.71:
        return min(5, max_budget)
    return max_budget


def token_embeddings(model: SentenceTransformer, texts: list[str]) -> list[np.ndarray]:
    encoded = model.encode(
        texts,
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    result = []
    for tokens in encoded:
        arr = tokens.detach().cpu().numpy() if hasattr(tokens, "detach") else np.asarray(tokens)
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) > 2:
            arr = arr[1:-1]
        result.append(arr)
    return result


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    q = normalize(query_tokens)
    d = normalize(doc_tokens)
    return float(np.max(q @ d.T, axis=1).mean())


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    documents = [
        "TurboQuant uses random rotations and low-bit quantization to reduce vector-index memory.",
        "Product quantization compresses vector indexes using learned codebooks.",
        "TurboVec provides a Rust vector index with online ingest and filtered search.",
        "KV-cache quantization reduces memory during language-model inference.",
        "A reranker can rescore the candidate documents returned by a dense retriever.",
        "ColBERT keeps token-level embeddings and uses late interaction instead of only one pooled vector.",
        "Sheaf methods can model local-to-global consistency across related pieces of information.",
        "Natural gradient uses information geometry to precondition parameter updates.",
        "An optical resonator traps laser light between mirrors.",
        "A Lie algebra describes infinitesimal generators of continuous transformations.",
    ]

    cases = [
        (
            "How can I compress a RAG vector index without training a codebook?",
            0,
        ),
        (
            "How can I rerank dense-retrieval candidates using token-level semantic evidence?",
            5,
        ),
        (
            "What does TurboVec add on top of TurboQuant for practical retrieval?",
            2,
        ),
        (
            "How can mathematical consistency help filter context before an LLM answers?",
            6,
        ),
    ]

    pooled_docs = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64)
    doc_token_sets = token_embeddings(model, documents)

    fixed_budget = min(8, len(documents))
    total_adaptive = 0
    fixed_hits = 0
    adaptive_hits = 0
    stage1_hits = 0

    print(f"model: {MODEL_NAME}\n")

    for query, relevant in cases:
        q_pool = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float64)
        q_tokens = token_embeddings(model, [query])[0]

        scores = pooled_docs @ q_pool
        candidate_ids = np.argsort(-scores)[:fixed_budget]
        candidate_scores = scores[candidate_ids]
        features = neighborhood_features(candidate_scores, pooled_docs[candidate_ids])
        budget = choose_budget(features, fixed_budget)
        total_adaptive += budget

        stage1_pred = int(candidate_ids[0])

        fixed_scores = [maxsim(q_tokens, doc_token_sets[int(i)]) for i in candidate_ids]
        fixed_pred = int(candidate_ids[int(np.argmax(fixed_scores))])

        if budget == 0:
            adaptive_pred = stage1_pred
        else:
            chosen = candidate_ids[:budget]
            adaptive_scores = [maxsim(q_tokens, doc_token_sets[int(i)]) for i in chosen]
            adaptive_pred = int(chosen[int(np.argmax(adaptive_scores))])

        stage1_hits += int(stage1_pred == relevant)
        fixed_hits += int(fixed_pred == relevant)
        adaptive_hits += int(adaptive_pred == relevant)

        print(query)
        print(
            "  geometry: "
            f"margin={features['margin']:.3f} "
            f"entropy={features['entropy']:.3f} "
            f"eff_rank={features['effective_rank']:.2f} "
            f"difficulty={features['difficulty']:.3f}"
        )
        print(f"  adaptive budget: {budget}/{fixed_budget}")
        print(f"  stage1       : {documents[stage1_pred]}")
        print(f"  fixed MaxSim : {documents[fixed_pred]}")
        print(f"  adaptive     : {documents[adaptive_pred]}\n")

    n = len(cases)
    print("Summary")
    print(f"  pooled top-1 accuracy    : {stage1_hits / n:.3f}")
    print(f"  fixed MaxSim accuracy    : {fixed_hits / n:.3f}")
    print(f"  adaptive MaxSim accuracy : {adaptive_hits / n:.3f}")
    print(f"  fixed token reranks/query: {fixed_budget:.2f}")
    print(f"  adaptive reranks/query   : {total_adaptive / n:.2f}")


if __name__ == "__main__":
    main()
