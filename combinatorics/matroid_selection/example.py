"""Matroid-constrained selection for RAG context and LLM training data.

The lab keeps the scorer fixed and changes only the selection rule:

1. baseline: take the globally highest-scoring items;
2. intervention: greedily maximize the same additive score subject to a
   truncated partition-matroid constraint.

For non-negative additive weights, greedy selection is optimal over a matroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Item:
    name: str
    score: float
    partition: str


def top_k(items: Iterable[Item], budget: int) -> list[Item]:
    return sorted(items, key=lambda item: item.score, reverse=True)[:budget]


def matroid_greedy(
    items: Iterable[Item],
    budget: int,
    caps: dict[str, int],
) -> tuple[list[Item], list[Item]]:
    """Maximum-weight greedy selection for a truncated partition matroid."""

    selected: list[Item] = []
    dependent: list[Item] = []
    used = {part: 0 for part in caps}

    for item in sorted(items, key=lambda x: x.score, reverse=True):
        if len(selected) >= budget:
            break

        cap = caps.get(item.partition, budget)
        if used.get(item.partition, 0) < cap:
            selected.append(item)
            used[item.partition] = used.get(item.partition, 0) + 1
        else:
            dependent.append(item)

    return selected, dependent


def rag_demo() -> None:
    print("RAG CONTEXT SELECTION")
    print("-" * 72)

    # Think of score as the output of an already-computed reranker.
    candidates = [
        Item("quantization-memory-1", 0.99, "compression"),
        Item("quantization-memory-2", 0.97, "compression"),
        Item("quantization-memory-3", 0.95, "compression"),
        Item("exact-reranking", 0.91, "reranking"),
        Item("candidate-generation", 0.88, "reranking"),
        Item("tenant-filtering", 0.85, "filtering"),
        Item("online-ingest", 0.83, "operations"),
        Item("unrelated-cooking", 0.08, "unrelated"),
    ]

    relevance_threshold = 0.50
    related = [x for x in candidates if x.score >= relevance_threshold]
    unrelated = [x for x in candidates if x.score < relevance_threshold]

    budget = 4
    baseline = top_k(related, budget)
    selected, dependent = matroid_greedy(
        related,
        budget=budget,
        caps={
            "compression": 1,
            "reranking": 1,
            "filtering": 1,
            "operations": 1,
        },
    )

    required_facets = {"compression", "reranking", "filtering", "operations"}

    def coverage(selection: list[Item]) -> float:
        facets = {x.partition for x in selection}
        return len(facets & required_facets) / len(required_facets)

    def score_sum(selection: list[Item]) -> float:
        return sum(x.score for x in selection)

    print("Baseline top-k:")
    for item in baseline:
        print(f"  {item.name:28s} score={item.score:.2f} facet={item.partition}")

    print("\nMatroid-selected context:")
    for item in selected:
        print(f"  {item.name:28s} score={item.score:.2f} facet={item.partition}")

    print("\nRelevant but dependent/supporting evidence:")
    for item in dependent:
        print(f"  {item.name:28s} score={item.score:.2f} facet={item.partition}")

    print("\nUnrelated:")
    for item in unrelated:
        print(f"  {item.name:28s} score={item.score:.2f}")

    print("\nRAG metrics")
    print(f"  baseline relevance sum : {score_sum(baseline):.3f}")
    print(f"  matroid relevance sum  : {score_sum(selected):.3f}")
    print(f"  baseline facet coverage: {coverage(baseline):.3f}")
    print(f"  matroid facet coverage : {coverage(selected):.3f}")



def training_demo() -> None:
    print("\nLLM TRAINING-DATA SELECTION")
    print("-" * 72)

    # The scores are fixed quality/informativeness scores from an upstream
    # evaluator. The intervention changes only which scored samples consume
    # the training budget.
    pool = [
        Item("math-proof-1", 0.99, "math"),
        Item("math-proof-2", 0.98, "math"),
        Item("math-proof-3", 0.97, "math"),
        Item("python-debug-1", 0.94, "code"),
        Item("python-debug-2", 0.92, "code"),
        Item("physics-reasoning", 0.89, "science"),
        Item("structured-writing", 0.86, "writing"),
        Item("logic-reasoning", 0.84, "reasoning"),
    ]

    budget = 5
    baseline = top_k(pool, budget)
    selected, dependent = matroid_greedy(
        pool,
        budget=budget,
        caps={
            "math": 1,
            "code": 1,
            "science": 1,
            "writing": 1,
            "reasoning": 1,
        },
    )

    domains = {"math", "code", "science", "writing", "reasoning"}

    def mean_quality(selection: list[Item]) -> float:
        return sum(x.score for x in selection) / len(selection)

    def domain_coverage(selection: list[Item]) -> float:
        return len({x.partition for x in selection} & domains) / len(domains)

    def macro_domain_utility(selection: list[Item]) -> float:
        # Proxy for a balanced downstream suite: each domain contributes the
        # best selected example quality, and missing domains contribute zero.
        best = {domain: 0.0 for domain in domains}
        for item in selection:
            best[item.partition] = max(best.get(item.partition, 0.0), item.score)
        return sum(best.values()) / len(domains)

    print("Baseline top-quality samples:")
    for item in baseline:
        print(f"  {item.name:28s} score={item.score:.2f} domain={item.partition}")

    print("\nMatroid-selected training subset:")
    for item in selected:
        print(f"  {item.name:28s} score={item.score:.2f} domain={item.partition}")

    print("\nHigh-quality but dependent under the domain budget:")
    for item in dependent:
        print(f"  {item.name:28s} score={item.score:.2f} domain={item.partition}")

    print("\nTraining-selection proxies")
    print(f"  baseline mean quality       : {mean_quality(baseline):.3f}")
    print(f"  matroid mean quality        : {mean_quality(selected):.3f}")
    print(f"  baseline domain coverage    : {domain_coverage(baseline):.3f}")
    print(f"  matroid domain coverage     : {domain_coverage(selected):.3f}")
    print(f"  baseline macro-domain utility: {macro_domain_utility(baseline):.3f}")
    print(f"  matroid macro-domain utility : {macro_domain_utility(selected):.3f}")


if __name__ == "__main__":
    rag_demo()
    training_demo()
