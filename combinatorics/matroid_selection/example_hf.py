"""Frozen SentenceTransformer companion for matroid-constrained selection.

The encoder remains frozen. Embeddings are used only to score relevance and to
assign examples to semantic partitions; the matroid changes the subset that is
allowed to consume context/training budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Item:
    text: str
    score: float
    partition: str


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def assign_partitions(
    embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    anchor_names: list[str],
) -> list[str]:
    similarities = embeddings @ anchor_embeddings.T
    return [anchor_names[i] for i in np.argmax(similarities, axis=1)]


def matroid_greedy(items: list[Item], budget: int, cap: int = 1) -> list[Item]:
    selected: list[Item] = []
    counts: dict[str, int] = {}

    for item in sorted(items, key=lambda x: x.score, reverse=True):
        if len(selected) >= budget:
            break
        if counts.get(item.partition, 0) >= cap:
            continue
        selected.append(item)
        counts[item.partition] = counts.get(item.partition, 0) + 1

    return selected


def rag_demo(model: SentenceTransformer) -> None:
    print("RAG WITH FROZEN SENTENCE EMBEDDINGS")
    print("-" * 72)

    query = "How can low-bit vector quantization make RAG cheaper without losing retrieval quality?"
    docs = [
        "Low-bit quantization reduces the memory footprint of vector embeddings used by RAG indexes.",
        "Quantizing embedding vectors to two or four bits can shrink the memory used by a RAG index.",
        "Vector quantization compresses RAG embeddings so larger corpora can fit in memory.",
        "A cheap approximate index can retrieve candidates and exact float32 reranking can recover ranking quality.",
        "Metadata allowlists can filter a RAG search by tenant before semantic ranking.",
        "Online insertion lets a RAG index accept new embeddings without retraining its quantizer.",
        "Fresh basil and tomatoes are common ingredients in a simple pasta sauce.",
    ]

    anchors = {
        "compression": "reduce memory with low-bit vector quantization",
        "reranking": "recover retrieval quality with exact reranking",
        "filtering": "filter retrieval by metadata tenant or access control",
        "operations": "insert new vectors online without rebuilding the index",
    }

    q = encode(model, [query])[0]
    d = encode(model, docs)
    anchor_names = list(anchors)
    a = encode(model, [anchors[name] for name in anchor_names])

    relevance = d @ q
    partitions = assign_partitions(d, a, anchor_names)
    items = [Item(text, float(score), part) for text, score, part in zip(docs, relevance, partitions)]

    # Reranker/retriever first decides what is related. The matroid only acts
    # on the relevant pool, matching the intended separation of responsibilities.
    threshold = max(0.20, float(np.quantile(relevance, 0.20)))
    related = [x for x in items if x.score >= threshold]
    unrelated = [x for x in items if x.score < threshold]

    budget = 4
    baseline = sorted(related, key=lambda x: x.score, reverse=True)[:budget]
    selected = matroid_greedy(related, budget=budget, cap=1)

    print("Baseline top-k:")
    for item in baseline:
        print(f"  {item.score: .3f} [{item.partition:11s}] {item.text}")

    print("\nMatroid-selected context:")
    for item in selected:
        print(f"  {item.score: .3f} [{item.partition:11s}] {item.text}")

    print("\nUnrelated / filtered before the matroid:")
    for item in unrelated:
        print(f"  {item.score: .3f} [{item.partition:11s}] {item.text}")

    print("\nUnique semantic partitions")
    print(f"  baseline: {len({x.partition for x in baseline})}")
    print(f"  matroid : {len({x.partition for x in selected})}")



def training_demo(model: SentenceTransformer) -> None:
    print("\nTRAINING-DATA SELECTION WITH FROZEN EMBEDDINGS")
    print("-" * 72)

    samples = [
        ("Prove that the sum of two even integers is even.", 0.99),
        ("Solve a quadratic equation and explain each algebraic step.", 0.98),
        ("Derive the derivative of x squared from the limit definition.", 0.96),
        ("Find and fix the bug in this Python loop and explain the correction.", 0.94),
        ("Refactor a Python function to reduce duplicated logic.", 0.92),
        ("Explain why increasing pressure changes the boiling point of water.", 0.90),
        ("Rewrite this technical paragraph so that it is concise and precise.", 0.87),
        ("Given three premises, determine whether the conclusion follows logically.", 0.85),
    ]

    anchors = {
        "math": "mathematical proof algebra calculus",
        "code": "programming software debugging code",
        "science": "physics chemistry scientific explanation",
        "writing": "writing editing summarization communication",
        "reasoning": "logic reasoning deduction inference",
    }

    texts = [text for text, _ in samples]
    x = encode(model, texts)
    anchor_names = list(anchors)
    a = encode(model, [anchors[name] for name in anchor_names])
    partitions = assign_partitions(x, a, anchor_names)

    items = [
        Item(text=text, score=quality, partition=partition)
        for (text, quality), partition in zip(samples, partitions)
    ]

    budget = 5
    baseline = sorted(items, key=lambda x: x.score, reverse=True)[:budget]
    selected = matroid_greedy(items, budget=budget, cap=1)

    print("Baseline top-quality:")
    for item in baseline:
        print(f"  {item.score:.2f} [{item.partition:9s}] {item.text}")

    print("\nMatroid-selected training subset:")
    for item in selected:
        print(f"  {item.score:.2f} [{item.partition:9s}] {item.text}")

    print("\nDomain/skill coverage")
    print(f"  baseline: {len({x.partition for x in baseline})}")
    print(f"  matroid : {len({x.partition for x in selected})}")


if __name__ == "__main__":
    print(f"Loading frozen encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    rag_demo(encoder)
    training_demo(encoder)
