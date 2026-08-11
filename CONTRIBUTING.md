# Contributing

## Scope

This repository is about **modern mathematics used to improve or diagnose practical AI/LLM processes**. It is not a collection of pure-math demonstrations.

A new lab must modify at least one concrete process: retrieval, RAG, context selection, fine-tuning/PEFT, inference, routing, adapter composition, representation analysis, evaluation, compression, or optimization.

## Required lab contract

Every lab must contain at least:

- `README.md`
- `example.py`

The README must follow `LAB_TEMPLATE.md` and include academic sources. The example must include a baseline, the mathematical intervention, and at least one quantitative metric.

## Evidence levels

Label results conceptually as:

1. **mechanism demo** - synthetic/small controlled experiment;
2. **paper reproduction** - reproduces a published experiment;
3. **real-pipeline benchmark** - uses a real retrieval/LLM pipeline;
4. **extension** - tests a new hypothesis beyond the cited paper.

Do not present a mechanism demo as evidence of production improvement.
