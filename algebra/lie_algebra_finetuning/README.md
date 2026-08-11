# Lie-Algebra Constrained Fine-tuning

## AI process

Parameter-efficient adaptation of a frozen representation transform.

Conventional PEFT learns additive deltas such as low-rank updates. This lab tests a different idea: if the useful adaptation is known or suspected to live near a transformation group, optimize in its Lie algebra and map the update back to the group.

## Baseline

An unconstrained 2x2 linear adapter updated by gradient descent.

## Mathematical intervention

Use the one-dimensional Lie algebra of `SO(2)`:

\[
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\qquad
R(\theta)=\exp(\theta J).
\]

Only `theta` is trained. The resulting transformation stays orthogonal, so vector norms/angles are preserved.

## Hypothesis

When the downstream adaptation is approximately symmetry-preserving, a Lie-structured parameterization can reach the target with fewer trainable parameters and less representation distortion.

## Related papers

1. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022. Canonical additive low-rank PEFT reference.  
   https://openreview.net/forum?id=nZeVKeeFYf9

2. **Controlling Text-to-Image Diffusion by Orthogonal Finetuning** — Qiu et al., NeurIPS 2023. Uses orthogonal transformations for parameter-efficient adaptation while preserving hyperspherical structure.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract.html

3. **Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations** — Si et al., 2025. Treats parameter updates as Lie-algebra perturbations mapped to a Lie group via the exponential map.  
   https://arxiv.org/abs/2504.00851

4. **Orthogonal Finetuning Made Scalable** — Qiu et al., 2025. Scales orthogonal fine-tuning to large foundation models and quantized settings.  
   https://arxiv.org/abs/2506.19847

## Experiment

Frozen hidden vectors are adapted to a target task whose true transformation is a rotation. Compare:

- 4-parameter unconstrained linear adapter;
- 1-parameter Lie-algebra rotation adapter.

```bash
python example.py
```

## Metrics

- train MSE
- validation MSE
- norm distortion on validation representations
- trainable parameter count

## Next experiment

Apply block-orthogonal/Lie updates to a frozen transformer layer or LoRA subspace and compare quality-retention trade-offs with LoRA under an equal parameter budget.

## Observed v0.1 result

On a noisy six-example adaptation set, the unconstrained four-parameter matrix reaches validation MSE **0.00173722** with mean norm distortion **0.02569889**. The one-parameter `SO(2)` adapter reaches validation MSE **0.00015867** and norm distortion **0.0**.
