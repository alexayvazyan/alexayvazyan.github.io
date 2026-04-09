---
layout: post
title: "Linear Ablation with Rotating Representations of Refusal"
date: 2026-04-09
---

# Linear Ablation with Rotating Representations of Refusal

*Refusal direction ablation works beautifully on Qwen3-8B and completely fails on Gemma 4. The reason turns out to be architectural: Gemma's mixed sliding/full attention layers create two distinct refusal representations that point in nearly orthogonal directions.*

---

## Background

[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) showed that refusal in language models is mediated by a single direction in the residual stream. If you identify that direction and project it out at every layer, the model stops refusing harmful prompts while remaining coherent on harmless ones. They demonstrated this on Llama-family models with impressive results — near-zero KL divergence on harmless prompts (meaning the model's behavior is essentially unchanged for normal queries) while refusal is completely suppressed.

The technique is simple and elegant:
1. Collect residual stream activations on harmful and harmless prompts
2. Compute the difference in means — this is your candidate refusal direction
3. At every layer, install hooks that project out this direction from the block input, attention output, and MLP output
4. The model can no longer represent "I should refuse" and complies with everything

The key assumption is that a **single direction** can be ablated at **every layer** of the model, and that this direction is consistent enough across layers that removing it everywhere does more good (suppressing refusal) than harm (distorting other representations). I wanted to see how well this assumption holds across different architectures.

---

## Replicating on Qwen3-8B: It Just Works

As a starting point, I replicated the technique on Qwen3-8B. The methodology follows the paper: 100 harmful prompts (AdvBench), 100 harmless prompts (Alpaca), left-padded tokenization, activations collected at the end-of-instruction (EOI) token positions — the tokens between the user's message and the model's response in the chat template.

A sweep across all (position, layer) candidates gave a best direction at layer 19, position -9, with:

| Metric | Value |
|--------|-------|
| Refusal score | 0.0001 |
| KL divergence (harmless) | **0.0277** |

KL of 0.03 means the model's output distribution on harmless prompts is virtually identical with and without the ablation. Refusal score of 0.0001 means the model almost never starts a refusal response. This is consistent with the paper's results on Llama models.

---

## Gemma 4 E4B: It Completely Fails

Naturally, I tried the same thing on Google's Gemma 4 E4B (4 billion parameters). Same methodology, same datasets, same hook strategy.

| Metric | Qwen3-8B | Gemma 4 E4B |
|--------|----------|-------------|
| Best refusal score | 0.0001 | 0.0001 |
| KL divergence | **0.0277** | **~3–6** |

The refusal score looks fine — the model stops refusing. But the KL divergence is 100-200x worse. The model's behavior on harmless prompts is severely damaged. Asking it a simple question produces garbled or incoherent output. Ablating the refusal direction is destroying something important.

I also ran the experiment on Gemma 4 31B (4-bit quantized to fit in VRAM). Same story: KL divergence of ~20. The problem gets *worse* with the larger Gemma model.

This is puzzling. We're removing a single direction from a ~3000-dimensional space. That's less than 0.04% of the representation capacity. On Qwen this causes essentially no collateral damage. On Gemma it's catastrophic. Why?

---

## Debugging: Was It Just a Bug?

Before looking for architectural explanations, I spent a while making sure the methodology was identical between the two models. Several fixes along the way:

**Left-padding matters.** With right-padding, the "last token" position differs across prompts of different lengths. The diff-in-means then averages activations from different functional roles — some prompts contribute their actual EOI token, others contribute a pad token or something in the middle of the message. Left-padding aligns all prompts from the right, so negative indexing always picks the correct EOI positions. This was important for both models, but the same fix was applied to both.

**The paper's 3-hook strategy matters.** The paper installs hooks at three points per layer: a pre-hook on the block input, and post-hooks on both the attention and MLP outputs. Using only a single output hook per layer gives worse results. Again, same fix applied to both models.

After all debugging, Qwen still achieved KL=0.03, and Gemma was still at KL=3+. The gap was real and architectural.

---

## The Architectural Difference: Sliding Attention

Qwen3-8B uses standard full attention at every layer — each token attends to every other token. Gemma 4 uses a **hybrid attention pattern**: most layers use sliding window attention (local context only), with full attention layers interspersed every 6th layer (layers 5, 11, 17, 23, 29, 35, 41 in the 42-layer E4B model).

This is a fundamentally different information flow. In Qwen, every layer has access to the full context. In Gemma, most layers only see a local window, with full-context information periodically reinjected at the full attention layers. The question is whether the refusal direction can remain stable across these two different types of computation.

---

## Direction Stability Analysis

To test this, I trained linear probes (L2-regularized logistic regression, C=0.01) at every (layer, EOI position) combination on ~800 harmful and ~800 harmless prompts. The probe's weight vector at each layer gives us the direction that best separates harmful from harmless at that layer. If the refusal direction is truly a single consistent direction, then the probe directions at adjacent layers should have high cosine similarity.

### Qwen3-8B: Smooth and Stable

![Qwen3-8B decay from best layer — smooth mountain shape](/assets/images/refusal_qwen_decay_from_best.png)
*Cosine similarity of each layer's probe direction with the best layer's direction, for each EOI position. Qwen shows a smooth, symmetric decay — the refusal direction is stable across all 36 layers.*

The probe directions at adjacent layers have cosine similarity of 0.85–0.95 throughout the middle of the network. The pairwise cosine similarity matrix (below) shows a smooth gradient — nearby layers agree strongly, distant layers agree less, but there are no abrupt transitions.

![Qwen3-8B full direction stability — pairwise cosine matrices and adjacent similarity](/assets/images/refusal_qwen_direction_stability.png)
*Left column: pairwise cosine similarity matrix between all layer pairs. Middle: adjacent layer cosine similarity. Right: probe vs diff-in-means agreement. Each row is a different EOI token position.*

This is exactly the pattern that makes single-direction ablation work. The refusal direction rotates slowly across layers, so a single direction (from the best layer) has reasonable alignment everywhere. Ablating it at every layer removes a little bit of refusal at each layer, and the cumulative effect suppresses refusal while the collateral damage is minimal because you're always removing something close to the actual refusal direction.

### Gemma 4 E4B: A Phase Transition

![Gemma 4 E4B decay from best layer — fragmented peaks](/assets/images/refusal_gemma4_decay_from_best.png)
*Same analysis on Gemma 4. Instead of a smooth mountain, we see fragmented peaks. The refusal direction in layers 0–22 is nearly orthogonal to the direction in layers 23+.*

The Gemma 4 picture is dramatically different. There is a sharp discontinuity around layer 22–23 (which happens to be a sliding-to-full attention boundary). The adjacent layer cosine similarity at this transition:

```
layer 22 → 23:  +0.35  (sliding → full)  ← type boundary
layer 23 → 24:  +0.06  (full → sliding)  ← type boundary
```

A cosine similarity of 0.06 means these directions are nearly **orthogonal**. The refusal direction in layers 0–22 and the refusal direction in layers 23+ are essentially different vectors pointing in different directions in the ~3000-dimensional space.

![Gemma 4 E4B full direction stability — block diagonal structure visible](/assets/images/refusal_gemma4_direction_stability.png)
*The pairwise cosine similarity matrices show clear block-diagonal structure — two separate regimes of refusal representation with a sharp boundary around layer 23.*

The pairwise cosine matrices are the most striking visualization. Instead of Qwen's smooth gradient, Gemma shows two distinct blocks: layers 0–22 form one block of mutually similar directions, and layers 23+ form another. The off-diagonal regions (cross-block similarity) are near zero.

---

## Why Single-Direction Ablation Fails

This directly explains the catastrophic KL divergence. The best single direction typically comes from a later layer (around layer 30–36, where probe accuracy is highest). When this direction is ablated at all 42 layers:

- **Layers 23+**: the ablated direction is close to the actual refusal direction. Refusal is suppressed, minimal collateral damage. This is working as intended.
- **Layers 0–22**: the ablated direction is nearly orthogonal to the refusal direction at these layers. Ablating it removes essentially zero refusal signal, but removes a dimension of useful representation that happened to point in this unrelated direction. This is pure collateral damage.

The model has two distinct refusal subspaces, and single-direction ablation can only target one of them. The other half of the network gets a random dimension removed at every layer for no benefit.

---

## The Fix: Layer-Selective Ablation

The obvious fix: don't ablate where you don't need to. I tested six configurations, selecting the best direction for each regime by filtering to candidates with refusal score < 0.1 and then picking the one with minimum KL divergence:

| Configuration | Refusal | KL |
|--------------|---------|------|
| 1. Single direction (all layers) | 0.0206 | 1.2295 |
| 2. Regime A only (layers 0–22) | 0.0792 | 1.3573 |
| 3. **Regime B only (layers 23+)** | **0.0489** | **0.1886** |
| 4. Two-direction (A @ 0–22, B @ 23+) | 0.0246 | 1.2674 |
| 5. Single dir, regime A layers only | 0.9920 | 0.5839 |
| 6. Single dir, regime B layers only | 0.0282 | 0.8642 |

Config 3 is the clear winner: **refusal = 0.05, KL = 0.19**. This is a 6–30x improvement in KL over ablating at all layers, and refusal is still effectively suppressed.

Two findings jump out:

**The refusal decision lives primarily in the later layers.** Config 3 ablates only at layers 23+ and achieves refusal = 0.05 — nearly as good as ablating everywhere. Config 5 ablates only at layers 0–22 and achieves refusal = 0.99 — it does nothing. The model's refusal behavior is mediated almost entirely by the second half of the network.

**Touching the early layers is actively harmful regardless of direction.** Configs 2 and 4 both ablate in layers 0–22 and both pay a heavy KL penalty (>1.2). Even the two-direction approach (config 4), which uses a direction specifically optimized for the early layers, still causes large collateral damage. The early layers are fragile — they don't tolerate having any direction projected out, even one aligned with the local refusal direction.

---

## Where This Leaves Us

| Model | Architecture | KL (best) | Approach |
|-------|-------------|-----------|----------|
| Qwen3-8B | Uniform full attention | 0.03 | Single direction, all layers |
| Gemma 4 E4B | Sliding + full attention | 0.19 | Single direction, layers 23+ only |

The remaining 6x gap between Qwen and Gemma may be irreducible given the architecture, or there may be further gains from sweeping the layer boundary, using probe-derived directions instead of diff-in-means, or ablating at only the full-attention layers within regime B.

The broader point is that the assumption underlying refusal direction ablation — that a single linear direction can be ablated everywhere — is an assumption about the architecture, not a universal property of language models. It holds for uniform-attention models like Qwen and Llama, where the residual stream maintains a relatively stable basis across layers. It breaks for hybrid-attention models like Gemma 4, where different attention patterns create distinct representational regimes.

---

## Open Questions

- Is the phase boundary at layer 23 specific to refusal, or does it show up for other linear features too? If Gemma 4 has a general "representation regime change" at this boundary, it would affect any technique that assumes a stable linear direction across layers (steering vectors, activation patching, etc.).
- Would training separate SAEs for each regime capture different features?
- The 31B Gemma model had even worse KL (~20) than the 4B model (~3–6). Does the problem scale with model size, and if so, does the layer-selective fix scale too?
- Can we identify the phase boundary automatically (e.g., from the pairwise cosine similarity matrix) rather than reading it off a plot?

---

*Code on [GitHub](https://github.com/alexayvazyan/projects). This page was initially written by Claude and edited by Alex.*
