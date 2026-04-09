---
layout: post
title: "Linear Ablation with Rotating Representations of Refusal"
date: 2026-04-09
---

# Linear Ablation with Rotating Representations of Refusal

*Refusal direction ablation works beautifully on Qwen3-8B and completely fails on Gemma 4. Investigating why led to an interesting finding: the probe-derived refusal direction in Gemma is not stable across layers — it undergoes a sharp rotation partway through the network, making the single-direction assumption invalid.*

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

After all debugging, Qwen still achieved KL=0.03, and Gemma was still at KL=3+. The gap was real and not a methodological artifact.

---

## One Architectural Difference: Sliding Attention

The most obvious structural difference between the two models is their attention pattern. Qwen3-8B uses standard full attention at every layer — each token attends to every other token. Gemma 4 uses a **hybrid attention pattern**: most layers use sliding window attention (local context only), with full attention layers interspersed every 6th layer (layers 5, 11, 17, 23, 29, 35, 41 in the 42-layer E4B model).

This is a different information flow. In Qwen, every layer has access to the full context. In Gemma, most layers only see a local window, with full-context information periodically reinjected at the full attention layers. Whether this is actually what causes the problem is unclear — there are many other differences between the two models — but it motivated looking at whether the refusal direction remains stable across layers.

---

## Direction Stability Analysis

To test this, I trained linear probes (L2-regularized logistic regression, C=0.01) at every (layer, EOI position) combination on ~800 harmful and ~800 harmless prompts. The probe's weight vector at each layer gives us the direction that best separates harmful from harmless at that layer. If the refusal direction is truly a single consistent direction, then the probe directions at adjacent layers should have high cosine similarity.

### Qwen3-8B: Smooth and Stable

![Qwen3-8B decay from best layer — smooth mountain shape](/assets/images/refusal_qwen_decay_from_best.png)
*Cosine similarity of each layer's probe direction with the best layer's direction, for each EOI position. Qwen shows a smooth, symmetric decay — the probe direction changes gradually across all 36 layers.*

The probe directions at adjacent layers have cosine similarity of 0.85–0.95 throughout the middle of the network. The pairwise cosine similarity matrix (below) shows a smooth gradient — nearby layers agree strongly, distant layers agree less, but there are no abrupt transitions.

![Qwen3-8B full direction stability — pairwise cosine matrices and adjacent similarity](/assets/images/refusal_qwen_direction_stability.png)
*Left column: pairwise cosine similarity matrix between all layer pairs. Right: adjacent layer cosine similarity (probe in blue, diff-in-means in red). Each row is a different EOI token position.*

This is the kind of pattern you'd expect to see when single-direction ablation works well. The probe direction rotates slowly across layers, so a single direction (from the best layer) has reasonable alignment everywhere. Ablating it at every layer removes something close to the local probe direction, keeping collateral damage low.

### Gemma 4 E4B: A Phase Transition

![Gemma 4 E4B decay from best layer — fragmented peaks](/assets/images/refusal_gemma4_decay_from_best.png)
*Same analysis on Gemma 4. Instead of a smooth mountain, we see fragmented peaks. The probe direction in layers 0–22 is nearly orthogonal to the probe direction in layers 23+.*

The Gemma 4 picture is dramatically different. There is a sharp discontinuity around layer 22–23 (which coincides with a sliding-to-full attention boundary, though we can't say whether that's causal). The adjacent layer cosine similarity at this transition:

```
layer 22 → 23:  +0.35  (sliding → full)  ← type boundary
layer 23 → 24:  +0.06  (full → sliding)  ← type boundary
```

A cosine similarity of 0.06 means these directions are nearly **orthogonal**. The probe direction that best separates harmful from harmless in layers 0–22 and the corresponding direction in layers 23+ are essentially unrelated vectors in the ~3000-dimensional space.

![Gemma 4 E4B full direction stability — block diagonal structure visible](/assets/images/refusal_gemma4_direction_stability.png)
*Left: pairwise cosine similarity matrices. Middle: adjacent layer cosine similarity (probe in blue, diff-in-means in red). Right: cosine similarity between probe and diff-in-means directions at each layer.*

The three panels tell different parts of the story. In the left panels, layers 0–22 show heat that stays close to the diagonal — the probe direction is drifting, with each layer only similar to its immediate neighbors. After layer 23, the structure changes to a solid block: all layers in this range share a common direction. The contrast with Qwen's smooth band is stark.

The middle panels confirm this: adjacent cosine similarity hovers around 0.5–0.8 in the early layers, then crashes at the layer 22–23 boundary before recovering to 0.8+ in the later layers. Something changes at that boundary.

The right panels show that the probe and diff-in-means directions maintain consistently strong agreement throughout (cosine ~0.8, though often negative — meaning they found the same separating direction but with flipped sign convention). This rules out the possibility that the instability is an artifact of degenerate probes or noisy diff-in-means estimates — both methods independently find similar directions at each layer, those directions just happen to rotate sharply at layer 23.

---

## Why Single-Direction Ablation Fails (Hypothesis)

The direction instability offers a plausible explanation for the catastrophic KL divergence. The best single direction typically comes from a later layer (around layer 30–36, where probe accuracy is highest). When this direction is ablated at all 42 layers:

- **Layers 23+**: the ablated direction is close to the local probe direction. Refusal is suppressed, with relatively low collateral damage.
- **Layers 0–22**: the ablated direction is nearly orthogonal to the local probe direction. Ablating it is unlikely to remove much refusal signal, but it does remove a dimension of the residual stream that may be encoding something useful at those layers.

If this is correct, the early layers are getting a random dimension removed at every layer for no benefit. To test this, I tried restricting ablation to only the later layers.

---

## The Fix: Layer-Selective Ablation

I tested six configurations, selecting the best direction for each regime by filtering to candidates with refusal score < 0.1 and then picking the one with minimum KL divergence:

| Configuration | Refusal | KL |
|--------------|---------|------|
| 1. Single direction (all layers) | 0.0206 | 1.2295 |
| 2. Regime A only (layers 0–22) | 0.0792 | 1.3573 |
| 3. **Regime B only (layers 23+)** | **0.0489** | **0.1886** |
| 4. Two-direction (A @ 0–22, B @ 23+) | 0.0246 | 1.2674 |
| 5. Single dir, regime A layers only | 0.9920 | 0.5839 |
| 6. Single dir, regime B layers only | 0.0282 | 0.8642 |

Config 3 is the clear winner: **refusal = 0.05, KL = 0.19**. This is a 6–30x improvement in KL over ablating at all layers, and refusal is still effectively suppressed. This is consistent with the hypothesis above — not ablating in the early layers avoids the collateral damage.

Two observations from the table:

**Ablating only at layers 23+ is sufficient to suppress refusal.** Config 3 achieves refusal = 0.05 — nearly as good as ablating everywhere. Config 5 ablates only at layers 0–22 and achieves refusal = 0.99 — it does essentially nothing. Whatever the mechanism, the ablation is doing its work in the later layers.

**Ablating in the early layers hurts regardless of direction.** Configs 2 and 4 both ablate in layers 0–22 and both pay a heavy KL penalty (>1.2). Even the two-direction approach (config 4), which uses a direction specifically optimized for the early layers, still causes large collateral damage. I don't have a good explanation for why the early layers are so fragile — it could be that the probe directions there are less meaningful, or that early layers are more sensitive to any perturbation.

### Sweeping the boundary

Layer 23 was chosen because it's where the phase transition appeared in the direction stability analysis. But is it actually the optimal cutoff? I swept the start layer from 20 to 35:

| Start layer | Layers ablated | Refusal | KL |
|------------|---------------|---------|------|
| 20 | 22 | 0.0514 | 0.2508 |
| 21 | 21 | 0.0466 | 0.3334 |
| 22 | 20 | 0.0480 | 0.3340 |
| **23** | **19** | **0.0489** | **0.1886** |
| 24 | 18 | 0.0474 | 0.1733 |
| 25 | 17 | 0.0454 | 0.1730 |
| 27 | 15 | 0.0466 | 0.1625 |
| 30 | 12 | 0.2208 | 0.1386 |
| 33 | 9 | 0.9304 | 0.0930 |
| 35 | 7 | 0.9688 | 0.0661 |

There's a clear operating range. Starting anywhere from layer 23–27 gives both low refusal (<0.05) and low KL (<0.19). Starting earlier (layers 20–22) maintains refusal suppression but KL rises — consistent with the early layers being sensitive to perturbation. Starting later (layer 30+) preserves KL but refusal suppression breaks down, meaning there are layers in the mid-20s that are doing meaningful refusal-related work.

The tradeoff is smooth and well-behaved within the working range: KL decreases monotonically as we ablate fewer layers, while refusal stays flat until we cut too deep. Layer 27 gives the best balance in this sweep (refusal=0.047, KL=0.16).

---

## Where This Leaves Us

| Model | Architecture | KL (best) | Approach |
|-------|-------------|-----------|----------|
| Qwen3-8B | Uniform full attention | 0.03 | Single direction, all layers |
| Gemma 4 E4B | Sliding + full attention | 0.16 | Single direction, layers 27+ only |

The remaining ~5x gap between Qwen and Gemma may be irreducible, or there may be further gains from using probe-derived directions instead of diff-in-means, or from ablating at only the full-attention layers within the later regime.

The broader takeaway is that the assumption underlying refusal direction ablation — that a single linear direction can be ablated at every layer — does not hold universally. It works on Qwen3-8B (and on the Llama models tested in the original paper), where the probe direction is stable across layers. It fails on Gemma 4, where the probe direction undergoes a sharp rotation midway through the network.

---

## Is This Specific to Refusal?

A natural question: is the direction instability something particular about how Gemma 4 represents refusal, or is it a general property of the architecture? To test this, I ran the same direction stability analysis on a completely different feature — truth — using datasets from [Marks & Tegmark (2023), "The Geometry of Truth"](https://arxiv.org/abs/2310.06824).

I trained probes at every layer on two datasets:
- **Cities**: "The city of Sharjah is in the United Arab Emirates" (true) vs "The city of Sharjah is in China" (false) — 800 train, 200 val
- **Larger-than**: "Fifty-one is larger than fifty-two" (false) vs "Sixty is larger than thirty-nine" (true) — 800 train, 200 val

![Truth direction stability for cities and larger-than datasets](/assets/images/refusal_gemma4_truth_stability.png)
*Top row: cities dataset. Bottom row: larger-than dataset. Both show the same structure as refusal — heat close to the diagonal in early layers, consolidating into a more stable block in later layers.*

The pattern looks similar to refusal. Both truth datasets show drifting probe directions in the early layers (heat stays close to the diagonal in the pairwise cosine matrix) followed by a more consolidated representation in the later layers. The adjacent cosine similarity shows periodic dips that coincide with the attention type boundaries.

The truth direction is noisier than refusal — the oscillations in adjacent cosine are larger, and the late-layer block is less cleanly defined. But the overall structure is the same: early instability, late consolidation, with some kind of transition in the same region of the network.

This suggests the direction instability is a **general property of Gemma 4's architecture**, not something specific to refusal. Any technique that assumes a stable linear direction across all layers — steering vectors, activation patching, linear ablation — should expect the same problem on this model regardless of which feature is being targeted.

---

## Open Questions

- Would training separate SAEs for each regime capture different features?
- The 31B Gemma model had even worse KL (~20) than the 4B model (~3–6). Does the problem scale with model size, and if so, does the layer-selective fix scale too?
- Can we identify the phase boundary automatically (e.g., from the pairwise cosine similarity matrix) rather than reading it off a plot?

---

*Code on [GitHub](https://github.com/alexayvazyan/projects). This page was initially written by Claude and edited by Alex.*
