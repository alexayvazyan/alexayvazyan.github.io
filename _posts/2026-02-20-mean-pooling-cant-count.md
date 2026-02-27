---
layout: post
title: "Softmax attention with mean pooling can't count"
date: 2026-02-20
---

# Softmax attention with mean pooling can't count

*A small architectural choice made a surprisingly large difference, and the reason is mathematically clean.*

---

## The Empirical Observation

While working on a [TFT placement prediction model](/2026/02/19/tft-mechanistic-interpretability.html), the number of champions on a board is one of the strongest predictive features — more champions means you survived longer into the game. The model should learn this easily. It didn't.

With mean pooling after the transformer, predictions grouped by champion count looked like this:

![Mean pooling fit](/assets/images/pooling_mean_only.png)

The tails are predicted badly. A board with 3 champions and a board with 10 champions are being treated similarly. The model can't count how many tokens it received.

Switching to sum pooling fixed it immediately:

![Sum vs Mean pooling fit](/assets/images/pooling_sum_vs_mean.png)

MAE dropped from 1.55 to 1.45, with the tails now tracked properly. But why?

## A Toy Experiment

To isolate the phenomenon, I built the simplest possible test: a model whose only job is to count how many tokens are in a sequence. Vocabulary of 120, sequences of length 1-10, target = number of non-pad tokens. Four model variants:

| Model | Best MAE |
|-------|---------|
| Bag of embeddings + sum pool | **0.000** |
| Transformer + sum pool | **0.001** |
| Transformer + mean pool | 1.252 |
| Bag of embeddings + mean pool | 2.507 |

Sum pooling solves counting perfectly — both with and without attention. Mean pooling can't do it at all. The bag of embeddings with mean pooling is the worst — it predicts ~5.5 for everything, completely blind to count. The transformer with mean pooling manages to partially count (MAE 1.25), but poorly, with large variance at every true count.

This isn't a training issue. The model architecturally cannot represent the count after mean pooling. Here's why.

## The Derivation

Consider a sequence of N tokens with embeddings e_1, ..., e_N in R^d.

**Sum pooling** gives us:

    h_sum = e_1 + e_2 + ... + e_N

If every embedding has a constant component c along some dimension (which the bias term in the projection provides), then that dimension of h_sum equals N*c. A linear head reads off N trivially.

**Mean pooling** gives us:

    h_mean = (e_1 + e_2 + ... + e_N) / N

That constant component c becomes N*c/N = c. The count has been divided out. No matter how many tokens, that dimension is always c. The count information is gone.

But wait — the transformer with mean pooling did partially learn to count (MAE 1.25 vs 2.5 for pure bag-of-embeddings). What's going on?

## What Softmax Attention Can Partially Recover

Before mean pooling, the transformer applies self-attention + FFN. Can attention help?

The attention output for token i is:

    attn_i = Σ_j softmax(q_i · k_j / √d)_j · v_j

The softmax weights sum to 1 by construction. So the attention output is a **convex combination** of value vectors — a weighted average. This is inherently count-invariant: whether there are 3 tokens or 10, the output is always a weighted average of the values, bounded within their convex hull.

If all tokens are identical (same embedding), then softmax gives uniform weights 1/N, and the attention output is just v — identical regardless of N. Mean pooling of N copies of v is still v.

But the transformer with mean pooling *did* partially learn to count. What mechanism is responsible — the attention, the FFN, or both? And can we just scale it up until it works?

## Isolating the Mechanism

To answer this, I tested every combination on a minimal setup: vocabulary of 5, sequences of length 1-4, target = count non-pad tokens. This gives only 780 possible unique inputs, so memorization is theoretically feasible with enough parameters.

Three types of model, all with mean pooling:
- **Attention only**: Embed → Multi-head attention + residual + LayerNorm → mean pool → linear head
- **FFN only**: Embed → FFN + residual + LayerNorm → mean pool → linear head (per-token FFN, no attention)
- **Full transformer**: Embed → TransformerEncoderLayer(s) (attention + FFN) → mean pool → linear head

| Model | Layers | Heads | d_model | FFN width | Params | MAE | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|-------|--------|-------|---------|-----------|--------|-----|--------|--------|--------|--------|
| Bag + mean pool | — | — | 8 | — | 65 | 0.981 | 2.42 | 2.42 | 2.42 | 2.42 |
| Bag + sum pool | — | — | 8 | — | 65 | **0.000** | 1.00 | 2.00 | 3.00 | 4.00 |
| FFN only | — | — | 8 | 32 | 633 | 0.981 | 2.42 | 2.42 | 2.43 | 2.42 |
| FFN only | — | — | 8 | 128 | 2,265 | 0.979 | 2.25 | 2.26 | 2.26 | 2.26 |
| FFN only | — | — | 8 | 512 | 8,793 | 0.979 | 2.18 | 2.18 | 2.18 | 2.18 |
| FFN only | — | — | 16 | 512 | 17,073 | 0.979 | 2.32 | 2.33 | 2.33 | 2.33 |
| Attention only | — | 1 | 8 | — | 369 | 0.450 | 1.17 | 2.20 | 2.92 | 3.25 |
| Attention only | — | 1 | 16 | — | 1,249 | 0.505 | 1.20 | 2.27 | 2.83 | 3.21 |
| Attention only | — | 1 | 32 | — | 4,545 | 0.494 | 1.20 | 2.21 | 2.76 | 3.17 |
| Full transformer | 1 | 1 | 8 | 32 | 937 | 0.397 | 1.15 | 2.12 | 2.91 | 3.33 |
| Full transformer | 1 | 1 | 8 | 128 | 2,569 | 0.251 | 1.14 | 1.97 | 2.88 | 3.67 |
| Full transformer | 1 | 1 | 8 | 512 | 9,097 | 0.215 | 1.12 | 1.96 | 2.90 | 3.73 |
| Full transformer | 2 | 1 | 8 | 128 | 5,073 | 0.212 | 1.15 | 1.92 | 2.92 | 3.73 |
| Full transformer | 1 | 4 | 16 | 64 | 3,409 | 0.250 | 1.18 | 1.95 | 2.93 | 3.68 |
| Full transformer | 2 | 4 | 16 | 64 | 6,689 | 0.214 | 1.13 | 1.91 | 2.93 | 3.72 |

Three clear findings:

**1. FFN alone does nothing.** Even with 17,073 parameters and a 512-wide FFN, the model predicts ~2.3 for every input — identical to bag + mean pool. The per-token FFN sees each token independently *before* mean pooling, so no amount of FFN width helps. This rules out memorization through the FFN.

**2. Attention is the counting mechanism.** Attention-only models get to MAE ~0.45 with just 369 parameters, correctly ordering the counts (1.17, 2.20, 2.92, 3.25). The mechanism is likely that the attention *pattern* itself varies with sequence length — with more tokens, attention entropy increases and self-attention weight decreases. This leaves a count-correlated signature in the residual stream that survives mean pooling.

**3. FFN amplifies the attention signal, but it saturates.** The full transformer (attention + FFN) gets down to MAE 0.21, better than attention alone. The FFN transforms the attention-modified representations to make the count signal more linearly readable. But scaling further doesn't help — 9,097 params and 6,689 params both plateau around MAE 0.21. With only 780 possible inputs, the model could in principle memorize them all, but it can't: the information bottleneck of mean pooling prevents it. The exhaustive test set MAE is actually *worse* than the random test MAE, confirming no memorization is occurring.

## Does Scaling Break Through?

A natural question: maybe the models above are just too small. If we scale the full transformer aggressively — more layers, more heads, wider dimensions — can it eventually learn to count through mean pooling?

| Model | Layers | Heads | d_model | FFN neurons | Params | MAE | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|-------|--------|-------|---------|-------------|--------|-----|--------|--------|--------|--------|
| Full transformer | 2 | 4 | 16 | 64 | 6,689 | 0.214 | 1.13 | 1.91 | 2.93 | 3.72 |
| Full transformer | 4 | 1 | 8 | 512 | 36,193 | 0.207 | 1.20 | 1.97 | 2.91 | 3.76 |
| Full transformer | 4 | 4 | 16 | 256 | 38,593 | **0.190** | 1.11 | 1.89 | 2.91 | 3.74 |
| Full transformer | 8 | 4 | 32 | 256 | 168,449 | 0.204 | 1.18 | 1.95 | 2.90 | 3.76 |
| Full transformer | 4 | 8 | 64 | 512 | 332,545 | 0.191 | 1.09 | 1.87 | 2.85 | 3.73 |
| Full transformer | 8 | 8 | 64 | 512 | 664,577 | 0.196 | 1.14 | 1.93 | 2.92 | 3.75 |

No. Going from 6,689 to 664,577 parameters (100x) barely moves the needle — MAE plateaus around 0.19. The predictions are stuck: true count 1 gets predicted ~1.1, true count 4 gets predicted ~3.7. The model consistently compresses the range by about 10% on each end and cannot fix this no matter how many parameters you give it.

For comparison, bag + sum pool solves this perfectly (MAE 0.000) with **65 parameters**.

## The Fundamental Issue

The core problem is compositional:

1. **Softmax normalizes** — attention weights sum to 1, so the weighted average is bounded regardless of N
2. **Mean pooling normalizes** — the sum is divided by N

Both operations independently destroy count information. Together, there's no clean path for the count to survive to the output. Attention can leak a partial signal through pattern variation, but the FFN can only amplify what attention provides — it can't create count information from scratch.

**Sum pooling** fixes this because the un-normalized sum grows linearly with N. Even if each token's representation is count-invariant after attention, summing N of them produces a vector whose norm scales with N. The linear head after pooling can simply read the norm.

In our TFT model, this meant the embedding's bias term (which is constant across all tokens) became a built-in counter after sum pooling: the bias contributes (0, 0, ..., c, ..., 0) per token, and after summing N tokens, that dimension reads N*c.

## The Practical Takeaway

If your model needs to know how many tokens it received — which is common in set-based prediction tasks where cardinality matters — mean pooling after attention will silently destroy that information. Sum pooling preserves it for free.

This seems like a small architectural choice, but in our case it was the difference between the model learning one of the two most important features in the data and being blind to it.

*This observation came up during a [mechanistic interpretability study of a TFT placement prediction transformer](/2026/02/19/tft-mechanistic-interpretability.html). Code on [GitHub](https://github.com/alexayvazyan/projects).*
