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

The per-token FFN inside the TransformerEncoderLayer is what provides the partial counting ability. The FFN can transform each token's representation *after* attention, and the slight variations in attention output across different sequence lengths give the FFN something to work with. But this is an indirect, lossy signal.

## The Fundamental Issue

The core problem is compositional:

1. **Softmax normalizes** — attention weights sum to 1, so the weighted average is bounded regardless of N
2. **Mean pooling normalizes** — the sum is divided by N

Both operations independently destroy count information. Together, there's no clean path for the count to survive to the output.

**Sum pooling** fixes this because the un-normalized sum grows linearly with N. Even if each token's representation is count-invariant after attention, summing N of them produces a vector whose norm scales with N. The linear head after pooling can simply read the norm.

In our TFT model, this meant the embedding's bias term (which is constant across all tokens) became a built-in counter after sum pooling: the bias contributes (0, 0, ..., c, ..., 0) per token, and after summing N tokens, that dimension reads N*c.

## The Practical Takeaway

If your model needs to know how many tokens it received — which is common in set-based prediction tasks where cardinality matters — mean pooling after attention will silently destroy that information. Sum pooling preserves it for free.

This seems like a small architectural choice, but in our case it was the difference between the model learning one of the two most important features in the data and being blind to it.

*This observation came up during a [mechanistic interpretability study of a TFT placement prediction transformer](/2026/02/19/tft-mechanistic-interpretability.html). Code on [GitHub](https://github.com/alexayvazyan/projects).*
