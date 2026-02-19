---
layout: post
title: "Mechanistic Interpretability of a 2,424-Parameter TFT Transformer"
date: 2026-02-19
---

# Mechanistic Interpretability of a 2,424-Parameter TFT Transformer

*What does a tiny model actually learn about Teamfight Tactics — compositional rules, or statistical shortcuts?*

---

## Why This Project

Teamfight Tactics (TFT) has a clean compositional structure: champions belong to traits, trait counts hit breakpoints that unlock bonuses, and strong compositions reach high-count "prismatic" thresholds (e.g., 10 Bilgewater). A model that truly understood TFT would learn to count trait units and recognize breakpoints.

I wanted to practice mechanistic interpretability on a model small enough to fully examine every parameter. The goal wasn't to build a good predictor — it was to understand *exactly what* the model learned, form hypotheses about its internal representations, and test those hypotheses with controlled experiments. A 2,424-parameter transformer is small enough that we can inspect every embedding, every attention pattern, and every MLP neuron, and know whether the model has learned the real rules or found shortcuts.

## The Model

**Task**: Predict TFT placement (1st–8th) from a board composition (up to 15 champions + 5 emblems).

**Architecture**:
- **Unified token sequence**: Champions and emblems share a single vocabulary (124 tokens: 1 pad + 101 champions + 22 emblems) and attend to each other in the same sequence
- `input_proj`: Linear(124, 8) — one-hot vocabulary to 8-dimensional embeddings
- 1 TransformerEncoder layer, 1 attention head, feedforward dim 32
- **Sum pooling** over non-padded tokens
- MLP head: Linear(8, 32) → ReLU → Dropout → Linear(32, 8)
- Output: softmax over 8 placements → expected placement via weighted sum
- Loss: CrossEntropy + MAE (ordinal-aware)
- **Total: 2,424 parameters**

**Data**: ~24,344 boards from Challenger players (OCE server, Set 16), scraped via Riot API. 50/50 train/test split.

**Training**: Up to 1000 epochs with early stopping (patience 50), Adam with weight decay 1e-4, ReduceLROnPlateau.

![Training curve showing train and test MAE converging around 1.48](/assets/images/training_curve.png)
*The model converges to a test MAE of ~1.48 placements. For context, always guessing the middle placement gives MAE 2.0.*

## Finding 1: Sum Pooling Preserves Count Information

An early architectural choice had a surprisingly large effect. Switching from mean pooling to sum pooling improved test MAE from 1.557 to 1.483 — a 5% improvement with no other changes.

The reason is straightforward: mean pooling divides by token count, destroying information about *how many* units are on the board. With sum pooling, the magnitude of the pooled vector scales with board size. This matters because champion count is strongly predictive in the data:

| Champions | Data avg placement | Sum pool prediction | Mean pool prediction |
|-----------|-------------------|--------------------|--------------------|
| 1–3       | 3.25              | ~3.6               | ~5.2               |
| 6         | 6.71              | 6.65               | 6.40               |
| 10        | 2.77              | 3.11               | 3.28               |
| 12        | 1.89              | **1.87**            | 2.66               |

Sum pooling dramatically improved predictions at the extremes. Mean pooling couldn't distinguish a 1-unit board from a 12-unit board. This also has downstream effects on what the model can learn — with sum pooling, even weak champions need coherent embeddings because their count contributes to the pooled vector's magnitude.

## Finding 2: Champion Strength as a Binary Feature

The dominant feature in the 8-dimensional embedding space is **individual champion strength** — the average placement when that champion appears in the training data.

![Cosine similarity matrix showing strong champions clustered together](/assets/images/cosine_similarity_by_strength.png)
*Champions sorted by average placement. Strong champions (left/top) have high pairwise cosine similarity. Weak champions (right/bottom) are scattered.*

The evidence:
- **PC1 correlates with champion strength at r = 0.55–0.61**
- Strong champions (avg placement < 3.9) cluster tightly — mean pairwise cosine similarity 0.56–0.67
- Weak champions are scattered — mean pairwise cosine similarity 0.02–0.15
- Embedding **norm** does not correlate with strength (r = −0.047) — the signal is purely directional

This is a **binary** feature, not a linear scale. The model has learned a "strong champion" direction in embedding space. Strong champions agree on this direction; weak champions point in many directions. With sum pooling, stacking many strong champions produces a pooled vector with large magnitude along the "strong" direction, predicting low (good) placement.

![Strength analysis showing binary clustering of champion embeddings](/assets/images/strength_analysis.png)
*The binary nature of the strength feature: strong champions form a tight cluster in embedding space, while weak champions scatter across multiple directions.*

## Finding 3: Attention is Nearly Uniform

The single attention head distributes attention almost uniformly across all tokens. In a 12-token board, each token receives roughly 8% of attention — close to the uniform 1/12 = 8.3%.

![Attention patterns showing near-uniform distribution](/assets/images/attention_patterns.png)
*Attention weights across a sample board. There is no strong selective attention between same-trait champions or between emblems and their associated champions.*

I tested this with a controlled experiment using a 10 Bilgewater composition (8 BG champions + 2 fillers + 2 BG emblems): every token paid ~17% attention to emblems and ~83% to champions, roughly proportional to token count.

**Residual stream decomposition** confirms this. With a single transformer layer, we can cleanly decompose predictions into the direct path (embeddings only) and the attention-modified path:

| Composition | Full model | Direct path | Attention shift |
|-------------|-----------|-------------|-----------------|
| 10 BG (with emblems) | 2.57 | 3.89 | −1.32 |
| 10 Yordle (with emblems) | 6.32 | 3.88 | +2.43 |
| 8 BG (no emblems) | 3.88 | 3.96 | −0.09 |

Without emblems, the attention layer contributes almost nothing (−0.09). The model is essentially a **bag-of-champions**: sum the embedding vectors and predict from the aggregate. Attention only matters when emblems are present — and even then, it doesn't always help (the Yordle case actually gets worse).

## Finding 4: An Emblem Tier List, Not Trait Counting

This is where hypotheses start getting falsified. The natural hypothesis: the model learned that emblems boost boards when they match the board's trait. The controlled experiments tell a different story.

**Experiment 1**: Take 8 Bilgewater champions + 2 fillers, add different emblems:

| Emblems added | Prediction |
|--------------|-----------|
| 2 BG (correct match) | 1.57 |
| 2 Yordle (wrong) | 2.62 |
| 2 Warden (wrong) | 3.75 |
| 2 Noxus (wrong) | 6.94 |
| None | 5.11 |

Looks reasonable — matching emblems help most. But:

**Experiment 2**: Take 8 Yordle champions + 2 fillers:

| Emblems added | Prediction |
|--------------|-----------|
| 2 Yordle (correct match) | 1.76 |
| 2 BG (**wrong**) | **1.44** |

BG emblems on a Yordle board predict *better* than the correct Yordle emblems. The model has learned an **emblem tier list** — BG > Yordle > Warden > Noxus — rather than understanding trait matching.

**Experiment 3** (the critical test): 8 Noxus champions + 2 fillers:

| Emblems | Noxus count | Prediction | Should be |
|---------|------------|-----------|-----------|
| 2 Noxus (correct) | 10 | 1.95 | ~1.1 |
| 2 BG (**wrong**) | 8 | **1.53** | bad |

The model thinks 8 Noxus + 2 BG emblems is *better* than 10 Noxus. It hasn't learned trait counting.

## Finding 5: But Emblems Aren't Universally "Good"

Before concluding the model is purely memorizing an emblem tier list, I needed a control: what happens when you add "good" emblems to boards that don't need them?

| Board | No emblems | +2 BG emblems | +2 Yordle emblems |
|-------|-----------|--------------|------------------|
| Control 1 (Piltover-heavy) | 2.23 | 2.30 (+0.07) | 2.08 (−0.15) |
| Control 2 (Demacia-heavy) | 2.66 | 2.59 (−0.07) | 3.02 (+0.36) |
| 8 BG + 2 fillers | 3.64 | **1.16** (−2.48) | — |
| 8 Yordle + 2 fillers | 2.28 | — | **1.44** (−0.84) |

On control boards, emblems barely matter (±0.3 placement). On trait-heavy boards, they shift predictions by 1–2.5 placements. The emblem effect is **contextual** — it depends on which champions are present.

The model has learned champion-emblem co-occurrence patterns from training data rather than the compositional rule "count matching trait units → breakpoint bonus." It knows that BG emblems appear alongside BG champions in winning boards, but it hasn't abstracted the *reason*.

## Finding 6: N13 — A Contextual Emblem Detector

The MLP head has 32 hidden neurons, each contributing to the final placement prediction. Most encode diffuse features, but one stands out.

**Neuron 13** pushes the prediction toward 1st place, and its activation pattern is remarkably clean:

| Board condition | N13 activation |
|----------------|---------------|
| 8 BG + 2 BG emblems | +0.98 |
| 8 Yordle + 2 Yordle emblems | +0.95 |
| 8 Ionia + 2 Ionia emblems | +1.00 |
| Control board 1 + 2 BG emblems | **0.00** |
| Control board 2 + 2 BG emblems | **0.00** |

N13 fires when matching emblems appear on trait-heavy boards, and *does not fire* when the same emblems appear on control boards. This is the most interpretable finding in the model: a single neuron has learned to detect "emblems on a synergy-heavy board" — a genuine contextual feature, not just emblem presence.

Other MLP neurons show more trait-specific patterns:
- **N8** detects BG emblem presence (fires regardless of board context)
- **N26** is Noxus-specific (suppressed by Noxus emblems on Noxus boards)
- **N16** and **N15** are suppressed by Yordle and Ionia emblems respectively on matching boards

The MLP uses a mix of generic and trait-specific neurons. The generic ones (N13) capture "emblems help trait boards." The trait-specific ones respond to particular emblem types but haven't fully learned that the emblem *must* match the board's trait.

## The Pooled Representation

The 8-dimensional sum-pooled vector — the only input to the MLP — encodes a smooth placement gradient:

![Pooled vector analysis showing 8D encoding of placement information](/assets/images/pooled_analysis.png)
*Mean pooled vector by placement, showing smooth gradients across multiple dimensions from 1st to 8th place.*

| Dimension | Correlation with placement | Role |
|-----------|--------------------------|------|
| D0 | r = −0.40 | Strongest "good board" signal |
| D6 | r = +0.40 | Second strongest |
| D5 | r = +0.33 | Third |
| D7 | r = −0.30 | Fourth |
| D2, D3 | r ≈ 0 | Non-placement features |

The model distributes placement information across multiple dimensions rather than using a single "good/bad" axis. Adding matching emblems consistently shifts D5 up, D6 down, and D1 down — a consistent "emblem boost" direction in 8D space regardless of which trait.

## Residual Embedding Structure

After projecting out the dominant strength direction, the remaining 7 dimensions show weak trait signals:

![Residual clusters after removing strength direction](/assets/images/residual_clusters.png)
*Embedding structure after removing the strength axis. Trait groupings are visible but weak.*

Trait signals are present but weak (gaps of 0.1–0.26 between trait clusters). The model has not cleanly separated traits into distinct directions — they're superimposed across multiple dimensions. Notably, emblem tokens do **not** land near their trait's champions in this residual space (only 2 of 21 emblems have their nearest neighbor sharing the same trait).

## Summary: Shortcuts vs. Understanding

The model predicts placement through a hierarchy of shortcuts:

1. **Champion identity** (dominant): Sum the "strength" vectors of individual champions. More strong champions → lower predicted placement.
2. **Token count**: More tokens = later in the game = generally stronger board. Sum pooling encodes this directly in vector magnitude.
3. **Emblem tier list**: Certain emblems (BG, Yordle) are associated with winning boards in training data. They shift the pooled vector in a consistent "good" direction.
4. **Contextual co-occurrence**: The emblem effect is partially contextual — BG emblems help more on BG-heavy boards — but this is co-occurrence memorization, not trait counting.

What the model has **not** learned:
- Trait counting (how many units share a trait)
- Trait breakpoints (thresholds like 10 Bilgewater that trigger bonuses)
- That an emblem must match the board's trait to be useful
- Compositional rules that transfer to unseen champion combinations

## What I Took Away

**On interpretability methodology**: The most productive approach was forming specific, falsifiable hypotheses and designing controlled experiments. "The model uses trait matching" is a hypothesis; swapping BG emblems onto a Yordle board is the experiment that falsifies it. Working with a 2,424-parameter model meant I could always ground my hypotheses in actual parameter values rather than relying on behavioral tests alone.

**On model behavior**: Even tiny models find statistical shortcuts. The model achieves reasonable accuracy (MAE 1.48 vs. 2.0 baseline) without learning any of the compositional rules that actually govern TFT. This is a miniature version of a broader question in interpretability: when a model performs well on a benchmark, which of the task's actual rules has it internalized, and which has it approximated with heuristics?

**On architecture choices**: Sum vs. mean pooling — a one-line code change — fundamentally altered what the model could represent. It's a reminder that architectural decisions constrain the hypothesis space before training even begins.

## Open Questions

- Can a larger model (d_model=16 or 32) learn true trait counting, or is the training data too sparse for compositional features?
- Would explicit count features (e.g., trait count vectors appended to the pooled representation) enable breakpoint learning?
- Is the attention head doing anything useful, or would a pure bag-of-embeddings model perform identically?
- Could data augmentation — permuting champion order, synthetically generating prismatic boards — help the model learn sparse compositional features?
- In a slightly larger model, would we find evidence of superposition (more decodable features than embedding dimensions)?

---

*The code for this project is on [GitHub](https://github.com/alexayvazyan/cpplearning).*
