---
layout: post
title: "Softmax attention with mean pooling - can it count?"
date: 2026-02-20
---

# Softmax attention with mean pooling - can it count?

*A small architectural choice made a surprisingly large difference, and the reason is mathematically clean.*

---

## The Empirical Observation

While working on a [TFT placement prediction model](/2026/02/19/tft-mechanistic-interpretability.html), the number of champions on a board is one of the strongest predictive features — more champions means you survived longer into the game. The model should learn this easily. It didn't.

With mean pooling after the transformer, predictions grouped by champion count looked like this:

![Mean pooling fit](/assets/images/pooling_mean_only.png)

The tails are predicted badly. A board with 3 champions and a board with 10 champions are being treated similarly. The model was struggling count how many tokens it received.

Switching to sum pooling fixed it immediately:

![Sum vs Mean pooling fit](/assets/images/pooling_sum_vs_mean.png)

MAE dropped from 1.55 to 1.45, with the tails now tracked properly. But why?

## A Toy Experiment

My initial hypothesis was that a Softmax Attention + Mean Pool transformer wouldn't be able to count how many tokens were passed in.

To isolate the phenomenon, I built the simplest possible test: a model whose only job is to count how many tokens are in a sequence. Vocabulary of 120, sequences of length 1-10, target = number of non-pad tokens. Four model variants:

| Model | Best MAE |
|-------|---------|
| Bag of embeddings + sum pool | **0.000** |
| Transformer + sum pool | **0.001** |
| Transformer + mean pool | 1.252 |
| Bag of embeddings + mean pool | 2.507 |

Sum pooling solves counting perfectly — both with and without attention. The bag of embeddings with mean pooling is the worst — it predicts ~5.5 for everything, completely blind to count. The transformer with mean pooling manages to partially count (MAE 1.25), but poorly, with large variance at every true count.

This was a relatively surprising result. The toy replica transformer could count, just badly?

Naturally this led me to a new hypothesis. If we were seeing some loss reduction compared to baseline, it seemed like the model was doing some sort of memorization across permutations, and the fact that the loss was not 0 was just a result of the fact that the model was not large enough. After all, for 120 vocab and a max of 10 seq lengths, there are just about (>) 10^120 permutations. Maybe we just need to scale the model up?


## Isolating the Mechanism

To start testing, I ran every combination of components on a minimal setup: vocabulary of 5, sequences of length 1-4, target = count non-pad tokens. This gives only 780 possible unique inputs, so memorization is theoretically feasible with enough parameters.

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

**1. FFN alone does nothing.** Even with 17,073 parameters and a 512-wide FFN, the model predicts ~2.3 for every input — identical to bag + mean pool. The per-token FFN sees each token independently *before* mean pooling, so no amount of FFN width helps. This makes perfect intuitive sense.

**2. Attention can count, but struggles more than expected.** It seems attention hits a hard wall of ~0.5 MAE, compared to the baseline of 1. Scaling seems to not help it. Another interesting finding, there appears to be some asymetry in predictions (the 4 count predictions avg ~3.2 whereas each other class gets predicted with what looks like a ~0.2 bias towards the centre mass). 

**3. FFNs help Attention (Full Transformer) but still fail to break through.** Completing the full transformer circuit immediately yields large gains, taking us down from 0.5 -> 0.21 MAE quickly, but again hitting a wall rather early.

All of this raises quite a few questions. First, lets rerun a scaling experiment with just the Full Transformer.


## Does Scaling Break Through?

Our hypothesis predicts that with enough parameters, the model should eventually memorize its way to zero error. The models above are small — maybe they just need more capacity. If we scale the full transformer aggressively — more layers, more heads, wider dimensions — can it eventually learn to count through mean pooling?

| Model | Layers | Heads | d_model | FFN neurons | Params | MAE | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|-------|--------|-------|---------|-------------|--------|-----|--------|--------|--------|--------|
| Full transformer | 2 | 4 | 16 | 64 | 6,689 | 0.214 | 1.13 | 1.91 | 2.93 | 3.72 |
| Full transformer | 4 | 1 | 8 | 512 | 36,193 | 0.207 | 1.20 | 1.97 | 2.91 | 3.76 |
| Full transformer | 4 | 4 | 16 | 256 | 38,593 | **0.190** | 1.11 | 1.89 | 2.91 | 3.74 |
| Full transformer | 8 | 4 | 32 | 256 | 168,449 | 0.204 | 1.18 | 1.95 | 2.90 | 3.76 |
| Full transformer | 4 | 8 | 64 | 512 | 332,545 | 0.191 | 1.09 | 1.87 | 2.85 | 3.73 |
| Full transformer | 8 | 8 | 64 | 512 | 664,577 | 0.196 | 1.14 | 1.93 | 2.92 | 3.75 |

No. Going from 6,689 to 664,577 parameters (100x) barely moves the needle — MAE plateaus around 0.19. The predictions are stuck: true count 1 gets predicted ~1.1, true count 4 gets predicted ~3.7. The model consistently compresses the range by about 10% on each end and cannot fix this no matter how many parameters you give it.


## Where Does the 0.19 Wall Come From?

The next step is something that, in hindsight, I should have done considerably earlier. Lets actually look at what we are getting wrong.

Things get illuminated so quickly. We see errors immediately in sequences such as `A`, `AA`, `AAA`, `AAAA`. All of these examples predict the exact same score. A quick look at how our transformer operates makes it pretty clear why. Each token will attend the exact same with each other token as it does with itself (there is no positional embedding, else this would be a piece of cake). Therefore, when the softmaxed attended vectors are averaged out, they produce just the same result as if a singular token was passed in. Similar arguments hold aswell for sequences like `AB` and `AABB`.

If we look at how we defined the sample space and loss function, we can numerically explain this wall. The sampling mechanism was to first sample a uniform digit from 1 to 4, to decide the seq length. Then, we sample from all the possible permutations in that tier. There are 5 samples possible for l=1, 25 for l=2, 125 for l=3 and 625 for l=4. We can obtain a representative sample by taking 625*4 samples, 2500. Lets assume we predict perfectly for all permutations without internal repetition. For those with repetition, we use the MSE optimal prediction (as our loss function in our experiments was MSE). 

For the single token class (`A`, `AA`, `AAA`, `AAAA`) our representative sample will have 125 l=1 samples, 25 l=2 samples, etc. The MSE optimal prediction comes out to 1.244. The total error for this class would be 304.68 on a sample of 2500, as there are 5 such members of this class.

For the dual token class (`AB`, `AABB`) our representative sample will have 50 l=2 samples, 6 l=4 samples. MSE optimal -> 2.214. The total error for this class would be 214.16 on a sample of 2500, as there are 10 such members of this class.

Out of our 2500 samples, 156*5 + 56*10 = 1340 will have error. The sum of these errors would be 518.84. A theoretical minimum MAE of 0.207536, very inline with the empirical findings!
The results also align with the average predicted result for l=1 being slightly larger and for the average predicted result being shrinked for all other metrics.

## Removing the Collisions

To test this directly, I filtered out all sequences that are composed entirely of repeated identical sub-sequences — the sequences where `gcd(count_A, count_B, ...) > 1`. This removes `AA`, `AABB`, `AAA`, `AAAA`, `ABAB`, etc. from both training and test sets. If the 0.19 wall is caused by these proportion collisions, removing them should let the model reach near-zero MAE.

| Model | Layers | Heads | d_model | FFN neurons | Params | MAE |
|-------|--------|-------|---------|-------------|--------|-----|
| Bag + mean pool | — | — | 8 | — | 65 | 1.005 |
| Bag + sum pool | — | — | 8 | — | 65 | **0.000** |
| FFN only | — | — | 8 | 128 | 2,265 | 1.005 |
| Attention only | — | 1 | 8 | — | 369 | 0.310 |
| Full transformer | 1 | 1 | 8 | 128 | 2,569 | 0.040 |
| Full transformer | 2 | 4 | 16 | 64 | 6,689 | 0.007 |
| Full transformer | 4 | 4 | 16 | 256 | 38,593 | **0.000** |

The transformer + mean pool achieves **MAE 0.000** — perfect counting — once proportion collisions are removed from the data. This proves that the 0.19 wall was *exactly and entirely* caused by proportion collisions, not by any other architectural limitation.

The progression is also telling: the full transformer at 2,569 params already gets to 0.040, and at 6,689 params it's at 0.007. The model is genuinely learning to distinguish sequences with different proportions — it just can't distinguish sequences with *identical* proportions, because after mean pooling they are mathematically identical representations.


## The Fundamental Issue

The core problem is compositional:

1. **Softmax normalizes** — attention weights sum to 1, so the weighted average is bounded regardless of N
2. **Mean pooling normalizes** — the sum is divided by N

Both operations independently destroy count information. Together, there's no clean path for the count to survive to the output. Attention can leak a partial signal through memorization of permutations, but the FFN can only amplify what attention provides — it can't create count information from scratch.

**Sum pooling** fixes this because the un-normalized sum grows linearly with N. Even if each token's representation is count-invariant after attention, summing N of them produces a vector whose norm scales with N. The linear head after pooling can simply read the norm.

In our TFT model, this meant the embedding's bias term (which is constant across all tokens) became a built-in counter after sum pooling: the bias contributes (0, 0, ..., c, ..., 0) per token, and after summing N tokens, that dimension reads N*c. This is a much more efficient way of storing this count information, as opposed to forcing the model to figure it out itself, which is why we saw such a large improvement.

## The Practical Takeaway

Exploting your model's architecture to build in important features is a relatively obvious way to inject some bias in to achieve much better loss for the same size of model. What was interesting was the underlying mechanisms that lead to this better loss. Over this deepdive, I've had a number of my hypothesis disproven, and its been a lesson in the value of empircal testing against intuition's limitations.

*This observation came up during a [mechanistic interpretability study of a TFT placement prediction transformer](/2026/02/19/tft-mechanistic-interpretability.html). Code on [GitHub](https://github.com/alexayvazyan/projects). This page was initially written by Claude and edited extensively by Alex*
