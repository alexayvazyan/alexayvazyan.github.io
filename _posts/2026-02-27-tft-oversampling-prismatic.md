---
layout: post
title: "What happens to N13 when you force the model to care about prismatic boards?"
date: 2026-02-27
---

# What happens to N13 when you force the model to care about prismatic boards?

*N13 was the only neuron encoding prismatic trait detection. So we oversampled until the model had no choice but to take it seriously — and watched it completely rewire itself.*

---

## The Setup

In the [first post](/2026/02/19/tft-mechanistic-interpretability.html), I identified a neuron — N13 — that appeared to detect "trait-dense + emblem" boards: compositions where a single trait reaches a critical mass of units plus matching emblems. The clearest case of this is a **prismatic board**, where one trait reaches 10 units (e.g. 8 Bilgewater champions + 2 Bilgewater emblems), which in the actual game triggers a powerful bonus and nearly guarantees first place.

The model had clearly picked something up here. N13 activated on heavy trait boards with matching emblems and was near-zero everywhere else. But N13's weight toward 1st place (+0.33) was modest compared to the emblem-presence neurons (+1.0–+1.8), and the whole thing felt fragile — a weak signal riding in on co-occurrence patterns from only ~120 samples out of 24,344.

The natural follow-up was: what if we gave the model more of these boards? Not synthetically generated — just oversample the real ones. If the signal is genuinely there, the model should learn a cleaner, stronger version. If N13 was incidental, it should fall apart.

I trained six fresh models from scratch: oversample factors ×1 (baseline), ×5, ×10, ×25, ×50, ×100. Everything else held constant — same architecture, same test set, same early stopping.

**Prismatic definition**: any board where a single trait reaches ≥10 units (champions + matching emblems). There are 118 such boards in the dataset, 88% of which place 1st. At ×100, prismatic boards make up 36.6% of the training set.

---

## What Happened to N13

Short answer: it died at ×5 and never came back.

![N13 activation on prismatic vs near-prismatic vs control boards across oversample factors](/assets/images/oversample_n13_collapse.png)
*N13 fires strongly on prismatic boards at ×1. At ×5 it collapses to near-zero on everything. It never recovers a specific prismatic role at any higher factor.*

At ×1, N13 fires at ~1.45 on prismatic boards and 0.00 on controls — reasonably clean. At ×5, it drops to ~0.07 on prismatic and 0.13 on controls. It's not just weakened, it's lost specificity entirely. At ×25 it reappears slightly (+0.12 specificity), but simultaneously fires equally on near-prismatic and control boards — it's no longer doing anything useful for prismatic detection. At ×100, it's zero everywhere.

N13 was not a stable "prismatic neuron." It was whatever the model happened to use in one run under low-signal conditions. When the signal got stronger, the model rewired completely and N13 became irrelevant.

---

## The Neurons That Replaced It

Here's where things got interesting. The model didn't just lose N13 — it replaced it with a completely different set of neurons at every oversampling level, each with *higher* specificity than N13 ever had.

| Factor | New prismatic neurons (prismatic−control gap)                           |
|--------|-------------------------------------------------------------------------|
| ×1     | N13 (+1.45), N12 (+0.82), N23 (+0.63)                                  |
| ×5     | N1 (+1.07), N17 (+1.02), N27 (+0.94)                                   |
| ×10    | N18 (+3.18), N16 (+2.60), N29 (+2.45), N10 (+1.43)                     |
| ×25    | N17 (+1.73), N31 (+1.72), N27 (+1.51), N26 (+1.32), N1 (+1.23)         |
| ×50    | N19 (+2.42), N23 (+2.00), N26 (+1.93), N25 (+1.76), N1 (+1.53)         |
| ×100   | N18 (+3.26), N0 (+3.04), N11 (+2.11), N28 (+2.08), N30 (+1.89)         |

At ×1, the prismatic signal is scattered weakly across a few neurons with gaps of ~0.6–1.5. At ×10+, 4–6 neurons each achieve gaps of 2–3+, all with near-zero activation on both near-prismatic and control boards. The representation has become **cleaner and stronger** — the model is allocating dedicated capacity with a much more binary activation profile.

The heatmap makes the instability obvious:

![Heatmap of neuron prismatic-vs-control gaps across all 32 neurons and 6 oversample factors](/assets/images/oversample_neuron_heatmap.png)
*Each row is a neuron, each column is an oversample factor. Red = fires more on prismatic than controls, blue = fires less. The gold box highlights N13. No neuron consistently encodes prismatic across all factors — the "winning" neurons change entirely between runs.*

No single neuron appears consistently. N18 appears at ×10 (+3.18) and ×100 (+3.26), but is near-zero at ×5, ×25, ×50. N17 is strong at ×5 (+1.02) and ×25 (+1.73) but disappears at ×50 (−2.91, actively suppressed). The model reaches a different local minimum each time, and different neurons fill the functional slot.

Also notable: the feature splits. At ×1 one neuron roughly handles it. At ×10+, 4–6 neurons each carry part of the load. Whether this is superposition being resolved (multiple features encoded in one neuron, then separated), or just a natural consequence of a stronger gradient signal recruiting more capacity, I'm not sure. Worth looking at in a bigger model where you'd expect cleaner superposition dynamics.

---

## The Activation Profile Gets Cleaner

At ×1, N13 fired at 1.45 on prismatic and 0.49 on near-prismatic — not sharp. The model was partially activating on 8-trait boards even without matching emblems, suggesting it was picking up something about champion density rather than the specific trait+emblem combination.

At ×10 and ×100, the new winning neurons (N18, N0, N11) activate at 2–3+ on prismatic boards and are essentially zero on near-prismatic and controls. The model has learned a more **precise binary detector**: trait+emblem combinations trigger it, neither trait alone nor emblem alone does.

| Board type | N13 activation (×1) | Best neuron activation (×10) |
|------------|---------------------|------------------------------|
| Prismatic (8 trait + 2 matching emb) | 1.45 | 3.18 (N18) |
| Near-prismatic (8 trait, no emb) | 0.49 | 1.06 (N18) |
| Control + emblems | 0.00 | 0.41 (N18) |

That said, this improvement is partly illusory: the ×10 model is seeing 5.5% prismatic training data vs 0.6% for ×1. It has much more direct supervision signal. Whether this constitutes a more principled understanding of the trait mechanic, or just better memorization of the specific champion+emblem configurations that appear in the training set, is hard to say. My suspicion is it's mostly the latter — the same co-occurrence logic at higher signal.

---

## Predictions Get Better, Mostly

![Predicted placement for prismatic, near-prismatic, and control boards across oversample factors, with test MAE on secondary axis](/assets/images/oversample_predictions.png)
*Predicted placement (lower = better) for the three board categories across oversample factors. Test MAE (right axis, also lower = better) is non-monotone: best at ×50.*

The model's predictions for prismatic boards improve immediately with even small oversampling — from ~1.53 at ×1 to ~1.31 at ×5 and ×10. Near-prismatic and control board predictions also improve in contrast (the gap widens), which is good. The separation between prismatic and control grows substantially at ×50: prismatic gets a predicted placement of 1.40 while controls hit 5.00.

What's more interesting is the overall test MAE, which is non-monotone. ×25 actually performs *worse* than baseline (1.4792 vs 1.4679). The best overall MAE is at ×50 (1.4588). So moderate oversampling helps; extreme oversampling (×100) starts to hurt the model on the common cases that make up most of the test set.

---

## Attention Still Doesn't Care

Across all six models, the fraction of attention directed at emblem tokens on prismatic boards (0.043–0.066) was statistically indistinguishable from that on control boards with emblems (0.035–0.070). Attention entropy showed no consistent decrease on prismatic boards at any factor.

Even when prismatic boards are 36.6% of training data, the attention head doesn't learn to attend to emblems differently. The prismatic circuit lives entirely in embedding → sum pool → MLP. The attention head can't participate here — detecting "this specific champion + this specific matching emblem = trait bonus" would require attending to relationships between tokens, and a single rank-1 attention head with collapsed Q/K matrices (as described in the first post) simply doesn't have that capacity. The problem would need either a larger model or a fundamentally different architecture.

---

## The Real Lesson: Neuron Identity Is Arbitrary

This experiment started as a question about N13. It ended up revealing something more fundamental: **neuron-level mechanistic findings from a single training run aren't reproducible**. The neuron assigned to a function is determined by random initialization and gradient descent converging to one of many equivalent local minima — not by anything intrinsic about what that function needs.

The *circuit structure* is reproducible: give the model enough prismatic signal and it will always produce a set of dedicated neurons with high specificity and near-zero cross-activation. The *neuron identity* — which one of the 32 neurons fills that role — is not. N13 was just a name for a functional slot. If I had trained the original model with a different random seed, it might have been N7 or N22 or any other.

This has a practical implication: when reporting mechanistic interpretability results, the interesting thing to describe is the activation profile and the functional role, not the neuron index. "N13 detects prismatic traits" is a finding about one training run; "the model dedicates 1–6 neurons to detecting trait-dense + matching-emblem compositions, with near-zero activation on controls" is the actual generalizable result.

---

## Open Questions

- The feature splits from ~1 neuron at ×1 to 4–6 neurons at ×10+. Is this resolution of superposition, or just gradient signal recruiting more capacity? A model with d_model=16 or 32 would have more room for superposition and might show a cleaner version of this.
- Can we systematically find which training samples have the largest gradient contribution to a specific neuron, and oversample those? This could be a more principled version of what we did here.
- If we add 4 attention heads instead of 1, does attention ever learn to track emblem-champion co-occurrence? The first post noted that grokking in a related experiment required exactly 4 heads — maybe that's the minimum for compositional detection to emerge.

*The code for this project is on [GitHub](https://github.com/alexayvazyan/projects).*
