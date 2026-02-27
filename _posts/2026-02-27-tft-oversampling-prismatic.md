---
layout: post
title: "How does a model's prismatic detection circuit evolve under signal pressure?"
date: 2026-02-27
---

# How does a model's prismatic detection circuit evolve under signal pressure?

*Prismatic boards are the rarest and most predictive feature in the dataset. We oversampled them up to ×100 to watch what the model does as the signal grows.*

---

## The Setup

In the [first post](/2026/02/19/tft-mechanistic-interpretability.html), I found that the model had learned something about **prismatic boards** — compositions where a single trait reaches 10 units (e.g. 8 Bilgewater champions + 2 Bilgewater emblems). These trigger a powerful in-game bonus and nearly guarantee first place. The model was predicting them reasonably well, and tracing back through the MLP revealed a small set of neurons with elevated activation on these boards.

But the encoding felt weak and noisy — understandable, given that only ~120 of 24,344 boards are prismatic (~0.5%). The model isn't being asked to care about them much. The natural question was: if we forced it to, what would the circuit actually look like? How many neurons would it use, how cleanly would they fire, and would the attention head ever get involved?

I trained six models from scratch with oversample factors ×1 (baseline), ×5, ×10, ×25, ×50, ×100. Everything else held constant — same architecture, same test set, same early stopping. At ×100, prismatic boards are 36.6% of the training set.

**Prismatic**: any board where a single trait reaches ≥10 units (champions + matching emblems). 88% place 1st in the real data.

Probe boards used throughout:
- **Prismatic**: 8 trait champions + 2 fillers + 2 matching emblems = 10 of that trait
- **Near-prismatic**: same 8 trait champions, no emblems (8 of trait)
- **Controls**: non-trait boards with and without emblems

---

## The Baseline Circuit Is Weak and Noisy

At ×1, a handful of MLP neurons activate more on prismatic boards than controls, with the largest gaps around +0.6 to +1.5. But the specificity is poor: the leading neurons also partially fire on near-prismatic boards (8 trait champions, no emblems), suggesting they're picking up on champion density rather than the specific trait+emblem combination that defines a prismatic board.

| Board type | Best neuron activation (×1) |
|------------|------------------------------|
| Prismatic (8 trait + 2 matching emb) | ~1.45 |
| Near-prismatic (8 trait, no emb)     | ~0.49 |
| Control + emblems                    | ~0.00 |

The model is not cleanly detecting "this board has reached the prismatic breakpoint." It's doing something messier — responding to high trait density with some additional response when matching emblems are also present. A weak, blurry version of the feature.

---

## The Circuit Sharpens Dramatically With More Signal

This is the main result. As oversampling increases, the circuit changes in two important ways: it recruits **more neurons**, and each of those neurons develops a **cleaner, more binary** activation profile.

![Heatmap of neuron prismatic-vs-control gaps across all 32 neurons and 6 oversample factors](/assets/images/oversample_neuron_heatmap.png)
*Each row is a neuron, each column is an oversample factor. Red = fires more on prismatic than controls, blue = fires less. At ×1 a couple of neurons show weak red. By ×10–×100, several neurons show deep red with much higher magnitude.*

The table of top neurons tells the story numerically:

| Factor | Top prismatic neurons (prismatic−control gap) |
|--------|-----------------------------------------------|
| ×1     | 2–3 neurons, gaps of +0.6 to +1.5             |
| ×5     | 3 neurons, gaps of +0.9 to +1.1               |
| ×10    | 4 neurons, gaps of +1.4 to +3.2               |
| ×25    | 5 neurons, gaps of +1.2 to +1.7               |
| ×50    | 5 neurons, gaps of +1.5 to +2.4               |
| ×100   | 4–5 neurons, gaps of +1.9 to +3.3             |

At ×10 and above, the winning neurons are near-zero on both near-prismatic and control boards — not just lower, but essentially silent. The model has learned a precise binary detector: the specific trait+emblem co-occurrence triggers it, neither trait alone nor emblem alone does. This is a qualitatively different circuit from the baseline.

| Board type | Best neuron activation (×1) | Best neuron activation (×10) |
|------------|-----------------------------|-------------------------------|
| Prismatic (8 trait + 2 matching emb) | ~1.45 | ~3.18 |
| Near-prismatic (8 trait, no emb)     | ~0.49 | ~1.06 |
| Control + emblems                    | ~0.00 | ~0.41 |

Whether this constitutes genuine understanding of the prismatic mechanic, or just stronger memorization of the champion+emblem co-occurrence patterns that happen to appear together in winning training boards, is hard to say. My suspicion is mostly the latter — the same shortcut logic at higher fidelity. But the circuit structure is meaningfully cleaner either way.

---

## Which Neurons? It Doesn't Matter

The interesting thing about the heatmap is not which specific neurons light up — it's that they're completely different at every oversampling level.

At ×5, the top neurons are N1, N17, N27. At ×10, N18, N16, N29. At ×25, N17, N31, N27, N26. At ×50, N19, N23, N26, N25. At ×100, N18, N0, N11, N28. No single neuron consistently develops prismatic specificity across all runs. The model reaches a different local minimum each time and assigns a different set of neurons to fill the same functional role.

This is a general point about interpreting small transformer MLPs: the neuron index is arbitrary. Any of the 32 neurons could become a "prismatic detector" given enough gradient signal — which ones do is determined by random initialization, not by the feature itself. The reproducible finding is the **circuit structure** (dedicated neurons, binary activation profile, silent on controls), not the **neuron identity**.

---

## Predictions Get Better, Up To a Point

![Predicted placement for prismatic, near-prismatic, and control boards across oversample factors, with test MAE on secondary axis](/assets/images/oversample_predictions.png)
*Predicted placement (lower = better) for the three board categories across oversample factors. Test MAE (right axis, also lower = better) is non-monotone: best at ×50.*

The model's predictions for prismatic boards improve immediately — from ~1.53 at ×1 to ~1.31 at ×5 and ×10. The gap between prismatic and control board predictions also widens substantially, from ~2.1 at baseline to ~3.6 at ×50.

Test MAE is non-monotone. ×25 actually performs slightly worse than baseline (1.4792 vs 1.4679) — the distributional shift is hurting the common cases. The best overall MAE is at ×50 (1.4588). Beyond that, the model starts overfitting to the overrepresented prismatic boards at the cost of everything else.

---

## Attention Still Doesn't Participate

Across all six models, the fraction of attention directed at emblem tokens on prismatic boards (0.043–0.066) was indistinguishable from that on control boards with emblems (0.035–0.070). Attention entropy showed no consistent decrease on prismatic boards at any factor.

Even at ×100, with prismatic boards comprising over a third of training data, the attention head doesn't differentiate them. The prismatic circuit lives entirely in the embedding → sum pool → MLP pathway. This makes sense given the architecture: a single attention head with collapsed Q/K matrices (as described in the first post) encodes champion strength, not compositional relationships. Detecting "this champion + this matching emblem = trait bonus" would require the head to track cross-token interactions of a kind it has no capacity for. The problem is architectural, not a matter of signal.

---

## Conclusions

The main things I took from this:

- **The circuit sharpens with signal.** At baseline, prismatic detection is weak and noisy — partial activation on champion density, not specifically the trait+emblem combination. With heavier oversampling, 4–6 dedicated neurons emerge with clean binary profiles: silent on near-prismatic and controls, strongly activated on prismatic boards.
- **The feature splits.** What is handled by 1–2 neurons at ×1 is distributed across 4–6 neurons at ×10+. Whether this is superposition being resolved or just gradient signal recruiting more capacity, I'm not sure. A larger model (d_model=16 or 32) would be the right setting to investigate.
- **Neuron identity is not the finding.** The interesting thing to describe is the activation profile and the circuit structure — not which specific neuron index carries the feature, since that's arbitrary across training runs.
- **Attention is a dead end for this feature.** Not for lack of signal — even ×100 oversampling doesn't move it. The head needs a structural upgrade to participate in compositional detection.

## Open Questions

- The feature splits from ~1–2 neurons at ×1 to 4–6 at higher factors. Is this superposition resolving as the feature becomes important enough to deserve its own dedicated directions? A larger embedding space would let us probe this more directly.
- Can we identify which training samples drive the largest gradient updates to the prismatic circuit, and use that to oversample more efficiently rather than just repeating all prismatic boards equally?
- If we use 4 attention heads instead of 1, does any head learn to attend to emblems differently on trait-heavy boards? The first post noted that grokking in a related experiment required exactly 4 heads — maybe that's the minimum for compositional detection to emerge in attention.

*The code for this project is on [GitHub](https://github.com/alexayvazyan/projects).*
