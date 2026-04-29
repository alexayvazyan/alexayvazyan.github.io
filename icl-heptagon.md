---
layout: post
title: "The ICL Heptagon — Research Thread"
permalink: /icl-heptagon.html
---

# The ICL Heptagon — Research Thread

*A running synthesis of the work surrounding a single observation: feeding a base language model 21 in-context demonstrations of a modular-7 successor rule over 7 semantically unrelated nouns can induce a 7-fold cyclic arrangement of word centroids in the residual stream. The thread now spans four phases of work — discovery, cross-model sweep, causal patching, and the deduction wall — and the picture that's emerged is sharper than I expected. The ring is real, it's causally load-bearing at a specific layer in a specific subspace, and the way models fail to use it for compositional deduction is informative about the difference between "task-solving" and "task-representing".*

---

## What this page is

This is a thread page, not a single experiment. The individual posts and writeups are linked throughout. The point of this page is to show how the four phases of work fit together, and what the synthesis is — because the headline result of the thread ("the ring is the causal locus where word identity is first written") is sharper than any single phase implies on its own.

Each phase below answers a question the previous one raised.

---

## Phase 1 — The original observation (Gemma 2 2B)

The task: take 7 single-token English nouns with no semantic ordering — `apple, mountain, violin, dragon, forest, window, river`. Define a modular-7 successor rule over them in some hidden order. Build a prompt with 21 in-context demonstrations of the form `After X: Y` (where Y is the successor of X), then query `After {word}:` and look at the residual stream at the last token (the prediction site).

On Gemma 2 2B base, at layers 16-20, the 7 word centroids arrange themselves into a 7-fold cyclic structure in the PC1-PC2 plane — the same cycle the task specifies. The nouns have no natural ordering; the ring has to be **induced** by the in-context demos, not read off pretrained relationships.

This was the seed that made the rest of the thread worth pursuing. The natural next question: is this Gemma's idiosyncrasy, or a general property of ICL?

---

## Phase 2 — Cross-model sweep + phase structure

[Full post: ICL can induce a cyclic manifold — but only in some models](/2026/04/22/icl-induced-cyclic-manifold-cross-model.html)

Same task, ran across 13 model variants spanning four families (Gemma, Llama, Pythia, Qwen, GPT-2) and four parameter scales (124M → 9B). Headline table:

| Model | Cyclic acc | Peak cyclic order | Peak ring radius | Ring classification |
|---|---|---|---|---|
| GPT-2 small (124M) | 87% | 3/7 | — | No ring |
| Pythia 1.4B | 54% | 4/7 | — | No ring (subthreshold task accuracy) |
| Gemma 2 2B base | 94% | 7/7 (L16-20) | 3.75 | Cyclic ring, tight |
| Gemma 2 2B IT | 93% | 7/7 (L20) | 3.96 | Cyclic ring, tight |
| Pythia 2.8B | 98% | 7/7 (L23-30) | 1.85 | Ring in PC1-2, identity in PC4+ (orthogonal subspaces) |
| Llama 3.2 3B | 100% | 7/7 (L15-27) | 5.75 | Cyclic ring, tight |
| Llama 3.2 3B Instruct | 100% | 7/7 (L23) | 3.71 | Cyclic ring, tight |
| Qwen 2.5 3B | 100% | 5/7 | 3.54 | Ring but cycle scrambled |
| Gemma 4 E2B / E4B / 31B | 14-21% | ≤5/7 | ≤2.5 | At chance — cannot solve task |
| Gemma 2 9B | 100% | 7/7 (L22-28) | 4.60 | Cyclic ring, tight |

![PC1-PC2 projection of the cyclic-condition prediction-site residual at each model's peak-order layer. Six panels show the ring forming cleanly in Gemma 2 2B/9B, Llama 3.2 3B, Pythia 2.8B; Qwen 2.5 3B has a ring but with cycle order scrambled; GPT-2 and Pythia 1.4B show no ring.](/assets/images/icl_manifold_7panel.png)
*Per-model PC1-PC2 at peak-order layer. The ring forms cleanly in 4 of 7 base models. Qwen 2.5 3B has a ring with the cycle scrambled — solves the task at 100% via a different mechanism. GPT-2 small still solves at 87% with no ring.*

Three things this gave us:

**(a) The ring generalizes — but it isn't necessary for solving the task.** Qwen 2.5 3B is the kill case: 100% task accuracy, 5/7 order, ring radius 3.54 — the centroids sit on a ring but they're arranged off-cycle. GPT-2 small at 87% accuracy has no ring at all. Solving the successor task is not sufficient to force the ring geometry.

**(b) The Gemma 4 family is broken.** All three Gemma 4 dense models (E2B, E4B, 31B) fail this task at chance. Same lab, newer generation. Some interaction of training data, architecture, or attention pattern in Gemma 4 destroys the inductive capability that Gemma 2 2B/9B had. This is the strongest single-model anomaly in the sweep.

**(c) Phase structure.** Every task-solving model shows a three-phase trajectory across layers at the prediction site:
- **Phase A (early):** pretrained word-identity clusters dominate; silhouette in top-3 PCs near +0.9.
- **Phase B (mid, "compression" not "dissolution"):** identity subspace is preserved but variance-compressed relative to the rest of the residual. Top-3 PC silhouette crashes to 0; LDA silhouette stays high. The residual is being rewritten to carry task context; identity stops being a dominant variance axis but stays linearly recoverable.
- **Phase C (ring-forming):** the cycle direction enters PC1-PC2 and brings enough identity structure with it for the top-3 PC silhouette to rebound. Task-failing models (Gemma 4, Pythia 1.4B) never reach this rebound.

A subspace-geometry distinction worth flagging: Gemma 2 2B/9B and Llama 3B all keep the ring and identity in the *same* plane — what you see in PC1-PC2 is the full picture. Pythia 2.8B keeps the ring in PC1-PC2 and identity in PC4+, *orthogonal* subspaces. The ring-radius metric is biased against this case (1.85 for Pythia vs 3.75-5.75 for the others) even though the cycle is perfectly ordered. The original post treated ring formation as binary; the truer picture is that "does it form a ring" and "where does the identity live" are separate axes, and Pythia 2.8B is the cleanest test case for that distinction.

---

## Phase 3 — The deduction wall

[Full writeup: `manifold_days/DEDUCTION_AND_PATCHING.md`]

The natural next question after Phase 2: if the model has formed this clean cyclic ring, does it actually *use* it for compositional reasoning? Specifically — can a model that solves `direct` (next-successor demos) at 95-100% also solve `mixed` (demos use random ±2, ±3 relations, requiring the model to reconstruct the cycle and then apply +1)?

The answer is a clean no.

| Mode | Demos show | Best top-1 across all models | Chance |
|---|---|---|---|
| `direct` | +1 | 93-100% | 14.3% |
| `backward` | -1 | ≤21% | 14.3% |
| `k2` | +2 only | ≤8% | 14.3% |
| `k3` | +3 only | ≤8% | 14.3% |
| `mixed` | random {±2, ±3} | ≤8% | 14.3% |
| `twoahead` | demos +1, query +2 | ≤8% | 14.3% |

Most cells are *below* chance — models are systematically wrong, not random. They pattern-match the demo's stated answer (e.g. on `k2` they emit +2, not +1). The instruction-prefix sweep that followed (5 prefaces × 4 models) gave modest lifts on instruct models but never crossed 36%. The single preface that hurt the most was `gauss_euler` ("think like Gauss / Euler about modular arithmetic") — abstract framings consistently degrade performance. This recurs in Phase 4 with `compose_hint`.

Read straight up: the models that ace `direct` are *copying*, not doing modular composition. The ring is consistent with the task they're solving (which is local-pattern matching of `After X: Y` demos), not with a learned cyclic structure they can compose over.

---

## Phase 4 — Causal patching localizes the ring

[Full writeup: `manifold_days/DEDUCTION_AND_PATCHING.md`]

If the ring at L16-20 is real, is it doing causal work, or is it a downstream byproduct? The decisive experiment: cross-task activation patching.

**Setup.** Capture the last-token residual from `direct(qw)` at layer L. Transplant into `mixed(qw)` at the same layer. Three conditions: baseline (no patch), same-word patch, wrong-word donor patch (donor ≠ recipient). 140 trials per layer on Gemma 2 2B.

```
L  | base  | same  | wrong-recipient | wrong-donor (donor's answer)
 4 | 0.107 | 0.100 |     0.114       |      0.121          no effect
10 | 0.107 | 0.107 |     0.114       |      0.136          no effect
14 | 0.107 | 0.093 |     0.121       |      0.129          no effect
16 | 0.107 | 0.643 |     0.079       |      0.486          ONSET
18 | 0.107 | 0.757 |     0.043       |      0.593
20 | 0.107 | 0.843 |     0.021       |      0.800          peak
22 | 0.107 | 0.886 |     0.021       |      0.857
25 | 0.107 | 0.907 |     0.029       |      0.879          near logits
```

**The clean step at L14 → L16.** Same-word patch jumps from 9.3% to 64.3% in a single layer — the same band the ring forms in. The donor-leak control is what makes the result decisive: at L20, transplanting a *donor's* residual into a different recipient makes the model emit *donor + 1* 80% of the time, while the recipient's correct answer drops to 2.1% (well below chance). The patch carries the donor's specific identity, not generic task signal.

**Triviality controls.** Gaussian noise patch (matched norm) at L20: 22.1%. Mean-of-other-words patch at L20: 7.1%. Both well below same-word's 84.3% — the patch effect is content-specific, not vector-overwriting.

**The PC-k decomposition is the cleanest finding in the thread.** At the *onset* layer L=16:

| k | var explained | same top-k | donor leak |
|---|---|---|---|
| 1 | 0.251 | 0.157 | 0.200 |
| 2 | 0.467 | 0.443 | **0.471** ← peak |
| 3 | 0.646 | 0.300 | 0.286 ← drops adding PC3 |
| 5 | 0.928 | 0.407 | 0.386 |
| 10 | 1.000 | 0.414 | 0.393 |

Donor-leak peaks at k=2 and is *higher* than at k=full. PC1-PC2 alone carry ~97% of the donor-leak signal at the onset layer (0.471 vs full 0.486). PC3+ doesn't just fail to add useful signal — it *contaminates* the patch, because PC3+ at L=16 carries prompt-context features that don't transfer between donor and recipient prompts.

**Logit lens at the patch layer disambiguates "patching does structural work" from "patching is just early logit injection":**

| Layer | logit-lens | same-full patch | gap |
|---|---|---|---|
| 14 | 0.107 | 0.093 | -0.01 |
| 16 | 0.300 | 0.643 | **+0.34** |
| 18 | 0.564 | 0.757 | +0.19 |
| 20 | 0.836 | 0.843 | +0.01 |

L=16 is the causally informative layer: 30% logit-lens vs 64% patched is +34 points of genuine structural work. By L=20 the residual ≈ logits, so patching becomes equivalent to logit injection and the gap collapses. **The ring at L=16 is the layer where word-identity is first written into the residual; later layers refine and diffuse this representation into higher-dimensional subspaces.**

The deduction failure from Phase 3 now has a sharper read. It's not that models can't *use* the ring — patch the right ring position into a `mixed` prompt and they predict the right answer 84% of the time. The failure is upstream: from `mixed` demos, the model never *forms* the right ring position in the first place. Hand it the ring, and the rest of the network knows what to do.

---

## Phase 5 — The closed +2 chain ceiling

[Memo: `project_closed_chain_deduction.md`]

The cleanest possible deductive cyclic ICL: 7 demos of "Two after X: Y" only, with `gcd(2, 7) = 1` so the chain is fully closed. Query asks `+1` — correct answer requires composing `+2` four times mod 7. 12 models × 4 prefaces × 2 query styles × 35 trials.

Best top-1 per model (chance = 0.143):
- Gemma 2 2B base & IT, Llama 3.2 3B base: **0.286** (cycle_hint)
- Qwen 2.5 3B base & IT, Gemma 2 9B IT: 0.257
- Gemma 4 E2B / E4B / 31B IT: ≤ 0.114, often below chance
- GPT-2: 0.200, Pythia 2.8B: 0.143

**Three findings worth carrying forward:**

1. **Scale anti-correlates within Gemma family.** 31B < 9B < 2B on this task. The failure is representational, not capacity-bound. Strengthens the patching story: hand the right ring → model solves it; absent the ring → bigger models don't help.
2. **Abstract framings hurt.** `compose_hint` ("four +2s = one +1, 8 mod 7 = 1") *reduces* accuracy. Same pattern as `gauss_euler` from Phase 3. Concrete framings (`cycle_hint`, `explicit_task`) win.
3. **The ceiling is ~28% even on the cleanest deductive variant.** This is the number to beat for any future architectural intervention. Hybrid demos (one +1 fact mixed in to anchor the ring) are the next obvious experiment.

---

## Synthesis

The clean form of the claim that this thread now supports:

> The cyclic ring at the prediction site is the causal locus where word-identity is first written into the residual stream of task-solving base LMs. PC1-PC2 alone carry ~97% of the patch-transfer signal at the onset layer. Solving the ICL successor task is not sufficient to produce this geometry — Qwen 2.5 3B and GPT-2 small both solve via different internal representations. Forming the ring is what enables compositional use; failing to form the right ring under deductive variants is the upstream cause of the systematic compositional failure observed across all tested models.

A few things this synthesis specifically rules out:

- **"The ring is decorative."** Killed by the donor-leak result and the PC1-PC2 onset-layer specificity.
- **"Patching at L=16 is just early logit injection."** Killed by logit-lens (30% lens vs 64% patched).
- **"Bigger models will solve compositional ICL deduction."** Killed by within-Gemma-family scale-anti-correlation on the closed +2 chain.
- **"Abstract framings help reasoning."** Killed by `gauss_euler` and `compose_hint` both hurting.

A few things this synthesis specifically *does not* rule out:

- That different model families form the ring through different mechanisms (induction-head copy vs something else). Gemma 4's uniform failure suggests architecture-or-training-data matters; the sweep doesn't yet say which.
- That the ring is the only causally-relevant geometry. Qwen 2.5 3B solves with a different geometry, which is its own thread to investigate.
- That the deduction wall is fundamental rather than a property of in-context-only computation. CoT-generation accuracy with explicit_task preface is the obvious follow-up.

---

## What I'm working on next

In rough order:

1. **Replicate Phase 4 (patching + PC-k decomposition) on Llama 3.2 3B, Pythia 2.8B, Gemma 2 9B.** The §3 headline of the joint paper depends on whether onset-layer-PC1-2-causality holds across models. Pythia 2.8B is the sharpest test because its ring and identity sit in *orthogonal* subspaces — the cleanest possible test of the PC1-PC2-causality claim.
2. **Reverse patching: `mixed → direct`.** Does `mixed` produce *any* coherent ring position at L=16, or just noise? Tells us whether the deduction failure is "no ring position formed" or "wrong ring position formed."
3. **Hybrid demos.** Closed +2 chain plus a single +1 fact. Does one anchor unlock the ring?
4. **Gemma 4 attention / induction-head probe.** Cleanest differential signal in the thread. Why does this entire model family fail?
5. **CoT-generation scoring.** First-token scoring forecloses CoT. Scoring the parsed final answer of a 100-token generation distinguishes "latent but unreachable" from "absent."
6. **SAE feature decoding on the ring layers.** Gemma Scope covers all Gemma 2 layers. Once the ring is localized, decoding what the 7 axes correspond to is the natural mech-interp endpoint of this thread.

---

## Reading the thread end-to-end

In the order I'd recommend if you want the full picture:

1. **The original post**, [ICL can induce a cyclic manifold — but only in some models](/2026/04/22/icl-induced-cyclic-manifold-cross-model.html), for the foundational result and full methodology.
2. **`manifold_days/DEDUCTION_AND_PATCHING.md`** for the Phase 3 deduction-failure tables and the Phase 4 patching + PC-k decomposition writeup.
3. **`project_closed_chain_deduction.md`** (in research notes) for the Phase 5 closed +2 chain ceiling result.

The earlier post's two known imprecisions are corrected here: instruct tuning does *not* break the ring on Gemma 2 2B (the original 37% claim came from full-vocab argmax — under restricted top-1, IT solves at 93%); and the binary "ring / no ring" framing under-reads the Pythia 2.8B case, which is better described as "ring in PC1-2, identity in orthogonal subspaces."
