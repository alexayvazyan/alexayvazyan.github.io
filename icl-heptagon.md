---
layout: post
title: "The ICL Heptagon — Cross-Model Sweep, Causal Patching, and the Deduction Wall"
permalink: /icl-heptagon.html
---

# The ICL Heptagon — Cross-Model Sweep, Causal Patching, and the Deduction Wall

*Feed a base language model 21 in-context demos of a modular-7 successor rule over 7 semantically unrelated nouns, look at the residual stream at the last token, and in 4 of 7 base models you find a 7-fold cyclic ring of word centroids in the PC1-PC2 plane — the same cycle the task specifies. Activation patching localizes this ring as the **causal locus where word identity is first written into the residual**: at the onset layer of Gemma 2 2B (L=16), PC1-PC2 alone carry ~97% of the patch-transfer signal. Despite this clean structure, every tested model fails at compositional ICL deduction: even the cleanest deductive variant caps at ~28% accuracy. Solving the successor task and forming the ring are dissociable from each other and from compositional generalization.*

---

## What this page is

A consolidated writeup of the heptagon thread, covering the original cross-model sweep, the phase-by-layer geometry analysis, the cross-task activation patching that localizes the ring's causal contribution, the PC-k decomposition that further localizes it to PC1-PC2 at a specific onset layer, and the closed +2 chain sweep that establishes the deduction-failure ceiling. Each phase below answers a question the previous one raised.

---

## Phase 1 — Task setup and the original observation

### The task

Seven single-token English nouns with no semantic ordering: `apple, mountain, violin, dragon, forest, window, river`. Each tokenizes to one token (with a leading space) in every model tested. Define a modular-7 successor rule over them in a hidden order — treat word `i` as the predecessor of word `(i + 1) mod 7`.

The prompt has three parts:

1. A header line: `Cycle: apple -> mountain -> violin -> dragon -> forest -> window -> river -> apple (repeats).`
2. 21 demonstrations of the form `After X: Y\n`, where `Y = words[(idx(X) + 1) mod 7]`, shuffled.
3. A query: `After {word}:`

A complete `direct`-mode prompt looks like this (header on, query word `dragon`, gold answer `forest`):

```
Cycle: apple -> mountain -> violin -> dragon -> forest -> window -> river -> apple (repeats).

After window: river
After mountain: violin
After dragon: forest
After river: apple
After violin: dragon
After apple: mountain
After forest: window
After mountain: violin
After dragon: forest
After window: river
After forest: window
After apple: mountain
After river: apple
After violin: dragon
After dragon: forest
After window: river
After mountain: violin
After violin: dragon
After apple: mountain
After river: apple
After forest: window
After dragon:
```

Notice the literal pair `After dragon: forest` appears in the demos. In the `direct` mode the model can answer correctly by retrieving any line whose left-hand side matches the query — which is the pure-copy baseline that all later modes are designed to remove.

The header is optional (controlled by `--no_header`); we ran with and without and accuracy was unchanged. The cross-model sweep used `n_reps=3` (each of the 7 words appears 3 times on the left), giving 21 demo lines.

Three conditions per prompt:

- **baseline** — plain English sentences mentioning the word, no ICL demos. Controls for the model's pretrained word geometry.
- **cyclic** — 21 successor-rule demos as above. The condition whose geometry we care about.
- **shuffled** — same template as cyclic but `Y` is a random word, breaking the cycle. Controls for ICL density and prompt length.

For each prompt, residuals are taken at the final token (`lpos` — the prediction site) across every layer.

> **Why the prediction site?** An earlier version of this experiment looked at the *query-word token position* (`qpos`) and got a null result. The cycle structure lives at the position where the model is actually computing the next token, not at the token spelling the question.

### The metrics

Three per-layer quantities, computed on per-condition PCA(3) projections of 210 prompts:

**Angular order score (out of 7).** Take the 7 per-word centroids in PC1-PC2. Compute their angular order around the centroid. Check whether that order matches the true cycle (in either direction, at any starting offset). Report the *best* match across all 7 rotations × 2 directions. **7/7 means the 7 words lie in perfect cycle order around a ring.**

**Ring radius.** For each point, compute its distance from the PC1-PC2 centroid. Report `mean(radius) / std(radius)`. High values mean a tight ring; low values mean the "ring" is a smear with varying radii.

**k-fold FFT power.** Histogram angular positions into 120 bins, take FFT, report the peak frequency and its power. Noisy with 7 groups of 30 points; treated as supporting, not primary.

**Task accuracy.** Restricted top-1 over the 7 word tokens. Shuffled accuracy should sit at chance (1/7 ≈ 14%).

Cycle order is the load-bearing signal. Ring radius is the tiebreaker: a model can achieve 7/7 order while its points smear radially, and that smeared 7-cycle is a weaker result than a tight ring.

### The original observation

On Gemma 2 2B base, at layers 16-20, the 7 word centroids arrange themselves into a 7-fold cyclic structure in the PC1-PC2 plane — the same cycle the task specifies. The nouns have no natural ordering; the ring is **induced** by the in-context demos, not read off pretrained relationships. The natural next question: is this Gemma's idiosyncrasy, or a general property of ICL?

---

## Phase 2 — Cross-model sweep

Same task, ran across 13 model variants spanning four families and four parameter scales (124M → 9B). All base models below; instruct variants discussed at the end of the section.

| Model | Params | Layers | Family |
|---|---|---|---|
| GPT-2 small | 124M | 12 | OpenAI GPT-2 |
| Pythia 1.4B | 1.4B | 24 | EleutherAI GPT-NeoX |
| Gemma 2 2B | 2B | 26 | Google Gemma |
| Pythia 2.8B | 2.8B | 32 | EleutherAI GPT-NeoX |
| Llama 3.2 3B | 3B | 28 | Meta Llama |
| Qwen 2.5 3B | 3B | 36 | Alibaba Qwen |
| Gemma 2 9B | 9B | 42 | Google Gemma |

### Headline table

| Model | Cyclic acc | Shuf acc | Peak cyclic order | Peak ring radius | Peak band (% depth) | Ring forms? |
|---|---|---|---|---|---|---|
| GPT-2 (124M) | 87% | 17% | 3/7 | — | — | **No** |
| Pythia 1.4B | 54% | 14% | 4/7 | — | — | **No** (subthreshold accuracy) |
| Gemma 2 2B | 94% | 18% | **7/7** | 3.75 | L16-20 (62-77%) | Yes, tight |
| Pythia 2.8B | 98% | 17% | **7/7** | 1.85 | L23-30 (72-94%) | Yes — angular only |
| Llama 3.2 3B | 100% | 18% | **7/7** | 5.75 | L15-27 (54-96%) | Yes, tight |
| Qwen 2.5 3B | 100% | 18% | 5/7 | 3.54 | — | **Ring, but cycle scrambled** |
| Gemma 2 9B | 100% | 18% | **7/7** | 4.60 | L22-28 (52-67%) | Yes, tight |

### The ring, visualized

![Per-model PC1-PC2 grid at peak-order layer. Six panels: Gemma 2 2B/9B, Llama 3.2 3B and Pythia 2.8B form clean cyclic rings (green titles, 7/7 angular order). Qwen 2.5 3B has a ring but with cycle order scrambled (yellow title). GPT-2 and Pythia 1.4B have no ring (red titles).]({{ site.baseurl }}/assets/images/icl_manifold_7panel.png)
*PC1-PC2 projection of the cyclic-condition prediction-site residual at each model's peak-order layer. Small dots are individual prompts; large circles are per-word centroids; the gray polygon connects centroids in PC1-PC2 angular order.*

### The layer band

![Cyclic angular order vs normalized depth for the seven base models. Ring-forming models all hit 7/7 in a mid-to-late band (~50-80% depth); below that the representation hasn't organized, above that it weakens as the model commits to next-token logits.]({{ site.baseurl }}/assets/images/icl_manifold_depth_sweep.png)
*Cyclic-condition angular order vs normalized depth. The ring-forming models hit 7/7 in a mid-to-late band (~50-80% depth) and weaken in the last decile as the model commits to logits.*

### Ring radius separation

![Ring radius (cyclic) vs shuffled control across models. Llama 3.2 3B and Gemma 2 9B sit 2-3x above shuffled. Pythia 2.8B sits at 1.85 vs ~1.8 — angular order is 7/7 but the points scatter radially in PC1-PC2.]({{ site.baseurl }}/assets/images/icl_manifold_ring_radius.png)
*Even among 7/7 models the ring tightness varies widely. Llama 3.2 3B and Gemma 2 9B sit cleanly above shuffled; Pythia 2.8B is angular-only — its identity structure lives in PC4+ rather than PC1-PC2.*

### The Qwen kill-case

The single most informative result in the sweep is **Qwen 2.5 3B**: 100% top-1 on the cyclic task, 18% on shuffled (so it genuinely learns the rule), peak angular order 5/7, ring radius 3.54 vs shuffled 2.22 — its centroids sit on a ring of roughly the same geometric quality as Gemma 2 2B's, but they aren't arranged in cycle order around it.

Compare to Llama 3.2 3B: same parameter count, same 100% accuracy, similar ring quality — but 7/7 cyclic order. Both models compress the 7 words onto a circle. Only one of them does so in the order dictated by the task.

**Qwen kills the hypothesis that "the model encodes the successor rule as a cyclic ring" is a necessary consequence of solving the task.** Qwen is solving at 100% with a circular geometry that isn't aligned to the cycle. The cycle sits on a different subspace — or a different mechanism — that PC1-PC2 at this layer doesn't expose. Candidate mechanisms: a higher-dimensional mapping, a direct induction-head copy from the demonstrations, or a non-geometric indexing scheme. The same point applies (more weakly) to GPT-2 small at 87% with no ring at all.

### Within-family scaling

Pythia 1.4B → Pythia 2.8B gives a clean comparison: 54% accuracy and no ring at 1.4B → 98% accuracy and 7/7 order at 2.8B. Scale crosses both the task-solving threshold and the ring-formation threshold somewhere between 1.4B and 2.8B *within this family*. But at comparable scale, Llama 3.2 3B and Gemma 2 2B form geometrically tighter rings (radius 3.75-5.75) than Pythia 2.8B (1.85). Scale is a necessary condition within a family, not the only factor determining ring quality across families.

### Instruct tuning

I also ran four instruct variants (Gemma 2 2B-IT, Gemma 2 9B-IT, Llama 3.2 3B-Instruct, Qwen 2.5 3B-IT). Instruct tuning is essentially a no-op for this probe under restricted top-1 scoring: where base forms a ring, IT preserves it with small radius shifts (Gemma 2 2B 3.75 → 3.96; Llama 3.2 3B 5.75 → 3.71). Where base fails (Gemma 4), IT also fails. The cyclic-ICL capability is baked in at pretraining; this probe is blind to alignment.

> A note: an earlier version of this writeup claimed IT *broke* the Gemma 2 2B ring. That was a methodology artifact — full-vocab argmax was scoring against punctuation and chat tokens. Under restricted top-1 across the 7 word tokens, IT is at 93%, indistinguishable from base.

### The Gemma 4 anomaly

All three Gemma 4 dense models tested (E2B, E4B, 31B) fail this task at chance (14-21% top-1). Same lab as Gemma 2, newer generation. Something in Gemma 4's pretraining (data mix, attention pattern, or architecture) destroys the inductive capability that Gemma 2 2B/9B both had. This is the strongest single-model anomaly in the sweep and the cleanest mechanistic target.

### Phase structure across layers

Every task-solving model shows a three-phase trajectory at the prediction site:

- **Phase A (early layers).** Pretrained word-identity clusters dominate. Top-3 PC silhouette near +0.9 in ≥2B models. Centroids form 7 distinct positions whose angular arrangement is arbitrary w.r.t. the synthetic cycle (since pretraining has no notion of it).
- **Phase B (mid layers — *compression*, not dissolution).** Top-3 PC silhouette crashes to ~0, but clusters remain linearly separable (LDA silhouette stays ≥+0.78). The identity subspace is preserved but variance-compressed relative to the rest of the residual. Between-class variance drops ~50× from early to mid. The residual is being rewritten to carry task context; identity stops being a dominant variance axis but stays recoverable.
- **Phase C (ring-forming layers).** The cycle direction enters PC1-PC2 and brings enough identity structure with it for the top-3 PC silhouette to rebound. Task-failing models (Gemma 4, Pythia 1.4B) never reach this rebound.

A subspace-geometry distinction worth flagging: Gemma 2 2B/9B and Llama 3B keep ring and identity in the *same* plane — what you see in PC1-PC2 is the full picture. Pythia 2.8B keeps the ring in PC1-PC2 and identity in PC4+, *orthogonal* subspaces. The ring-radius metric is biased against this case (1.85 vs 3.75-5.75 for the others) even though the cycle is perfectly ordered. "Does it form a ring" and "where does identity live" are separate axes.

---

## Phase 3 — The deduction wall

If the model has formed this clean cyclic ring, does it actually *use* it for compositional reasoning? Specifically — can a model that solves `direct` (next-successor demos) at 95-100% also solve a variant where the literal pair `(qw, qw+1)` never appears in any demo, so the answer must be deduced rather than copied?

Clean no.

### The six modes

Six modes, all with the same gold answer (the +1 successor of the query word). They differ only in what relation the demos exhibit:

| Mode | Demo template | Query | Cognitive demand |
|---|---|---|---|
| `direct` | `After X: Y` (Y = X+1) | `After {qw}:` | Pure copy — the line `After {qw}: {qw+1}` literally appears |
| `backward` | `Before X: Y` (Y = X−1) | `After {qw}:` | Single inversion |
| `k2` | `Two after X: Y` (Y = X+2) | `After {qw}:` | Compose +2 → +1 (mod 7) |
| `k3` | `Three after X: Y` (Y = X+3) | `After {qw}:` | Compose +3 → +1 |
| `mixed` | random offset per fact from {±2, ±3} | `After {qw}:` | Reconstruct the full cycle, then +1 |
| `twoahead` | `After X: Y` (Y = X+1) | `Two after {qw}:` | Compose +1 with itself |

Concretely, for query word `dragon` (gold `forest`, since dragon → forest in the cycle), the demos look like:

```
direct           backward         k2                       k3
─────────────    ─────────────    ─────────────────────    ──────────────────────
After violin:    Before violin:   Two after violin:        Three after violin:
  dragon           apple            forest                   window
After dragon:    Before dragon:   Two after dragon:        Three after dragon:
  forest           violin           window                   river
After mountain:  Before mountain: Two after mountain:      Three after mountain:
  violin           apple            dragon                   forest
...              ...              ...                      ...
After dragon:    After dragon:    After dragon:            After dragon:
```

The point: in every mode except `direct`, the literal pair `After dragon: forest` is **never** in the demos. The query syntax (`After dragon:`) and the demo syntax (`Two after ...:` / `Before ...:` / etc.) don't match. The model can't answer by retrieving a matching line — it has to synthesize.

For `mixed`, each of the 21 demo lines independently picks a random offset from {+2, +3, −2, −3}, so a single prompt looks like:

```
Two before window: violin
Three after apple: dragon
Two after river: violin
Three before mountain: window
Two after dragon: window
...
After dragon:
```

To answer correctly the model has to (a) parse "K after/before X: Y" as "Y = X ± K", (b) collect enough constraints to reconstruct the 7-cycle, and (c) emit X+1 for the query word.

For `twoahead`, the demos show +1 ("After violin: dragon") but the query asks +2 ("Two after dragon:") — the model has to compose the +1 it has just absorbed with itself.

### Results

Top-1 accuracy across 7 models, 140 queries per cell. Chance = 0.143.

```
                              direct    k2     k3   back   mixed  twoah
gemma-2-2b                     0.936  0.007  0.036  0.064  0.121  0.021
gemma-2-9b                     1.000  0.000  0.000  0.029  0.036  0.000
Llama-3.2-3B                   1.000  0.000  0.000  0.079  0.079  0.000
Llama-3.2-3B-Instruct          1.000  0.000  0.000  0.214  0.064  0.000
Qwen2.5-3B                     1.000  0.000  0.000  0.050  0.064  0.000
pythia-2.8b                    1.000  0.064  0.043  0.007  0.143  0.000
gemma-4-E4B (broken control)   0.129  0.136  0.143  0.136  0.129  0.107
```

Most deduction cells are *below* chance — models are systematically wrong, not random. They pattern-match the demo's stated answer: on `k2` they emit +2 (`Two after dragon: window` → predict `window`), not +1 (`forest`). On `backward` they pick up the inverted relation only weakly, and only with instruct tuning (Llama-Instruct at 21.4%). Composition modes (`k2`, `k3`, `twoahead`) sit at floor for every model.

The Gemma 4 E4B row is the calibration control: at 13-14% across every mode (including `direct`), it confirms that 14.3% really is the floor when the model can't do the task at all.

### Instruction prefaces

Could prompting fix this? Tested 5 prefaces × 4 models. The prefaces, verbatim:

- **`cycle_hint`** — "These 7 words (apple, mountain, violin, dragon, forest, window, river) form a hidden cycle of length 7. Use the facts below to deduce the ordering, then answer the final query."
- **`gauss_euler`** — "Think like Gauss or Euler. The 7 words form a cyclic group of order 7. Reason about modular arithmetic on the cycle to deduce the answer."
- **`step_by_step`** — "Reason carefully step by step about the cyclic ordering of the 7 words before producing the answer. Compose offsets where needed."
- **`anti_copy`** — "Do NOT copy from the lines below. The facts use offsets like 'Two after' or 'Three before' — they do NOT directly state the +1 successor. Deduce the cycle, then apply +1 to the query word."
- **`explicit_task`** — "Task: from the facts below, deduce the cyclic ordering of the 7 words {apple, mountain, violin, dragon, forest, window, river}. Each fact states a relation of the form 'K after X: Y' meaning Y is K positions after X in the cycle. After deducing the cycle, answer the query."

Best per model on the deduction modes (chance = 0.143):

| Model | best preface | back | mixed | k2 | k3 | twoahead |
|---|---|---|---|---|---|---|
| Gemma 2 2B-it | `explicit_task` | 20.7% | 16.4% | 16.4% | 17.1% | 0.7% |
| Llama 3.2 3B-Instruct | `cycle_hint` | **36.4%** | 12.9% | 8.6% | 0.0% | 2.9% |
| Qwen 2.5 3B-Instruct | `explicit_task` | 34.3% | **24.3%** | 2.1% | 3.6% | 6.4% |
| Gemma 2 9B base | (any) | ≤5% | ≤14% | ≤1% | ≤2% | 0% |

Three readings:

- **Prefaces help only instruct models.** Gemma 2 9B base is inert to all of them. Whatever instruction-following circuitry is doing here, it's tuned in by IT, not pretrained.
- **Concrete framings beat abstract framings.** `cycle_hint` and `explicit_task` win. `gauss_euler` actively *hurts* — Llama-Instruct backward drops from 21.4% (no preface) to 12.1% with `gauss_euler`. Telling a model to "think like Gauss" is worse than saying nothing.
- **`step_by_step` is flat.** The probe scores the immediate next token; the model can't write out reasoning if you only read one token. CoT-aware scoring is one of the open follow-ups.

The single best deduction signal in the entire preface experiment is Qwen-Instruct + `explicit_task` on `mixed` = 24.3% — a real lift, but far from solving. Composition modes (k2/k3/twoahead) stay near floor under every preface.

Read straight up: the models that ace `direct` are *copying*. The ring is consistent with the local-pattern-matching task they're solving, not with a learned cyclic structure they can compose over.

---

## Phase 4 — Causal patching localizes the ring

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

**The clean step at L14 → L16.** Same-word patch jumps from 9.3% to 64.3% in a single layer — the same band the ring forms in. The donor-leak control is what makes the result decisive: at L20, transplanting a donor's residual into a different recipient makes the model emit *donor + 1* 80% of the time, while the recipient's correct answer drops to 2.1% (well below chance). The patch carries the donor's specific identity, not generic task signal.

**Triviality controls (at L=20).** Gaussian noise patch (matched norm): 22.1%. Mean-of-other-words patch: 7.1%. Both well below same-word's 84.3% — the patch effect is content-specific, not vector-overwriting.

### PC-k decomposition

The cleanest finding in the thread. At the *onset* layer L=16:

| k | var explained | same top-k | donor leak |
|---|---|---|---|
| 1 | 0.251 | 0.157 | 0.200 |
| 2 | 0.467 | 0.443 | **0.471** ← peak |
| 3 | 0.646 | 0.300 | 0.286 ← drops adding PC3 |
| 5 | 0.928 | 0.407 | 0.386 |
| 10 | 1.000 | 0.414 | 0.393 |

Donor-leak peaks at k=2 and is *higher* than at k=full. PC1-PC2 alone carry ~97% of the donor-leak signal at the onset layer (0.471 vs full 0.486). PC3+ doesn't just fail to add useful signal — it *contaminates* the patch, because PC3+ at L=16 carries prompt-context features that don't transfer between donor and recipient prompts.

### Logit-lens disambiguation

Whether patching is doing structural work or just early logit injection:

| Layer | logit-lens | same-full patch | gap |
|---|---|---|---|
| 14 | 0.107 | 0.093 | -0.01 |
| 16 | 0.300 | 0.643 | **+0.34** |
| 18 | 0.564 | 0.757 | +0.19 |
| 20 | 0.836 | 0.843 | +0.01 |

L=16 is the causally informative layer: 30% logit-lens vs 64% patched is +34 points of genuine structural work. By L=20 the residual ≈ logits, and patching collapses to logit injection. **The ring at L=16 is the layer where word-identity is first written into the residual; later layers refine and diffuse this representation into higher-dimensional subspaces.**

The deduction failure from Phase 3 now has a sharper read. Models *can* use the ring — patch the right ring position into a `mixed` prompt and they predict correctly 84% of the time. The failure is upstream: from `mixed` demos, the model never *forms* the right ring position in the first place. Hand it the ring, and the rest of the network knows what to do.

---

## Phase 5 — The closed +2 chain ceiling

The cleanest possible deductive cyclic ICL. Use a single relation (+2) with `gcd(2, 7) = 1`, so 7 facts close the chain and uniquely determine the cycle. Single relation, no offset-mixing, fully deterministic. Query asks +1 — to answer, the model must compose +2 four times (4·2 = 8 ≡ 1 mod 7).

A complete prompt looks like:

```
Two after apple: violin
Two after violin: forest
Two after forest: river
Two after river: mountain
Two after mountain: dragon
Two after dragon: window
Two after window: apple

After apple:                    → gold = mountain   (apple +1 = mountain)
```

Sweep: 12 models × 4 prefaces × 2 query styles ("After {qw}:" vs "One after {qw}:") × 7 query words × 5 demo-shuffle seeds = 280 trials per model.

The new preface tested at this phase is `compose_hint`, verbatim: *"The facts below all use the relation 'Two after X: Y' (a +2 step in a hidden 7-cycle). Since gcd(2,7)=1, these +2 steps fully determine the cycle. To answer the query, you may need to compose multiple +2 steps (four +2 steps equal one +1 step, since 8 mod 7 = 1)."* This is the strongest possible spoon-feeding short of giving the answer outright — it states the exact arithmetic the model needs to do.

Best top-1 per model (chance = 0.143):

- Gemma 2 2B base & IT, Llama 3.2 3B base: **0.286** (cycle_hint)
- Qwen 2.5 3B base & IT, Gemma 2 9B-IT: 0.257
- Gemma 4 E2B / E4B / 31B-IT: ≤ 0.114, often below chance
- GPT-2: 0.200, Pythia 2.8B: 0.143

Three findings worth carrying forward:

1. **Scale anti-correlates within Gemma family.** 31B < 9B < 2B on this task. The failure is representational, not capacity-bound. Strengthens the patching story: hand the right ring → the model solves it; absent the ring → bigger models don't help.
2. **Abstract framings hurt.** `compose_hint` ("four +2s = one +1, 8 mod 7 = 1") *reduces* accuracy. Same pattern as `gauss_euler` from Phase 3. Concrete framings (`cycle_hint`, `explicit_task`) win.
3. **The ceiling is ~28% even on the cleanest deductive variant.** This is the number to beat for any future architectural intervention. Hybrid demos (one +1 fact mixed in to anchor the ring) are the next obvious experiment.

---

## Synthesis

The clean form of the claim this thread now supports:

> The cyclic ring at the prediction site is the causal locus where word-identity is first written into the residual stream of task-solving base LMs. PC1-PC2 alone carry ~97% of the patch-transfer signal at the onset layer. Solving the ICL successor task is not sufficient to produce this geometry — Qwen 2.5 3B and GPT-2 small both solve via different internal representations. Forming the ring is what enables compositional use; failing to form the right ring under deductive variants is the upstream cause of the systematic compositional failure observed across all tested models.

What the synthesis specifically rules out:

- **"The ring is decorative."** Killed by the donor-leak result and the PC1-PC2 onset-layer specificity.
- **"Patching at L=16 is just early logit injection."** Killed by logit-lens (30% lens vs 64% patched).
- **"Bigger models will solve compositional ICL deduction."** Killed by within-Gemma-family scale anti-correlation on the closed +2 chain.
- **"Abstract framings help reasoning."** Killed by `gauss_euler` and `compose_hint` both hurting.

What it does *not* rule out:

- That different model families form the ring through different mechanisms (induction-head copy vs something else). Gemma 4's uniform failure suggests architecture-or-training-data matters; the sweep doesn't yet say which.
- That the ring is the only causally-relevant geometry. Qwen 2.5 3B solves with a different geometry, which is its own thread.
- That the deduction wall is fundamental rather than a property of in-context-only computation. CoT-generation accuracy with `explicit_task` preface is the obvious follow-up.

---

## Reproducing

All code is at `manifold_days/` in the research repo.

```bash
# manifold + accuracy for one model
python icl_induce_manifold.py --model google/gemma-2-2b --out_dir results_icl_gemma2_2b

# accuracy check across models, no-header variant
python icl_accuracy_check.py --no_header --models google/gemma-2-2b meta-llama/Llama-3.2-3B

# deduction sweep
python harder_accuracy_sweep.py
python harder_preface_sweep.py

# cross-task patching
python patch_direct_to_mixed.py
python patch_controls.py
python patch_pc_sweep.py

# closed +2 chain
python closed_chain_sweep.py
```

Each manifold run produces `scores_lpos.json` (per-layer metrics), `icl_induced_lpos.html` (interactive 3D PCA with layer slider), and `cache.npz` of the collected residuals — useful for trying different analyses without rerunning the model.
