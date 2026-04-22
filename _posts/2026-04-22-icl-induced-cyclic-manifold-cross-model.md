---
layout: post
title: "ICL can induce a cyclic manifold — but only in some models"
date: 2026-04-22
---

# ICL can induce a cyclic manifold — but only in some models

*In-context demos of a modular-7 successor rule induce a 7-fold cyclic arrangement of word centroids in the residual stream of some base LMs (Gemma 2 2B/9B, Llama 3.2 3B, Pythia 2.8B), while other models that solve the task at 100% accuracy (Qwen 2.5 3B, GPT-2 small) either produce no ring or a ring whose cyclic order is scrambled. Task-solving does not imply cyclic ring geometry — the mapping of the cycle onto the residual stream is a model-specific representational choice.*

---

## Background

A few days ago I ran a small experiment on Gemma 2 2B base: feed 21 in-context demonstrations of a modular-7 successor rule over 7 semantically unrelated nouns, and look at the residual stream at the last token of the prompt (the prediction site). The 7 words arrange themselves into a cyclic 7-fold structure in the PC1-PC2 plane in layers 16-20 — the same cycle the task specifies. This is interesting because the nouns have no natural ordering; the ring has to be *induced* by the ICL demos, not read off pretrained relationships.

That result made me want to know whether it generalized. Is "cyclic ICL → cyclic residual geometry" a property of language models, or of *this* language model?

To answer that I ran the same experiment across seven base models spanning four families (Gemma, Llama, Pythia, Qwen) and four parameter scales (124M → 9B). The headline result is a clean dissociation: **models can solve this task at 100% accuracy without forming the ring**. Scale helps but isn't sufficient. Architecture or training data matters.

---

## The task, concretely

I use 7 single-token English nouns with no semantic ordering — `apple, mountain, violin, dragon, forest, window, river`. Each tokenizes to exactly one token with a leading space in every model I tested. Treat the word at index `i` as the "word before" the word at index `(i + 1) mod 7`.

The prompt has three parts:

1. A header line: `Cycle: apple -> mountain -> violin -> dragon -> forest -> window -> river -> apple (repeats).`
2. 21 demonstrations in the form `After X: Y\n`, where `Y = words[(idx(X) + 1) mod 7]`, shuffled.
3. A query: `After {word}:`

I compare three conditions:

- **baseline** — plain English sentences mentioning the word, no ICL demos. Controls for the model's pretrained geometry of these words.
- **cyclic** — 21 successor-rule demos as above. This is the condition whose geometry we care about.
- **shuffled** — same template as cyclic but `Y` is a random word, breaking the cycle. Controls for ICL density and prompt length.

For each prompt, I take the residual stream at the final token (`lpos` — the prediction site) across every layer.

> **Why the prediction site?** An earlier version of this experiment looked at the *query-word token position* (`qpos`). That gave a null result. The cycle structure lives at the position where the model is actually computing the next token, not at the token spelling the question.

---

## The metrics

Three quantities per layer, computed on per-condition PCA(3) projections of 210 prompts:

**Angular order score (out of 7).** Take the 7 per-word centroids in PC1-PC2. Compute the angular order of the 7 centers around their centroid. Check whether that order matches the true cycle `apple → mountain → ... → river → apple` (in either direction, at any starting offset). Report the *best* match across all 7 rotations × 2 directions, as the number of centers in correct cycle position. **7/7 means the 7 words lie in perfect cycle order around a ring.**

**Ring radius.** For each point, compute its distance from the PC1-PC2 centroid. Report `mean(radius) / std(radius)`. High values mean a tight ring — most points are equidistant from the centroid. Low values mean the "ring" is actually a smear with varying radii.

**k-fold FFT power.** Histogram angular positions into 120 bins, take FFT, report the peak frequency and its power. A clean 7-cycle of points (not just centers) would peak at `k=7`. In practice this metric is noisy for 7 groups of 30 points and I treat it as supporting, not primary.

Cycle order is the load-bearing signal. Ring radius is the tiebreaker: a model can achieve 7/7 order while its points smear radially, and that smeared 7-cycle is a weaker result than a tight ring.

**Task accuracy.** Separate from the manifold metrics, I measure whether the model *actually solves the successor task in context*. I take the model's next-token logits at the query position, restrict to the 7 word tokens, and report top-1 accuracy. I run this without the header prompt to make sure the model isn't just reading the answer off the header line. Shuffled accuracy should sit at chance (1/7 ≈ 14%).

## The models

Everything below uses the **base** (pretrained, not instruction-tuned) variant. I tested this earlier on Gemma 2 2B instruct and it got 37% on the task — instruct tuning breaks the underlying representation for this probe. All seven of the following are base models.

| Model | Params | Layers | Family |
|---|---|---|---|
| GPT-2 small | 124M | 12 | OpenAI GPT-2 |
| Pythia 1.4B | 1.4B | 24 | EleutherAI GPT-NeoX |
| Gemma 2 2B | 2B | 26 | Google Gemma |
| Pythia 2.8B | 2.8B | 32 | EleutherAI GPT-NeoX |
| Llama 3.2 3B | 3B | 28 | Meta Llama |
| Qwen 2.5 3B | 3B | 36 | Alibaba Qwen |
| Gemma 2 9B | 9B | 42 | Google Gemma |

## Results

### Headline table

| Model | Cyclic acc | Shuf acc | Peak cyclic order | Peak cyclic ring | Peak layer band (% depth) | Ring forms? |
|---|---|---|---|---|---|---|
| GPT-2 (124M)   | 87%  | 17% | 3/7     | —    | —           | **No** |
| Pythia 1.4B    | 54%  | 14% | 4/7     | —    | —           | **No** |
| Gemma 2 2B     | 94%  | 18% | **7/7** | 3.75 | L16-20 (62-77%)  | Yes, tight |
| Pythia 2.8B    | 98%  | 17% | **7/7** | 1.85 | L23-30 (72-94%)  | Yes, weak radius |
| Llama 3.2 3B   | 100% | 18% | **7/7** | 5.75 | L15-27 (54-96%)  | Yes, tight |
| Qwen 2.5 3B    | 100% | 18% | 5/7     | —    | —           | **No** |
| Gemma 2 9B     | 100% | 18% | **7/7** | 4.60 | L22-28 (52-67%)  | Yes, tight |

### The ring, visualized

![7-panel PC1-PC2 per-model grid]({{ site.baseurl }}/assets/images/icl_manifold_7panel.png)

Each panel shows the PC1-PC2 projection of the cyclic-condition prediction-site residual at each model's peak-order layer. Small dots are individual prompts, large circles are per-word centroids (colored by word), and the gray polygon connects centroids in PC1-PC2 angular order. Green titles mean a clean cyclic ring (7/7 order, Gemma 2B/9B, Llama 3B, Pythia 2.8B). The yellow title flags Qwen 2.5 3B — centroids sit on a ring but the colors are scrambled relative to the true cycle, so the ring doesn't encode the successor rule. Red titles mark GPT-2 and Pythia 1.4B where there's no ring at all.

### The layer band

![Cyclic order vs normalized depth]({{ site.baseurl }}/assets/images/icl_manifold_depth_sweep.png)

Plotting cyclic-condition angular order against normalized depth, the ring-forming models (solid lines) all hit 7/7 in a **mid-to-late band** — roughly 50-80% through the network. Below that depth the representation hasn't organized yet; above it the signal sometimes weakens again as the model commits to the next-token distribution.

### Ring radius separation

![Ring radius at peak layer]({{ site.baseurl }}/assets/images/icl_manifold_ring_radius.png)

Even among models that achieve 7/7 angular order, the *ring radius* (tightness of the circle) varies widely. Llama 3.2 3B and Gemma 2 9B have ring radii 2-3× the shuffled control — points really do sit on a ring. Pythia 2.8B is a marginal case: the angular order is 7/7 but the radius barely separates from shuffled (1.85 vs ~1.8) — the points are in cycle order but scattered radially.

---

## The most informative case: Qwen 2.5 3B

The single most informative result in this sweep is **Qwen 2.5 3B**. It has:

- **100% top-1** accuracy on the cyclic successor task (chance is 14%).
- Essentially chance accuracy (18%) on the shuffled control, so it genuinely learns the rule from demos — it isn't exploiting some word-frequency artifact.
- **Peak cyclic angular order of only 5/7** — 2 of the 7 centroids are in the wrong rotational position.
- But a **ring radius of 3.54** vs shuffled 2.22 — the centroids *do* sit on a ring-like shape, similar to Gemma 2 2B's (3.75 vs 1.94). They just aren't arranged in cycle order around it.

Compare to Llama 3.2 3B: same size, same 100% accuracy, same 18% shuffled baseline, and a ring of roughly the same geometric quality — but **7/7 cyclic order**. Both models arrange the 7 words on a circle. Only one of them does so in the order dictated by the task.

**Qwen's behavior kills the hypothesis that "the model encodes the successor rule as a cyclic ring" is a necessary consequence of solving the task.** Qwen is solving it at 100% with a circular geometry that isn't aligned to the cycle the task defines. Its representation of the 7 words is compressed onto a ring, but the cycle sits on a different subspace — or different mechanism — that PC1-PC2 at this layer doesn't expose. Possibilities include a higher-dimensional mapping, a direct induction-head copy from the demonstrations, or a non-geometric indexing scheme.

The same point applies to GPT-2 small (87% accuracy, 3/7 order, ring radius barely above shuffled). 87% is not chance. Whatever GPT-2 is using, it isn't the cyclic ring.

---

## The within-family scaling result

Pythia 1.4B → Pythia 2.8B gives a clean scale-within-family comparison:

- Pythia 1.4B: **54%** accuracy, 4/7 order, no ring.
- Pythia 2.8B: **98%** accuracy, 7/7 order at L23-30, weak ring radius.

So scale *does* help within the Pythia family — 1.4B is below the threshold where both task-solving and ring formation kick in; 2.8B clears it. But the ring Pythia 2.8B forms is angular-only (radius 1.85, barely above shuffled). Llama 3.2 3B and Gemma 2 2B, at smaller or comparable scale, both form geometrically tighter rings (radius 3.75-5.75). So scale is a necessary condition within a family but not the only factor determining ring quality.

---

## Discussion

The three things this sweep tells me:

**1. The ring is real and it generalizes across some families.** Four of seven base models show 7/7 cyclic angular order at the prediction site in a mid-to-late layer band. Gemma, Llama, and Pythia all form it. It's not an idiosyncrasy of the original Gemma 2B result.

**2. Task-solving is necessary but not sufficient.** You need the model to actually learn the successor rule from demos — Pythia 1.4B at 54% is subthreshold. But Qwen 2.5 3B at 100% shows that solving the task is not sufficient for the ring geometry to appear. Models can solve ICL tasks with qualitatively different internal representations.

**3. The ring quality separates further by architecture / training.** Among the models that form a ring, ring radius ranges from 1.85 (Pythia 2.8B — angular cycle only) to 5.75 (Llama 3.2 3B — clean tight ring). The "does it form a ring" / "how tight is the ring" axes give you two signals per model, and the second axis probably carries the mechanism story.

What's missing from this post: an attention-pattern probe on Qwen 2.5 3B and GPT-2 to see *how* they solve it without a ring. My guess is induction-head copy — especially for GPT-2, where the header literally spells out the cycle. But I haven't confirmed it yet, and Qwen's case is more interesting because it still works when I remove the header (100% accuracy, no-header variant).

Also missing: the SAE interpretation. The reason I chose models with available SAEs (Gemma Scope covers all Gemma layers; Llama and Pythia have community SAEs) is that once you've localized a ring, you can decode what features the 7 axes of separation actually correspond to. That's the natural next step — and the one that would turn "there's a ring" into "here's what the ring is made of."

---

## Reproducing

All code is at `manifold_days/` in the research repo. Reproduction:

```bash
# one model, one output directory
python icl_induce_manifold.py --model google/gemma-2-2b --out_dir results_icl_gemma2_2b

# accuracy check across models, no-header variant
python icl_accuracy_check.py --no_header --models google/gemma-2-2b meta-llama/Llama-3.2-3B
```

Each manifold run produces `scores_lpos.json` (per-layer metrics), `icl_induced_lpos.html` (interactive 3D PCA with layer slider), and a `cache.npz` of the collected residuals (per-condition, per-layer, per-prompt — useful if you want to try different analyses without rerunning the model).
