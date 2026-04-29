---
layout: post
title: "Geometry of Speed and Size Across Multimodal Stimulus in Gemma 4 31B"
date: 2026-04-29
---

# Geometry of Speed and Size Across Multimodal Stimulus in Gemma 4 31B

*A physical scalar like "how fast is this square moving" or "how big is this square" gets a clean low-dimensional representation in Gemma 4 31B's residual stream — a curved arc through PCA, ordered monotonically with the ground-truth scalar, causally steerable. But the moment you ask the model to do anything compositional with these scalars (compute momentum p = m·v, for example), accuracy collapses to chance. The geometry is there. The arithmetic on the geometry isn't. I think the bottleneck is information density: the task asks too many distinct quantities to be extracted, held, and combined inside a single forward pass.*

---

## Background

This is part of a pair of probes I've been running on Gemma 4 31B-it: one on perceived **speed** of a translating square across a sequence of frames ([gemma4-speed-probe](https://github.com/alexayvazyan)), and one on perceived **size** of a stationary square in a single frame ([gemma4-size-probe](https://github.com/alexayvazyan)). Both projects use the same scaffold: render simple synthetic stimuli, cache residual stream activations across the layer stack, look at the geometry of the variable's representation, and try causal interventions to confirm what the geometry is telling us.

The motivation came from recent geometry-of-relativity work framing scalar representations in vision-language models as low-dimensional manifolds with structure that the rest of the network can read. I wanted to test whether that picture held for Gemma 4 31B in particular, and whether it extended from "read off a scalar" to "do something with the scalar."

---

## Speed: clean manifold, clean steering

The speed stimulus is a sequence of N frames showing a black square translating horizontally on a grey canvas. The continuous variable is `dx`, the per-frame displacement. I cache the residual stream at the final image-token positions, sweep `dx`, do PCA across the manifold of (dx-binned) activations.

What falls out is the same shape that's been observed on Gemma 2: a curved arc through the top principal components, ordered monotonically with `dx`. The ordering is preserved layer by layer through a clear window in the middle of the network. Earlier layers have not yet computed the scalar; later layers begin to project it onto language-aligned directions and the geometry rotates into something less neatly recoverable.

![Gemma 4 31B-it speed manifold at L=30, six rotated views of the 3D PCA. The arc structure is visible from several angles: stationary stimuli cluster off to one side, and the moving stimuli sweep through a curved trajectory ordered by dx (per-frame displacement, colored).](/assets/images/gemma4_31b_speed_manifold_L30_panel.png)
*Speed manifold at L=30. Six rotated views of the top-3 PCA. Stationary stimuli (small dark markers) sit at one end; moving stimuli sweep through a curved trajectory ordered by dx.*

Causal steering works as expected: sample a "slow" cache and a "fast" cache, take the chord between them, add the chord (scaled) at the right layer, and the model's verbal report shifts predictably along the slow → medium → fast axis. The clean version of this, on the **size** axis, was the result I noted in my memory from a few days ago: chord-vs-arc steering with a 31B-it model at L=56, where the verbal bins shift one slot down from the manifold means — meaning the geometry exposes a finer structure than the model's verbal vocabulary can name, but the steering still tracks the geometry.

There's also a "dead-large" effect that mirrors the "dead-fast" result on speed: the verbal vocabulary saturates before the manifold does. The model declines to call any of the stimuli "fast" even when the displacement is 80% of canvas width, and similarly will not call a square "large" past a certain side length even though the manifold cleanly separates those bins. The geometry encodes more than the language report exposes.

So far so good. The scalar manifold story holds.

---

## Size: same manifold, same steering

The size probe is the static counterpart. One frame, one square at a known side length `s`, sweep `s`, cache, PCA. Same arc shape, same monotonicity, same steering behavior. The arc through the top components is the model's notion of "how big is this" before it gets compressed onto whatever discrete vocabulary it eventually uses to answer.

![Gemma 4 31B-it size manifold at L=56. Top-2 PCA. A clean curved arc sweeps from small squares (purple, lower-left) through medium (green/teal) to large (yellow, lower-right). Group means for tiny / small / medium / large are marked with stars, connected by grey lines showing the polyline through the four bins. Per-stim variance in the 3D size-subspace is 91.3%.](/assets/images/gemma4_31b_size_arc_L56.png)
*Size manifold at L=56. Top-2 PCA explains 98.3% of variance in the size subspace. A clean curved arc, monotonic in side length, with a polyline through the bin means showing the path the model takes from "tiny" to "large".*

The chord-vs-arc steering experiment tests two related things at once. First, whether the curvature of the arc is geometrically necessary — does steering along the chord (straight line from `μ_tiny` to `μ_large`) actually move the model through the intermediate verbal bins? Second, whether the verbal vocabulary itself is the bottleneck — can the model report "large" at all when steered onto the large region of its own manifold?

![Chord vs arc steering at L=56 in Gemma 4 31B-it. Two panels show P(label) vs steering coefficient α. Chord steering (left) produces clean graded transitions: tiny → small → medium dominate as α increases, with smooth crossovers between consecutive bins. Arc steering (right) is dominated by "small" across most of α, with "tiny" only at α=0 and "medium" appearing only near α=1. In both panels, "large" (red) never rises above zero.](/assets/images/gemma4_31b_size_steer_chord_vs_arc.png)
*Chord (left) vs arc (right) steering. Chord traversal cleanly elicits each verbal bin in order. Arc traversal gets stuck in "small" for most of α, consistent with the manifold's group means landing one slot below the verbal bins they were supposed to name. In both, "large" never wins — this is the **dead-large** result: the model's verbal scale saturates before its representational scale does.*

Two readings here. The "dead-large" effect mirrors a "dead-fast" result from the speed probe: the verbal vocabulary saturates before the manifold does. The model declines to call any of the stimuli "fast" even when displacement is 80% of the canvas, and similarly will not call a square "large" past a certain side length even though the manifold cleanly separates those bins. The geometry encodes more than the language report exposes.

The other reading is about the alignment between the manifold's group means and the verbal bins they were *meant* to name. Arc steering lands at `μ_small` when α=0.5 and the verbal report says "small" — but it also stays "small" most of the way to `μ_medium`, and only flips to "medium" right at the end. Chord steering, going through space the manifold doesn't actually pass through, somehow elicits cleaner graded reports. My current read is that the manifold means are shifted one slot down from the verbal bins (so `μ_medium` is closer to where "small" decodes from, etc.), and the chord just happens to clip across regions that decode more cleanly. Either way, both interventions confirm dead-large from a different direction: there is no place on either path that decodes as "large".

This pattern is consistent across the speed and size manifolds. Two different physical scalars, two different stimulus paradigms, both producing curved arcs that are causally functional but whose endpoints don't quite line up with the model's verbal report.

---

## Where it falls apart: momentum

The natural next question, and the one that motivated this post, is: if the model has clean representations of `s` (size) and `dx` (speed) on tap, can it compute something compositional from them? Specifically, momentum `p = s² · dx` (using the square's mass as proportional to area, side² for a 2D object).

I designed a clean two-stage test:

**Stage 1 (text-only).** Hand the model `s` and `dx` directly as numbers in the prompt and ask for `s² · dx`. This is just an arithmetic test, no vision required. The model gets this right a healthy fraction of the time when the numbers are small and exact, and degrades on larger values — the standard arithmetic-in-LMs behavior. Critically, the failure mode is *off-by-some*, not categorical: predictions cluster around the true value with a noisy spread. The arithmetic capability is there; the precision isn't perfect.

**Stage 2 (visual).** Render a frame with a square of side `s` at position `(x, y)`, hand the model the frame *plus* a stated displacement `dx` (or a frame sequence), and ask for `s² · dx`. This is the probe I actually care about, because it forces the model to extract `s` from the image (which the size manifold says it can do), combine with `dx`, and produce the product.

Accuracy on stage 2 collapses to essentially zero across the parameter sweep. Looking at the raw outputs, the model isn't doing the multiplication wrong — it's failing to read off the right inputs in the first place. Predictions look like the model is sometimes computing `dx²` (using the displacement as if it were the side length), sometimes computing `x · y` (using the position coordinates), sometimes returning a small integer that doesn't correspond to any obvious combination of inputs. The pattern varies trial to trial. The arithmetic *capability* is intact (stage 1 confirms this); the *binding* between the right scalar and the right symbol has broken.

This is striking because we have direct evidence — from the size manifold itself — that `s` is represented in the residual stream. The model "knows" how big the square is, in a probe-readable, causally-steerable sense. But that knowledge is not making it into the right slot of the arithmetic computation downstream.

---

## The information-density hypothesis

The picture I've converged on is that the failure isn't a missing capability, it's a **bandwidth** failure. A single forward pass through Gemma 4 31B has to do, in order:

1. Visual encoding: turn the image into image tokens.
2. Scalar extraction: build the size manifold from those tokens. (Confirmed.)
3. Symbolic binding: associate the extracted size with the variable name `s` from the prompt.
4. Co-extract a second variable, `dx`, either from a second image or from the prompt text.
5. Symbolic binding: associate `dx` with its variable name.
6. Arithmetic: compute `s² · dx`.
7. Decode: emit the right number.

Each of these steps consumes representation bandwidth. The size manifold work shows that step 2 is happening cleanly. Stage 1 of the momentum test shows that steps 6–7 are working. What's failing is the chain in between — specifically, holding two scalars *and* their bindings *and* the arithmetic plan in the same forward pass.

A few observations that support this:

- **Removing CoT makes it worse, not better, but adding CoT doesn't fully fix it.** When the model is allowed to write out intermediate scratch ("the side is about 48, the displacement is 88, so 48² is 2304, times 88 is..."), accuracy improves but not to ceiling, and the bottleneck shifts to whether the model writes down the *correct* `s` in the first sentence.
- **Position information is leaking in.** When the prompt asks for momentum from a single static frame (so dx must be inferred or supplied), the model frequently substitutes `x` or `y` (the position coordinates) for `dx`. The visual encoding has all of these scalars present, and the wrong one is winning the slot competition.
- **The error pattern is consistent across runs.** It's not noise around the right computation — it's the model picking a *different, internally-coherent* computation that happens to be wrong. This looks like a binding failure, not an arithmetic failure.

---

## Where this leaves the project

I've paused the momentum extension while I think about how to test the bandwidth hypothesis directly. The cleanest version would be:

**Step 1.** Confirm via patching that the size manifold *is* being read by the arithmetic head when the task is "what is `s²`" (single scalar, single op). This is the easiest version of compositional use and should work. If it doesn't, the diagnosis is wrong.

**Step 2.** Patch the size representation from a clean run into a momentum run, and see whether accuracy recovers. If yes, the bottleneck is extraction-under-load, not extraction in general.

**Step 3.** Try the explicit two-pass version: first prompt asks the model to emit `s` and `dx` from the frame as numbers; second prompt feeds those numbers back and asks for the product. If this works cleanly, the bottleneck is confirmed to be single-pass density, and the natural next question is whether the same model can do the task with intermediate hidden-state rollovers (a kind of recurrence) rather than full text decoding.

The broader bet is that this isn't a Gemma-specific failure. Any current-generation VLM that has to extract two scalars and combine them in a single pass should hit the same wall, and the wall should move outward roughly with parameter count. The clean way to test this would be to run the same momentum probe on Gemma 4 4B, 12B, and 31B and check whether the failure mode is severity-graded or qualitatively different. My guess is severity-graded, with the 31B model getting closest to ceiling but not crossing it.

---

## Connection to the broader research arc

This sits next to the [ICL deduction failure work](/) and the [refusal-direction rotation work](/2026/04/09/refusal-direction-ablation.html) as another instance of the same general pattern: models have clean low-dimensional representations of relevant features, but the *use* of those features in compositional downstream tasks is fragile in ways that the geometry alone doesn't predict. You can't tell, from looking at a clean refusal direction, whether ablating it will break the model. You can't tell, from looking at a clean size manifold, whether the model will be able to *use* size in a calculation. The geometry is necessary but not sufficient.

The thing I want to land on, eventually, is a story about which kinds of compositional uses are bandwidth-limited (single-pass extraction of multiple scalars) versus which are more deeply broken (compositional ICL deduction at chain length 2+, where my [closed-chain sweep](/) shows even the cleanest setup caps at ~28% accuracy). The momentum failure looks like the former. I think it's separable from the latter.

---

## Open questions

- Does patching `s` from a clean cache into a momentum-prompt forward pass restore accuracy?
- Does the bandwidth bottleneck scale predictably with parameter count, or is it a step function?
- Is there an architectural prediction here — that wider residual streams (more dim) should buy more simultaneous slots, all else equal?
- Does the same failure pattern hold when the two scalars come from two different modalities (e.g., size from image, displacement from text)? My current setup mostly mixes them; cleanly separating could isolate where the binding fails.
