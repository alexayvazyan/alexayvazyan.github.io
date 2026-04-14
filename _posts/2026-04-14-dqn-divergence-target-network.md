---
layout: post
title: "DQN Divergence without a frozen target approximator"
date: 2026-04-14
---

# DQN Divergence without a frozen target approximator. 

---

## Background

I was working through programming Pong to be played via DQN. I paused the project after realising my MacBook probably didn't have the compute to produce something that could actually win. Fast forward to getting a 5090: I tried again and ~100x'd my training, which quickly let me see some problems. Even with all the extra steps, my model wasn't learning any better, and having read the original DQN paper from Google DeepMind I had a suspicion of what the problem was. Sure enough, I plotted Q values over training and saw them clearly exploding to the millions — training was diverging. The DQN paper mentioned that one must freeze the target network during training, only periodically syncing it, and that sounded like exactly the fix I needed. As expected, training began to improve immediately. But I felt a bit unsatisfied. I didn't feel like I'd developed an intuition for why the Q values were diverging in the first place, nor for how freezing a target network actually helps. So I set up some demos to try and illuminate the problem.

The plan was the cleanest possible setup I could come up with:

- Two states S1 and S2, with S1 → S2 → terminal and reward 0 on every transition. The true value is trivially 0 everywhere.
- A tiny value network `V(s)` with a one-hot state, n hidden units, no biases, scalar output. That's `3n` weights total: `2n` in the input matrix `W` and `n` in the readout `v`. For simplicity, I mostly looked at the case where `n=1`.
- Randomly initialise and run the semi-gradient TD update. Check how often it blows up.

## Working a toy example by hand

Having read the [Deadly Triad paper](https://arxiv.org/pdf/1812.02648), I noticed this setup was similar to the example given by Tsitsiklis and Van Roy in 1997, which I first worked through by hand. One of the keys to causing the divergence was the random nature of the sampling. Obviously in our dynamical environment above we expect to hit S1 and S2 in equal proportion. But if we bootstrap, we'll certainly sample gradient updates from both states in a non-sequential order.

If we fix `n=1` and pin the input weights to `w1 = 1` and `w2 = 2`, leaving only the readout `v` to learn, then `V(S1) = v` and `V(S2) = 2v`. Sampling just from S1, the semi-gradient update is `Δv = lr · (2γv − v)`, which is a diverging step for `v` if `γ > 0.5`. If we sample from S2 next, we move `v` back toward 0 and the system definitely converges. But that's the intuition: if we sample sufficiently "unfairly" (which happens by chance when we bootstrap) in a system where the parameters we tune can affect downstream value calculations, our Q values can diverge. This is the deadly triad identified in the paper.

Now back to the program — no fixed weights this time. One step closer to the real problem, but less practical to work through by hand.

---

## The setup, concretely

The network has two sets of weights: an input matrix `W` of shape `(n, 2)` that maps the one-hot state to the hidden layer, and a readout vector `v` of length `n` that maps the hidden layer to the scalar output. With a linear activation the value function is just

```
V(s) = v · (W s)
```

Because `s` is a one-hot vector, `W s` simply picks out one column of `W`. Concretely, for state S1 we take column 0; for state S2 we take column 1. Calling those columns `w1` and `w2`,

```
V(S1) = v · w1
V(S2) = v · w2
```

The shared piece is `v` — it appears in both expressions. That's the only path by which a gradient on `V(S1)` can move `V(S2)`. The TD error is

```
δ = V(S1) - γ·V(S2) = v·w1 - γ·v·w2
```

The semi-gradient update (treating the target as constant, as in DQN) takes the gradient of `δ²/2` only through the `V(S1)` term, so it moves `v` and `w1` and leaves `w2` alone (`∂V(S1)/∂w2 = 0`). And because we only ever visit the S1 → S2 transition, `w2` never receives a gradient from anywhere else either. That last bit matters — it's the "skewed sampling" half of the deadly triad. The "function approximation" half is the shared `v`.

No biases, no non-linearity, no replay buffer. Just one transition, one gradient step, repeat.

---

## First try: does it actually diverge?

With `n=4`, `γ=0.99`, `lr=0.1`, i.i.d. standard-normal init: **zero of 200 seeds diverge**. Same with `lr=0.2`. I had to crank `lr` all the way to `0.5` before ~40% of seeds blew up, and at that point the target network was *also* diverging at a similar rate. So whatever was blowing up at `lr=0.5`, it wasn't the thing the target network is supposed to fix.

This surprised me. The textbook story says DQN *needs* a target network, but my three-weight toy was shrugging and converging just fine. Time to stop hand-waving and look at the dynamics.

---

## The contraction rate, by hand

One step of the update changes `v` and `w1` by

```
Δv  = -α·δ·w1
Δw1 = -α·δ·v
```

Call `a = V(S1) = v·w1`, `b = V(S2) = v·w2`, `c = ‖v‖²`, `d = ‖w1‖²`, and `f = w1·w2`. Expanding (keeping only the leading terms in `α`):

```
Δa ≈ -α·δ·(c + d)
Δb ≈ -α·δ·f
```

So the TD error itself evolves as

```
δ_new = Δa - γ·Δb = δ · [1 - α·(c + d - γ·f)]
```

That bracketed factor is the contraction rate. For the update to converge we need its magnitude below 1, i.e.

```
0  <  α·(c + d - γ·f)  <  2
```

Compare to the **target-network** case: there, `V(S2)` is frozen, so `Δb = 0` and the bracket becomes `[1 - α·(c + d)]`. The `γ·f` term — which is *exactly* the shared-weight coupling — is gone.

Now the textbook story sharpens into a testable claim. The target network specifically removes the `γ·f` term. So:

- **If `f < 0`** (columns anti-aligned): `c + d - γ·f` is *larger* than `c + d`. The bootstrap update is *less* contractive than the target-net update, and can blow past 2 into the unstable regime while the target-net update is still fine. This is the bootstrap-feedback divergence.
- **If `f > 0`**: `c + d - γ·f` is *smaller* than `c + d`. The bootstrap update is actually *more* contractive (you get one free bit of alignment). Target network doesn't help; can even hurt.
- **If `α·(c + d)` is already > 2**: both updates diverge. This is overshoot from the learning rate being too large relative to the weight magnitudes, and it has nothing to do with bootstrapping. Shared weights are irrelevant; target network can't save you.

With i.i.d. standard-normal init and `n=4`, `E[f] = 0` and `std(f) ≈ 2`, while `c + d` concentrates around `2n = 8`. For `γ·f` to flip the sign or push past 2 you need a very unlucky draw. That's why nothing diverged at moderate learning rates — the shared-weight coupling just wasn't strong enough on a random draw.

---

## Two modes, confirmed empirically

Once I had the algebra, the fix for the demo was obvious: let me initialise `w1` and `w2` with a chosen correlation instead of independently. Setting `ρ = corr(w1, w2) = -1` gives `f < 0` reliably and should expose the bootstrap regime cleanly.

![Divergence probability vs gamma for two init regimes](/assets/images/dqn-divergence_exp5_gamma.png)
*Fraction of seeds that diverge as γ sweeps from 0 to 1, every other hyperparameter held fixed (`lr=0.2`, `n=4`, `init_scale=1.2`). Left: anti-correlated init (`ρ=-1`) — the bootstrap regime. Divergence rises monotonically with γ, and the target network removes a consistent chunk of it. Right: i.i.d. init (`ρ=0`) — much less divergence overall at the same settings, and the target-net curve sits right on top of the no-target curve. The gap between the two panels is exactly the piece of the contraction rate that depends on `γ·w1·w2`.*

The left panel is the scenario every RL textbook is implicitly picturing: divergence probability grows with γ (more bootstrap weight → more target movement), and the target network helps. The right panel is what I stumbled into first by accident: with the columns drawn independently, the `γ·w1·w2` term averages to zero, very little diverges at all, and whatever does diverge is overshoot from the learning rate — which the target network has no opinion on.

The `(γ, lr)` heatmap tells the same story more densely:

![Gamma-lr heatmap, anti-correlated init](/assets/images/dqn-divergence_exp2_heatmap_anticorr.png)
*Divergence fraction over `(γ, lr)` with anti-correlated init. The third panel is `no-target minus target` — orange = target net reduces divergence. There's a clear band of learning rates (roughly `0.15–0.3`) where the bootstrap update is unstable but the target-net update is still fine.*

![Gamma-lr heatmap, iid init](/assets/images/dqn-divergence_exp2_heatmap_iid.png)
*Same sweep with i.i.d. init. The difference panel is almost perfectly white. The divergence that appears here is the bilinear-overshoot mode, which the target network can't touch.*

I found this distinction genuinely useful for internalising what the target network does. It's not "stabilises DQN" in some general sense — it specifically removes one additive term in the contraction rate, the one that couples `V(S1)` through `V(S2)`. Any other instability your algorithm has, the target network is going to walk right past.

---

## Drawing the divergence region

The question I actually wanted to answer for my own intuition was: *given γ and lr, which initialisations diverge?* For `n=1` the whole network has three scalar parameters — `w1`, `w2`, `w3` (where `w3` is the single readout weight). That's small enough to just sweep.

Fixing `γ=0.9`, `lr=0.05`, no target network, I swept every `(w1, w2, w3)` on a `51³` grid in `[-8, 8]³` and recorded which initialisations hit `|V| > 10⁶` within 5000 steps. Note the cube has to be this big — from the algebra, divergence requires

```
w1² + w3² - γ·w1·w2  >  2/lr = 40
```

so inside the unit ball nothing ever blows up.

![2D slices of the divergence set at fixed w3](/assets/images/dqn-divergence_exp6_slices.png)
*Red cells diverge; cream cells converge. Each panel is a `w1`-`w2` plane at a fixed `w3`. The blue dashed line is the TD fixed-point line `w1 = γ·w2`. At `w3 = 0` the readout is trivial and everything is stable — you can't blow up if the output is zero. As `|w3|` grows the red regions expand from the corners toward the fixed-point band, eventually leaving only a thin strip of stable initialisations around `w1 = γ·w2`.*

Two things clicked for me from this picture:

1. **The stable set is a band around `w1 = γ·w2`, not a ball.** The classical "small weights are safe" intuition is wrong in a specific way — small `w3` is safe (the `w3² + w1²` penalty term), but given a large `w3`, safety depends on `w1` and `w2` being on the right *line* together, not on being small.
2. **The divergence set has the right asymmetry.** The term that shows up in the contraction rate is `-γ·w1·w2`, not `-γ·|w1·w2|`. Points with `w1·w2 < 0` (opposite-sign columns) are the ones that get amplified. In the slices you can see the red corners concentrated where `w1` and `w2` have opposite signs at large `|w3|`.

The 3D view shows the full shape:

![3D shell of the divergence boundary coloured by w3](/assets/images/dqn-divergence_exp6_scatter.png)
*The boundary between stable and unstable initialisations, coloured by `w3`. The interior near `w3 ≈ 0` is stable; the two curved sheets at large `|w3|` are the divergence boundary. 22% of the cube diverges in total.*

If you wanted a one-image explanation of why DQN is unstable without a target network, I think the slice grid is probably it. The stability band tracks a fixed line that depends on `γ`. Training moves through this space, and the bootstrap update has no particular reason to keep you on the band.

---

## When target interval actually matters

The target update interval is usually treated as a hyperparameter to tune. On this toy I can actually see what it's controlling:

![Divergence fraction vs target network update interval](/assets/images/dqn-divergence_exp3_target_interval.png)
*Divergence fraction as the target network update interval varies from 1 (≡ no target) to 1000. Same `γ=0.99`, `lr=0.2`, `n=4`, `init_scale=1.2` in both panels — only `ρ` changes. Blue: anti-correlated (`ρ=-1`) — longer intervals monotonically reduce divergence, which matches the "freeze the target to break the feedback loop" story. Orange: i.i.d. (`ρ=0`) — much less divergence to begin with, and the interval barely moves the needle. The dashed lines are the corresponding `no target` baselines.*

So the interval is doing real work when the bootstrap term is active, and essentially nothing when it isn't. Which suggests the answer to *"how often should I refresh the target network"* depends on whether your actual environment has the anti-aligned-column structure that makes the γ·w1·w2 term bite — something you can probably diagnose empirically if you have the patience, but can't really know a priori.

---

## Summary

- I wanted to reproduce DQN divergence on a 2-state, 3n-weight toy with bootstrapping from a biased sample. The textbook mechanism — shared weights making the bootstrap target move — is real, but with i.i.d. random init it's typically *not* what actually blows up first. What blows up first is bilinear-overshoot instability from the learning rate.
- Writing out the contraction rate cleanly separates the two: without a target net the rate is `1 - α·(‖v‖² + ‖w1‖² - γ·w1·w2)`; with a target net it's `1 - α·(‖v‖² + ‖w1‖²)`. The target network's effect is one specific term: `γ·w1·w2`.
- That term only matters when `w1·w2 < 0`, i.e. when the state columns are anti-aligned. Initialising with `corr(w1, w2) = -1` reliably exposes the bootstrap regime, and in that regime the target network measurably reduces divergence probability.
- Drawing the divergence set in `(w1, w2, w3)`-space reveals that the "safe" region isn't a ball around the origin — it's a band around `w1 = γ·w2`, thinning as `|w3|` grows. Small weights aren't automatically safe; small-and-correlated-the-right-way weights are.

---

*Code lives under [rl-testing/divergence_demo](https://github.com/alexayvazyan/projects) in my projects repo. This page was drafted by Claude from my outline of the investigation and then edited by me.*
