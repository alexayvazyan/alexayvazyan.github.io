---
layout: post
title: "Why DQN needs a target network, visualised on the smallest possible toy"
date: 2026-04-14
---

# Why DQN needs a target network, visualised on the smallest possible toy

*A 3-weight network on a 2-state MDP is enough to see TD bootstrapping diverge. Drawing the set of initial weights that blow up turns out to be geometrically clean — and also exposes a second, completely different divergence mode that the target network does nothing about.*

---

## Background

Somewhere in the middle of writing my own DQN from scratch, I hit the part of the textbook that says *"and of course we need a target network, otherwise the bootstrap target is non-stationary and the update can diverge"*. That's a satisfying sentence until you actually try to reproduce the divergence. I went to write a quick demo for myself and immediately had to think about it for longer than I expected.

The plan was the cleanest possible setup I could come up with:

- Two states S1 and S2, with S1 → S2 → terminal and reward 0 on every transition. The true value is trivially 0 everywhere.
- A tiny value network `V(s)` with a one-hot state, n hidden units, no biases, scalar output. That's `3n` weights total: `2n` in the input matrix `W` and `n` in the readout `v`.
- Only ever sample the (S1 → S2) transition. Never sample the S2 → terminal one. The bootstrap target for S1 is `r + γ·V(S2)`.
- Randomly initialise and run the semi-gradient TD update. Check how often it blows up.

The intuition I was trying to demonstrate is the one everyone cites: `V(S1)` and `V(S2)` share the readout `v`, so a gradient step aimed at `V(S1)` also nudges `V(S2)`, which is the very thing the target was computed from. That shared-weight coupling is the bootstrap-feedback loop, and the target network's job is to break it by freezing `V(S2)` between updates.

That's the *story*. What I actually saw when I ran it was a bit more interesting.

---

## The setup, concretely

With one-hot `S1 = e_1`, `S2 = e_2` and a linear activation, the network is literally:

```
V(S1) = v · W[:,0]       # dot product of readout with column 0
V(S2) = v · W[:,1]       # dot product of readout with column 1
```

Call column 0 `w1` and column 1 `w2`. The TD error is

```
δ = V(S1) - γ·V(S2) = v·w1 - γ·v·w2
```

and the semi-gradient update (treating the target as constant, as in DQN) moves only `v` and `w1`, because `∂V(S1)/∂w2 = 0`. We only ever visit S1 → S2, so `w2` never receives a gradient at all. That last bit matters — it's the "skewed sampling" half of the deadly triad.

No biases, no non-linearity at first (I'll come back to ReLU), no replay buffer. Just one transition, one gradient step, repeat.

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
*Fraction of seeds that diverge as γ sweeps from 0 to 1. Left: anti-correlated init (`ρ=-1`) at `lr=0.2` — the classic bootstrap regime. Divergence rises monotonically with γ, and the target network removes a consistent chunk of it. Right: i.i.d. init (`ρ=0`) at `lr=0.3` — the overshoot regime. Divergence still rises with γ but the target network does nothing, because the failure mode isn't a moving target.*

The left panel is the scenario every RL textbook is implicitly picturing: divergence probability grows with γ (more bootstrap weight → more target movement), and the target network helps. The right panel is the scenario I stumbled into first by accident: even at very high γ, the divergence that shows up is from the learning rate overshooting the bilinear instability of `V = v·w`, and freezing the target is irrelevant.

The `(γ, lr)` heatmap tells the same story more densely:

![Gamma-lr heatmap, anti-correlated init](/assets/images/dqn-divergence_exp2_heatmap_anticorr.png)
*Divergence fraction over `(γ, lr)` with anti-correlated init. The third panel is `no-target minus target` — orange = target net reduces divergence. There's a clear band of learning rates (roughly `0.15–0.3`) where the bootstrap update is unstable but the target-net update is still fine.*

![Gamma-lr heatmap, iid init](/assets/images/dqn-divergence_exp2_heatmap_iid.png)
*Same sweep with i.i.d. init. The difference panel is almost perfectly white. The divergence that appears here is the bilinear-overshoot mode, which the target network can't touch.*

I found this distinction genuinely useful for internalising what the target network does. It's not "stabilises DQN" in some general sense — it specifically removes one additive term in the contraction rate, the one that couples `V(S1)` through `V(S2)`. Any other instability your algorithm has, the target network is going to walk right past.

---

## One seed, up close

With `ρ=-1`, `lr=0.2`, `n=4`, there are seeds where the no-target run explodes and the target-net run survives from the exact same initialisation:

![Single-seed comparison, anti-correlated init](/assets/images/dqn-divergence_exp1_anticorrelated.png)
*Same seed, same initial weights, same data. Left: without a target network, `V(S1)` and `V(S2)` shoot to ±10^5 in about 5 steps. Right: with a target network updated every 50 steps, both values oscillate but stay bounded. Each target refresh gives `V(S1)` a fresh fixed regression target, which it has time to chase before the target moves again.*

Divergence is always fast when it happens — the dynamics are multiplicative, so once the contraction rate goes past 1, you're doubling each step. There's no long gradual drift to watch. You either stay in the stable basin or you hit 10^6 within a few dozen steps.

For the i.i.d. case, the same plot makes the "target network can't save you" story concrete: both panels diverge in nearly the same number of steps.

![Single-seed comparison, iid init](/assets/images/dqn-divergence_exp1_iid.png)
*i.i.d. init, `lr=0.3`. Both the no-target and target-net runs diverge within ~5 steps from the same initialisation.*

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
*Divergence fraction as the target network update interval varies from 1 (≡ no target) to 1000. Blue: anti-correlated regime — longer intervals monotonically reduce divergence, which matches the "freeze the target to break the feedback loop" story. Orange: i.i.d. regime — the curve is essentially flat, because the target network is addressing the wrong failure mode.*

So the interval is doing real work in the bootstrap-dominated regime (longer = more stable, at least in this toy), and zero work in the overshoot regime. Which suggests the answer to *"how often should I refresh the target network"* depends on which kind of instability your actual environment is closer to — which isn't generally something you can know a priori, but is something you can probably diagnose empirically if you have the patience.

---

## Summary

- On a 2-state, 3n-weight toy with bootstrapping from a biased sample, I wanted to reproduce DQN divergence. The textbook mechanism — shared weights making the bootstrap target move — is real, but with i.i.d. random init it's typically *not* what actually blows up first. What blows up first is bilinear-overshoot instability from the learning rate.
- Writing out the contraction rate cleanly separates the two: without a target net the rate is `1 - α·(‖v‖² + ‖w1‖² - γ·w1·w2)`; with a target net it's `1 - α·(‖v‖² + ‖w1‖²)`. The target network's effect is one specific term: `γ·w1·w2`.
- That term only matters when `w1·w2 < 0`, i.e. when the state columns are anti-aligned. Initialising with `corr(w1, w2) = -1` reliably exposes the bootstrap regime, and in that regime the target network measurably reduces divergence probability.
- Drawing the divergence set in `(w1, w2, w3)`-space reveals that the "safe" region isn't a ball around the origin — it's a band around `w1 = γ·w2`, thinning as `|w3|` grows. Small weights aren't automatically safe; small-and-correlated-the-right-way weights are.

---

## Open questions

- With ReLU or tanh instead of linear, both the bilinear-overshoot and bootstrap-feedback terms change. I'd guess the basic picture survives (target network still specifically removes the `γ·w1·w2`-analog term), but the divergence set's *shape* probably becomes piecewise. Worth drawing.
- In real DQN you have a replay buffer, off-policy actions, and ε-greedy exploration, all of which change the expected gradient direction. I'd bet the reason target networks are empirically so much more important there than in this toy is that those mechanisms enrich the ways `w1·w2`-analog can become persistently negative along the training trajectory — but that's speculation I haven't tested.
- `target_update_interval → ∞` monotonically improved things in the anti-correlated regime in my sweep. I'd expect this to reverse once the target gets stale enough that tracking the optimal value is the limiting factor — but in this 2-state toy there's no optimal to chase beyond zero, so that tradeoff is invisible. Would need a richer MDP to see the other side of the curve.

---

*Code lives under [rl-testing/divergence_demo](https://github.com/alexayvazyan/projects) in my projects repo. This page was drafted by Claude from my outline of the investigation and then edited by me.*
