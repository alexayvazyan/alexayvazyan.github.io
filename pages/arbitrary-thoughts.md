---
layout: post
title: "Arbitrary Thoughts"
permalink: /arbitrary-thoughts.html
---

# Arbitrary Thoughts

*A collection of various thoughts, normally formulated in bed while trying to sleep.*

---

## 1 — Mean variance optimization

In the course of my work as a quant trader I encountered the mean-variance objective:

```
O(h) = hᵀμ − γ · hᵀΣh
```

where `h` is the proportion vector of holdings across various securities, `μ` and `Σ` are the mean return vector and covariance matrix of those securities, and `γ` is a hyperparameter.

Initially it wasn't obvious to me that such a formulation could be equivalent to maximizing Sharpe, even though I was told it was. I knew the reason it's normally written this way was a desire for linear and fast differentiation.

Thinking about it more though: if you fix the covariance matrix or the mean return vector, the optimizer maximizes/minimizes the other. In that sense, optimizing this objective guarantees you sit on the 'efficient frontier' described in Markowitz portfolio theory.

With `γ = 0` you land on the point of the frontier furthest from the origin. As `γ → ∞` you approach the basin of minimum variance. Moving `γ` up thus sweeps you across the efficient frontier, and at some point hits the optimal-Sharpe point. This also gives an intuitive reason why `γ` should scale hyperbolically.

![Efficient frontier for a 5-asset portfolio with the γ sweep marked from 0 (max return) through the max-Sharpe tangency point to γ→∞ (min variance)](/assets/images/mean_variance_efficient_frontier.png)

---

## 2 — 12 balls and 3 weighs

A classic puzzle. You have 12 balls, one of which is a fake that weighs slightly more or less than the others. What is the minimum number of scale-weighings needed to identify the fake? The answer turns out to be 3.

The standard way to solve this — and what almost anyone reaches for — is to split into cases and work through each path. If that's all there was to it the problem wouldn't be that interesting. But looking at just the first weighing, we can already see some structure in what we'd want to do.

To begin, weighing an uneven number of balls on either side clearly makes no sense. Our choices at the start are thus 1-vs-1, 2-vs-2, ..., 6-vs-6. It isn't too hard to deduce that 4-vs-4 is the best, and we can formalize why by taking an information-theoretic perspective.

There are 24 possible states of the system: 12 choices for which ball is fake, doubled by whether that ball is heavier or lighter. That's `log_2(24) ≈ 4.6` bits of entropy. A 4-vs-4 weighing effectively removes 1/3 of the possible states, bringing the entropy down by `~1.6` bits.

![First-weighing option landscape for n=12, 40, 100 — entropy reduction against the number of balls placed on the scale, red dot marks the greedy pick](/assets/images/balls_scale_initial_option_landscape.png)

What intrigued me about this problem was that the information theory perspective seems deceptively useful: it doesn't really tell us a lot about how to *solve* the problem. Sure, it gives us a greedy algorithm — at each step, do the weighing that reduces the number of possible states by the most — but that's pretty minimal insight over just looking at cases, which is essentially what we're still doing.

It would be excellent if we could deduce some upper or lower bounds for the expected entropy reduced from a single weighing. In our example, if every weighing were as good as the first, we'd be able to do the puzzle in 3 weighs indeed, since `1.6 × 3 > 4.6` bits of starting entropy. But we have no guarantee this is possible.

A good first step is to empirically study the system for `n` balls by writing a simple program. We model an algorithm that takes the choice maximizing expected information gain, and then hand it the worst possible outcome in terms of true information gained. (We always put an equal number of balls on each side — unbalanced weighings give zero information.)

The first thing we notice is that it doesn't seem possible to reduce entropy by more than `log_2(3)` bits in the worst case, regardless of ball count. We can convince ourselves of this by contradiction. Suppose there exists a weighing with `K` possible states where the worst-case outcome gives more than `log_2(3)` bits of information. That is equivalent to saying each outcome leaves strictly fewer than `K/3` surviving states. Since every state produces exactly one of the three outcomes (left down, right down, balanced), the three surviving sets partition the original `K` states, so:

```
K = N_L + N_R + N_bal < K/3 + K/3 + K/3 = K
```

a contradiction. So at least one outcome must preserve `≥ K/3` states — i.e., the worst-case reduction is bounded above by `log_2(3)`.

![Per-step entropy reduction for selected n, dashed line at log_2(3); early steps hug the ceiling, the last step drops](/assets/images/balls_scale_per_step_reduction.png)

Great. We now have a minimum-weighings function `W_min(n) = ⌈log_3(2n)⌉`. Running the greedy against a worst-case branch for every `n` from 4 to 100 shows we sit on this bound almost everywhere, stepping up exactly where `log_3(2n)` crosses an integer.

![Greedy weighings versus n with the log_3(2n) lower bound overlaid, showing the staircase match for n=4 to 100](/assets/images/balls_scale_weighings_vs_n.png)

The logic extends a bit further. To protect against the worst case, we ideally want `K/3` states distributed into each of the three outcomes. This gives a concrete reason why certain cases — like 13 balls in 3 weighings or 40 balls in 4 — aren't doable, even though `13 × 2 = 26 < 27 = 3^3` and `40 × 2 = 80 < 81 = 3^4`. The proof isn't hard, but it's easier to see with an example. Take 40 balls, 80 states. Our first weighing is either 26-vs-26 (28 off) or 28-vs-28 (24 off). Either way, at least one of the three outcome branches surfaces with 28 remaining states, which exceeds `3^3 = 27`, so three more weighings cannot finish the job — we need four.

It seems likely then that the bound `⌈log_3(2n)⌉` is met for any `n` except `a(n) = (3^n − 1)/2`, as there's probably enough latitude at other `n` to divide the state space into three roughly equal groups. I don't have a proof, though.
