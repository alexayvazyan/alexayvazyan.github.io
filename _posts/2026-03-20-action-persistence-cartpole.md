---
layout: post
title: "Hyperparameter tuning in Cartpole and Action Persistence"
date: 2026-03-20
---

## Background

<PLACEHOLDER: 2-3 sentences — frame this as a question about credit assignment: for a discount factor gamma to work, early actions need to receive credit for late outcomes. That requires the action's causal influence to persist long enough for gamma's discount curve to still be non-negligible. This experiment asks: empirically, how long does a single action actually matter in CartPole?>

As part of learning RL, I was playing around in the OpenAI gymnasium trying to implement a training algorithm for CartPole. This is a very simple game with two actions at any given point, move left or move right, with the goal of maintaining a pole vertically balanced for as long as possible. 
Over the course of training I plugged in some hyperparameters without thinking too much about them, only to find that I could not get my policy to converge to a optimal policy, even after a relatively large amount of training time (at least, relatively large compared to both me and Claude's priors).

Turns out, when using any algorithm for reinforcement learning that is based on discounted future expected rewards, the setting of gamma is very important for a game like CartPole. In particular, increasing this from 0.9 to 0.99 (effectively giving 10x further reach to future rewards) made a dramatic improvement.

Intuitively, this was easy to explain in hindsight. Cartpole is a game where the reward is given continuously as long as the terminal state is not reached. This means that with a gamma value of 0.9, the distribution of "rewards to go" has a very sharp mode at ~10, and almost all rewards to go look the same for any given action decision point during training. One can see how this could really impede efficient training. Additionally, from just watching a few simulations of CartPole, it is clear that sometimes bad moves can have consequences for future states quite distant, and a model can only learn to differentiate these actions as bad if the gamma spans the 'action persistence'. What follows was my attempts to somewhat quantitatively and mechanistically look at these concepts.

---

## Setup

<PLACEHOLDER: describe the fork methodology — from a random starting state, take both action 0 and action 1, then follow random policy for both forks and measure survival (how many steps each fork lasts). No training, no policy gradient, no neural networks. Two experiments: exp8b aggregates across forks (1000 starting states, per-fork measurement), exp8c goes deeper per starting state (200 states, 50 forks each to estimate full survival probability curves)>

---

## How Long Does an Action's Influence Persist?

<PLACEHOLDER: describe the key finding from exp8b — the mean absolute survival difference between the two fork actions peaks at around k=15-20 steps after the fork, not at k=1. The causal influence of the initial action choice builds up before it peaks, then decays. Also describe the winner consistency curve — the action that was better at k=1 remains the better action at nearly 100% of forks for the first ~15-20 steps, before dropping toward 0.5 (random chance), which means the initial action's advantage is coherent and directional, not noisy>

[Action persistence — mean absolute survival difference and winner consistency]({{ "/assets/images/action-persistence_exp8b_action_persistence.png" | relative_url }})
*<PLACEHOLDER: caption — left panel shows the mean |P(survive k | a=0) - P(survive k | a=1)| peaking around k=15-20, confirming the causal effect takes time to manifest. Right panel shows winner consistency near 1.0 for the first ~15-20 steps, meaning the better action at k=1 is consistently still the better action later — the signal is clean, not noisy>*

---

## Does This Hold Across Different Starting States?

<PLACEHOLDER: describe exp8c — rather than pooling across all starting states (which can cancel directional effects due to CartPole's left-right symmetry), this experiment estimates the full per-state survival probability curves P(survive k | action, state) using 50 forks per state across 200 starting states. Report: the distribution of peak MAD (mean ~0.38, median ~0.38 — action choice matters a lot at peak), and the distribution of peak locations (mean ~12.7, median ~13 — the peak is consistently in the 10-20 step range across states). The individual state curves shown in faint blue are consistent with the mean — this isn't driven by outliers>

![Action persistence across starting states — curves, peak heights, peak locations]({{ "/assets/images/action-persistence_exp8c_persistence_by_state.png" | relative_url }})
*<PLACEHOLDER: caption — left panel shows per-state persistence curves (faint) with mean and IQR band; the peak is consistently around k=10-15 across states. Middle panel shows the distribution of peak survival difference (~0.38 median — action choice matters substantially at peak). Right panel shows peak location distribution concentrated around 10-20 steps, confirming the delayed-peak pattern is a property of CartPole's dynamics, not a measurement artefact>*

---

## What Gamma Does to the Reward Signal

<PLACEHOLDER: 2-3 sentences bridging from action persistence to the reward signal — the previous sections showed *when* actions matter (peak at ~10-20 steps). This section shows *why gamma determines whether the training signal can differentiate those steps*. The mechanism: gamma controls the variance of the rewards-to-go (RTG) distribution, which is the only signal the actor receives.>

![RTG distribution by gamma — violin plot]({{ "/assets/images/action-persistence_exp4_rtg_distribution.png" | relative_url }})
*<PLACEHOLDER: caption — RTG distributions under a random policy, by gamma. At gamma=0.5, all RTGs cluster between 1-2 — every step looks identical to the advantage estimator. At gamma=0.99, RTGs span 1-60+, giving the actor a differentiable signal. The horizontal lines show the mean and median of each distribution. This is why low gamma fails: the advantage signal is intrinsically compressed regardless of critic quality.>*

---

## Summary

<PLACEHOLDER: 2-3 bullet points covering:
- The causal effect of a single action on survival in CartPole peaks ~10-20 steps later, not immediately — the environment's physics takes time to convert an action into an observable outcome difference
- This is measured purely from a random policy with no training — it is an intrinsic property of CartPole's dynamics
- Gamma controls whether training can exploit this: at gamma=0.9, gamma^15 ≈ 0.20 — 80% of the signal discarded at exactly the horizon where actions matter most. The RTG distribution at low gamma is so compressed that even a perfect critic produces near-zero advantages
- The combination of action persistence curves and RTG distributions tells a unified story: actions matter at k=10-20, and only high gamma preserves enough signal variance at that horizon for learning to occur>

---

## Open Questions

<PLACEHOLDER: what you'd investigate next — e.g. does the peak location of ~13 steps match the causal horizon estimate of 19 steps from the random policy episode length distribution (exp8/credit mass)? Is the peak location state-dependent in a way that correlates with the pole angle or cart velocity? Does the winner consistency result (near 1.0 for 15+ steps) mean that a very short lookahead (e.g. 15-step MC return) could be sufficient for CartPole, and if so, does that match what low-gamma training actually learns?>
