---
layout: post
title: "Hyperparameter tuning in Cartpole and Action Persistence"
date: 2026-03-20
---

## Background

As part of learning RL, I was playing around in the OpenAI gymnasium trying to implement a training algorithm for CartPole. This is a very simple game with two actions at any given point, move left or move right, with the goal of maintaining a pole vertically balanced for as long as possible. 
Over the course of training I plugged in some hyperparameters without thinking too much about them, only to find that I could not get my policy to converge to a optimal policy, even after a relatively large amount of training time (at least, relatively large compared to both me and Claude's priors).

Turns out, when using any algorithm for reinforcement learning that is based on discounted future expected rewards, the setting of gamma is very important for a game like CartPole. In particular, increasing this from 0.9 to 0.99 (effectively giving 10x further reach to future rewards) made a dramatic improvement.

Intuitively, this was easy to explain in hindsight. Cartpole is a game where the reward is given continuously as long as the terminal state is not reached. This means that with a gamma value of 0.9, the distribution of rewards to go (RTG) has a very sharp mode at ~10, and almost all rewards to go look the same for any given action decision point during training. One can see how this could really impede efficient training. Additionally, from just watching a few simulations of CartPole, it is clear that sometimes bad moves can have consequences for future states quite distant, and a model can only learn to differentiate these actions as bad if the gamma spans the 'action persistence'. What follows was my attempts to somewhat quantitatively and mechanistically look at these concepts.

---

## What Gamma Does to the Reward Signal

A sensible starting point might be to just look at the RTG distribution over different gammas. The hypothesis is that RTG are compressed to 10 during training, which limits advantage calculations and hence makes gradients rather small.

![RTG distribution by gamma — violin plot](/assets/images/action-persistence_exp4_rtg_distribution.png)
*RTG distributions under a random policy, by gamma. At gamma=0.5, all RTGs cluster between 1-2 — every step looks identical to the advantage estimator. At gamma=0.99, RTGs span 1-60+, giving the actor a differentiable signal. The horizontal lines show the mean and median of each distribution.*

---

## Action persistence

Next, we can take a deeper dive at this idea of a causal horizon between actions and outcomes being long in a game like CartPole. How exactly does one go about measuring this though? One potential way is to imagine taking a random policy and random state and then measuring the difference in terminal scores for each of the two actions taken. We can identify two limiting cases where either the state is already in a doomed position, where the fork will have little significance on survival. On the other side, we can imagine the state which is perfectly symetrical in the starting position, and again the fork should show little difference. The vast majority of states though (everything in between) should produce a discernable impact on survival when averaged over many trials.

![Action persistence — mean absolute survival difference and winner consistency](/assets/images/action-persistence_exp8b_action_persistence.png)
*<PLACEHOLDER: caption — left panel shows the mean |P(survive k | a=0) - P(survive k | a=1)| peaking around k=15-20, confirming the causal effect takes time to manifest. Right panel shows winner consistency near 1.0 for the first ~15-20 steps, meaning the better action at k=1 is consistently still the better action later — the signal is clean, not noisy>*

Indeed, we observe that the average causal horizon is around 15-20 steps, with the average probability of survival



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
