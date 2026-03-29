---
layout: post
title: "Hyperparameter tuning in Cartpole and Action Persistence"
date: 2026-03-20
---

## Background

As part of learning RL, I was implementing PPO from scratch in the OpenAI gymnasium, training on CartPole. This is a very simple game with two actions at any given point, move left or move right, with the goal of maintaining a pole vertically balanced for as long as possible.
Over the course of training I plugged in some hyperparameters without thinking too much about them, only to find that I could not get my policy to converge to an optimal policy, even after a relatively large amount of training time (at least, relatively large compared to both me and Claude's priors).

Turns out, when using any algorithm for reinforcement learning that is based on discounted future expected rewards, the setting of gamma is very important for a game like CartPole. In particular, increasing this from 0.9 to 0.99 (effectively giving 10x further reach to future rewards) made a dramatic improvement.

Intuitively, this was easy to explain in hindsight. Cartpole is a game where the reward is given continuously as long as the terminal state is not reached. This means that with a gamma value of 0.9, the distribution of rewards to go (RTG) has a very sharp mode at ~10, and almost all rewards to go look the same for any given action decision point during training. One can see how this could really impede efficient training. Additionally, from just watching a few simulations of CartPole, it is clear that sometimes bad moves can have consequences for future states quite distant, and a model can only learn to differentiate these actions as bad if the gamma spans the 'action persistence'. What follows was my attempts to somewhat quantitatively and mechanistically look at these concepts.

---

## What Gamma Does to the Reward Signal

A sensible starting point might be to just look at the RTG distribution over different gammas. The hypothesis is that RTG are compressed to 10 during training, which limits advantage calculations and hence makes gradients rather small.

![RTG distribution by gamma — violin plot](/assets/images/action-persistence_exp4_rtg_distribution.png)
*RTG distributions under a random policy, by gamma. At gamma=0.5, all RTGs cluster between 1-2 — every step looks identical to the advantage estimator. At gamma=0.99, RTGs span 1-60+, giving the actor a differentiable signal. The horizontal lines show the mean and median of each distribution.*

---

## Action persistence

Next, we can take a deeper dive at this idea of a causal horizon between actions and outcomes being long in a game like CartPole. How exactly does one go about measuring this though? One potential way is to imagine taking a random policy and random state and then measuring the difference in terminal scores for each of the two actions taken. We can identify two limiting cases where either the state is already in a doomed position, where the fork will have little significance on survival. On the other side, we can imagine the state which is perfectly symmetrical in the starting position, and again the fork should show little difference. The vast majority of states though (everything in between) should produce a discernable impact on survival when averaged over many trials.

We take N random states. At each state we take T trials. For each trial, we count the steps survived by each fork for each of the two actions possible. We then combine each to create a survival curve for going left, and a survival curve for going right at each state. Finally, we subtract these two survival curves and take the absolute value to form a blue line in our graph.

Note that there will be a bit of noise (even in a symetrical starting point, because we will expect some non zero diff survival function) which will go down as we do more forks per state. 

![Action persistence — mean absolute survival difference and winner consistency](/assets/images/action-persistence_exp8c_persistence_by_state.png)
*If we are to average across all states, we see that an action will peak in its survival probability effect in around ~12 steps after the action was taken*

Indeed, we observe that up until around ~5 steps taken, the vast majority of games show 0 survival difference (too early to make a difference), with this also converging to 0 survival difference if we look far enough into the future (both games are dead). 12 steps into the future is the average distance for peak action impact on survival probability.


---

## Summary

- High gamma is extremely important in training an RL model using discounted future state quality to learn CartPole.
- This is likely a result of the very unbalanced nature of the distributions of RTG when gamma < 0.9.
- This unbalanced distribution is likely a product of the fact that in CartPole specifically, actions have a peak causal horizon of 12 steps into the future.

To be honest, the causal horizon analysis seems difficult to generalize to any other problem that is not CartPole. For that reason I find that the RTG distribution finding is probably more relevant to future work. In particular, it would be good to analyze this distribution among other games and contrast if training also greatly improves from balancing this.

---

## Open Questions

RTG = Q - V for PPO (the model used in this analysis)
- Will using GRPO which takes the average of the group instead of V yield a much more balanced RTG distribution and hence much faster convergence to optimal policy?
