---
layout: post
title: "Dynamic Scaling of Hyperparameters During RL Training of Pong"
date: 2026-04-29
---

# Dynamic Scaling of Hyperparameters During RL Training of Pong

*Most RL pipelines pick a single value of gamma, epsilon decay, target sync rate, etc., at the start of training and never touch them again. But the regime training is in changes dramatically as the agent learns: a freshly initialized network and a near-converged one are not the same problem, and shouldn't be treated as such. This post is a sketch of a project I'm running on Pong DQN, where the hyperparameters reshape themselves online based on signals coming out of training itself — convergence vs divergence, advantage scale, episode length distribution.*

---

## Background

I've spent the last few weeks training DQN on Pong on a 5090. The path was: working through Tsitsiklis-style toy divergence cases ([blog post](/2026/04/14/dqn-divergence-target-network.html)), the gamma/action-persistence work in CartPole ([blog post](/2026/03/20/action-persistence-cartpole.html)), and now the actual Atari problem at compute scale where the choices start to matter for wall-clock as well as correctness.

The thing that kept nagging me through all of this: every hyperparameter I was sweeping had a "right" value that was clearly different at step 1k vs step 1M. A few examples I ran into:

- **Target network sync period.** Early on, the online network's Q-values are nonsense, so a stale target is also nonsense — copying more often does no harm. Late in training, the target needs to lag enough to stabilize the bootstrap, but lag too much and the online network outruns the policy improvement signal.
- **Epsilon decay shape.** A pure linear decay either burns budget on random play after the agent has learned the obvious moves, or starves it of exploration in regions of state space it only reaches once it's already decent.
- **Learning rate.** With a fresh replay buffer dominated by random transitions, large updates are wasted; once the buffer reflects a competent policy, large updates can outrun the target's stability margin and trip the divergence regime I demoed in the toy setup.
- **Gamma.** I argued in the CartPole post that high gamma is essentially mandatory because of RTG variance compression. But a high gamma with garbage Q-values gives you garbage bootstrap targets across long horizons. There's an argument for *ramping* gamma during training — start short-horizon while the value head is unreliable, lengthen as it stabilizes.

The common shape: each of these has an early-training optimum and a late-training optimum, and they're in tension. Picking one fixed value is picking a compromise. The hypothesis driving this project is that you can do meaningfully better by reading the state of training and adjusting on the fly.

---

## What "the state of training" actually means

The hard part isn't deciding what to change — it's deciding what to read. You need a small set of online signals that are (a) cheap to compute, (b) reliably informative about which regime you're in, and (c) hard to fool by the agent's own pathological behavior. Some candidates I'm experimenting with:

**Q-value magnitude trajectory.** The most direct divergence detector. If `mean |Q|` starts growing super-linearly with training steps, you are in the bootstrap-feedback regime described in the [DQN divergence post](/2026/04/14/dqn-divergence-target-network.html). The fix is unambiguous: longer target sync interval, smaller learning rate, or both. Conversely, if Q-magnitudes are stable and the loss is not going down, you are probably in a flat-gradient regime and can afford to be more aggressive — shorter sync, larger lr.

**TD error distribution.** I track median and 95th percentile TD error separately. A widening gap between them — large tail, small body — usually means the replay buffer has acquired a few outlier transitions (e.g., the first time the agent scores) that the network can't yet fit. That's a signal *not* to lengthen target sync, because the lag is currently helpful; the outliers will get absorbed.

**Episode length / return.** Pong has the convenient property that episode length is a clean proxy for skill: random play loses fast, competent play extends rallies. The slope of moving-average episode length over recent windows is a coarse but reliable progress signal. Plateau detection here can trigger exploration boosts (raise epsilon, anneal slower).

**Advantage variance.** Borrowing directly from the CartPole work: if advantages collapse toward zero, the policy gradient has nothing to push against. In a DQN setting the analogue is the spread of `Q(s,a) − Q(s,a')` across the action set at the states the agent actually visits. If this spread compresses, the policy is approaching greedy lock-in on some action — possibly the right one, possibly a degenerate one. Either way, that's a signal to increase epsilon or sharpen the target by reducing gamma temporarily.

**Online-target drift.** The L2 distance between online and target weights, normalized. This is the direct measurement of what the sync period is *for*. If drift is small and TD error is large, syncing more often is free. If drift is large and TD error is small, you are in danger of oscillation and should sync less often.

---

## The control loop

The structure I'm prototyping is deliberately boring: a thin controller that runs every N environment steps, reads the signals above, and adjusts hyperparameters on a multiplicative schedule. Multiplicative because all the quantities I care about (lr, sync period, epsilon) are scale-free — a 20% nudge means the same thing whether I'm at lr=1e-3 or lr=1e-5.

Pseudocode:

```
every N steps:
    s = collect_signals()
    if s.q_growth > threshold:
        sync_period *= 1.2
        lr *= 0.8
    elif s.q_growth < -threshold and s.td_error_decreasing:
        sync_period *= 0.9
        lr *= 1.05
    if s.episode_length_plateau and s.advantage_spread < threshold:
        epsilon = max(epsilon, epsilon_floor * 1.5)
    if s.advantage_spread > healthy and s.episode_length_growing:
        epsilon *= 0.95
```

The controller is intentionally *not* aggressive. The idea is to nudge, not jump — large discrete changes to lr or sync period are exactly the kind of perturbation that can knock training out of a stable regime. I want the controller to be roughly invisible during healthy training and only assert itself when one of the signals crosses a threshold.

The single biggest worry is feedback loops between the controller and the agent. If raising epsilon causes the advantage spread to shrink (because random actions are now polluting the data), and shrinking advantage spread causes the controller to raise epsilon further, you have a runaway. The mitigation I'm planning is a per-knob cooldown — once a knob moves, it can't move again for K steps, so the system gets a chance to equilibrate before the next decision.

---

## What I expect to find

A few specific hypotheses I'd like to either confirm or kill cleanly:

1. **Adaptive sync period dominates fixed sync period.** Specifically, that the optimal *fixed* sync period is a compromise between the early- and late-training optima, and an adaptive controller beats both endpoints of the sweep.
2. **Adaptive epsilon helps more in Pong than in CartPole.** Pong has obvious early skill plateaus (just learning to track the ball) where exploration is wasted, then later plateaus (learning to angle the paddle) where it's essential. CartPole's reward landscape is much smoother.
3. **Gamma ramping is real but small.** I expect a measurable but modest improvement from ramping gamma from ~0.95 to ~0.99 over the first half of training, vs starting at 0.99. The CartPole work showed gamma=0.99 is close to optimal at convergence, but the early-training argument is that bootstrap targets are unreliable, so reaching deep into the future amplifies noise.
4. **The controller will discover something I didn't program for.** This is the speculative one. With enough signals being read, there should be some interaction the hand-tuned schedule misses — e.g., a brief period where decreasing gamma temporarily lets the value head re-stabilize after a buffer shift.

The thing that would falsify the whole project is finding that a well-tuned fixed schedule (cosine lr decay, exponential epsilon decay, fixed sync period) is within noise of the controller. That would suggest the regime structure I'm assuming is mostly imagined, and the real lever is just total compute. I don't think this will be the case, but it's the null I'm running against.

---

## Why this project, why now

There's a more general thread here that I want to develop in subsequent work: a lot of the standard RL recipe is artifacts of a regime where compute was the binding constraint, so people picked schedules that were "good enough" and moved on. With more compute per second, the binding constraint shifts to *data efficiency* and *training stability*, and the value of online introspection rises sharply. The same is true of LLM training, but RL is the cleanest setting to study it in because the regime shifts are sharp and the signals are well-defined. If this works on Pong, the next target is something with sparser reward (Montezuma's Revenge, or a procedurally generated environment) where the regime structure should be even more pronounced.

---

## Open questions

- Is there a principled way to derive controller thresholds from the geometry of the loss surface, instead of hand-tuning them per game?
- How much of this carries over to PPO/GRPO? The advantage-spread signal does directly; the Q-divergence signal doesn't have an obvious analogue.
- Could the controller itself be learned? A meta-RL setup where the controller's policy is optimized over many short training runs. Probably premature, but the scaffolding is the same.
