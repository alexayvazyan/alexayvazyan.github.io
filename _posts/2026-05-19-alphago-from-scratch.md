---
layout: post
title: "AlphaGo from scratch"
date: 2026-05-19
---

# AlphaGo from scratch

*I want to build AlphaGo from the ground up — board, self-play, MCTS, the joint policy/value head — so that I understand every piece rather than importing it. This first post is the unglamorous leg: getting self-play PPO to actually learn on a real board. The interesting part turned out not to be the Go, but a class of importance-sampling bug that quietly makes off-policy PPO blow up, and the on-policy rollout pattern that makes the whole failure mode disappear.*

---

## Why start here

AlphaGo is a stack of ideas — a board you can search, a self-play loop, Monte Carlo Tree Search guided by a network, and a single network with two heads (policy and value) over a convolutional trunk. It would be easy to wire all of that together from a tutorial and never know which piece is doing the work. I'd rather climb it one rung at a time and earn the intuition, the same way I did with the [DQN divergence](/2026/04/14/dqn-divergence-target-network.html) and [gamma](/2026/03/20/action-persistence-cartpole.html) work.

So the constraints I set for this leg were deliberately strict: a plain MLP, no MCTS, no tree, no convolutions. Just an actor-critic trained by PPO self-play on a real Go board. If I can't get *that* to climb cleanly, there's no point bolting search on top. Everything fancy is a later post.

One control choice worth flagging: I'm running `γ = 0.9` even though my own [gamma work](/2026/03/20/action-persistence-cartpole.html) argues `0.99` is the better default. I wanted to vary one thing at a time while I debugged the self-play loop, and hold the discount fixed at a value I already understood the behaviour of. It'll go up later.

---

## The board

I wrote the environment from scratch — `go_env.py`, no Go library. It's 5×5, which is small enough to train on CPU in minutes but large enough to have real tactics (captures, ko fights, life-and-death in the corners). The rules I implemented:

- **Captures**: after you place a stone, any opponent group with zero liberties is removed. Flood-fill to find groups, count liberties, delete the dead ones.
- **Suicide**: a move is illegal if, *after* resolving captures, your own group has zero liberties. (Captures resolve first, so a move that looks suicidal but kills an opponent group is legal.)
- **Ko**: if a move captured exactly one stone, the point it was captured on is banned for the opponent's next move — the standard single-stone-recapture ban.
- **No komi, stones-as-score**: the game ends at a move limit (100) or when the player to move has no legal moves, and whoever has more stones on the board wins. No territory counting, no komi compensation for Black. That last simplification turns out to matter a lot — more below.

The state handed to the network is a flat 51-vector: 25 bits for White stones, 25 for Black, and 1 bit for whose turn it is. The action space is the 25 points; I mask illegal moves to `-1e9` logits before the softmax so the policy only ever puts mass on legal plays. Reward is terminal-only — `+1` win, `−1` loss, `0` draw — discounted back along the mover's own trajectory as `R · γ^i`. The critic is a separate MLP regressing those returns; the advantage is just `return − baseline`, normalised per minibatch. No GAE, nothing clever.

---

## First architecture: off-policy buffered PPO, and why it blew up

My first instinct came from the DQN work: a replay buffer (100k transitions), a frozen actor that generates games, ε-greedy exploration on top of the policy, and PPO minibatches sampled from the buffer. It seemed natural — reuse data, decouple collection from learning.

It did not work, in two escalating ways.

**Symptom 1 — it oscillates instead of learning.** With the small network, training ran to completion but the self-play win rate never settled. Over a single run the recent-100-game White/Black split lurched between `18/80` and `88/12` and back, generation after generation, with no monotone progress. The buffer is a blender: it holds games from many past frozen generations at once, so every gradient step is fighting a mixture of opponents that no longer exists.

**Symptom 2 — with a bigger network, it diverges to NaN.** When I scaled the MLP up (the run is literally named `selfplay_big_8x128`), the importance ratio climbed to **2.5 × 10²⁸** and the weights went `NaN` inside the first ~5,000 steps. That's not noise; that's a runaway. Worth understanding exactly why, because the mechanism is subtle and it's the real content of this post.

There are three things stacked on top of each other.

**(a) The PPO clip is asymmetric, and the `A < 0` branch is unbounded.** The PPO objective per sample is

```
L = min( r·A , clip(r, 1−ε, 1+ε)·A )
```

When the advantage `A > 0` and the ratio `r` runs large, the `clip` term caps the contribution at `(1+ε)·A` — bounded, by design. But when `A < 0` and `r > 1+ε`, the two candidates are `r·A` and `(1+ε)·A`, and since `A` is negative the *more negative* one is `r·A`. The `min` picks it. So for a bad action whose ratio has blown up, the loss is `−r·A = r·|A|`, which grows **without bound in `r`**. The clip that's supposed to be a safety rail only guards one side.

**(b) ε-greedy misspecifies the importance-sampling denominator.** This is the deep bug. PPO's ratio is meant to be `π_θ(a) / b(a)`, where `b` is the policy that actually *drew* the action. But my behaviour policy was

```
b(a) = ε · uniform(legal) + (1−ε) · 1[a = argmax logits]
```

while the `log_pi_old` I stored was `log softmax(logits)[a]` — the log-prob under the *softmax*, not under `b`. Those are different distributions. When `a` is the greedy action, `b(a) ≈ 0.95` but the softmax might assign it `0.2`, so the stored denominator is far too small and every such ratio is inflated. The importance weights aren't just noisy, they're systematically wrong, and the error is correlated with how greedy the action was.

**(c) Staleness and entropy collapse turn the inflation into an explosion.** The buffer's `log_pi_old` was recorded under frozen policies from many generations back. As the live policy sharpens (entropy fell toward zero in these runs), it concentrates mass on an action `a` that some old, diffuse policy gave a tiny probability. Then

```
r = exp( log π_live(a) − log π_old(a) )   →   exp( 0 − (large negative) )   →   huge
```

and by (a) nothing penalises it. One Adam step on a `r ≈ 10²⁸` gradient is the end of the run.

The honest takeaway is that *none* of these is exotic. They're the default failure surface of bolting a replay buffer and ε-greedy onto PPO, an algorithm that is on-policy by construction. I abandoned the buffered design entirely.

<!-- FIGURE: ratio_max over training, off-policy 8x128 (NaN) vs on-policy sync10k (self-corrects). Log-y. -->

---

## The fix: on-policy rollout PPO

The replacement is the boring CartPole pattern, and it dissolves the whole class of bug:

1. Collect a fresh rollout of ~50 self-play games against the current frozen opponent.
2. Record `log_pi_old` *at collection time, from the same softmax that drew the action* — no ε-greedy, no separate behaviour snapshot.
3. Do 4 PPO epochs over that rollout.
4. **Discard the data** and recollect with the updated policy.

The key line is step 2: because the action is sampled from the policy's own masked softmax and `log_pi_old` is read off that same distribution, the importance-sampling denominator is correct *by construction*. There is no `b ≠ π` mismatch to misspecify, because `b` *is* `π`. There's no buffer, so nothing is stale beyond one rollout. And empirically I no longer need gradient clipping to stay alive.

It's not that the ratio stays pinned at 1 — it doesn't. Within a rollout's later epochs the policy has moved off the collection policy, and as entropy collapses a single sample's ratio still spikes into the hundreds or low thousands (I logged a transient `ratio_max ≈ 2,300`). The difference is that the data is one rollout old, not a hundred-thousand transitions and many generations old, so the spike is local and self-corrects within the generation instead of compounding. `ratio_max` falls back to ~1.0 on its own. Bounded-and-recovering versus unbounded-runaway is the entire distinction between this and the buffered version.

---

## The sawtooth

The frozen opponent is resynced to the current learner every 10,000 gradient steps. That cadence produces the signal I was actually after — a sawtooth in evaluation win rate:

| Generation (steps) | Trough (just after resync) | Peak within generation |
|---|---|---|
| 0–10k   | 0.44 (start) | 0.92 |
| 10k–20k | 0.50 | **1.00** |
| 20k–30k | 0.44 | 0.96 |
| 30k–40k | 0.40 | **1.00** |
| 40k–50k | 0.42 | **1.00** |
| 50k–60k | 0.50 | 1.00 → *decays* |

*Learner win rate vs the frozen opponent, 50 stochastic games per eval point, even color split.*

The mechanism is clean. Right after a resync the frozen opponent **is** the learner, so the match is even and the win rate sits near 0.5. Over the next 10k steps the learner improves against a now-fixed target and climbs toward ~1.0. The next resync promotes that improved policy to be the new opponent, the win rate collapses back to 0.5, and the climb starts again from a higher absolute level. Each tooth is one generation of genuine improvement — an Elo treadmill where beating your past self only ever resets you to even. Five clean teeth over 60k steps was the "it's actually learning" result I wanted before adding any search.

<!-- FIGURE: eval/learner_winrate vs step for selfplay_sync10k, vertical lines at each 10k resync. The headline plot. -->

One honest caveat: the final generation (50k–60k) decays from 1.0 back toward 0.5 instead of climbing. By that point training entropy has fallen to essentially zero — the policy is fully deterministic, and a deterministic policy playing a deterministic frozen copy of itself can get stuck in a single losing line with no stochasticity to escape it. That's a thread to pull on (entropy floors, or temperature at collection), not something I've resolved.

---

## A fingerprint of no-komi: the color asymmetry

There's a subtlety hiding in those troughs. When I split the eval win rate by the color the learner played, I don't see a symmetric `0.5 / 0.5`. At a trough — when the learner and the frozen opponent are near-identical — the split is closer to `1.0 / 0.0`: the learner wins almost every game as one color and loses almost every game as the other. The 0.5 aggregate is hiding a structural bias, not reflecting a fair fight.

That's the no-komi simplification biting. With no compensation for moving first, on a 25-point board the first-move advantage is large enough that, between two equal players, the color decides the game. And the interesting part: *which* color dominates flips between generations. As the policy reshapes itself each generation, the structural advantage swings from one side to the other, so `as_white_winrate` and `as_black_winrate` trade places (I logged each spanning the full `0.0`–`1.0` range across the run). It's a nice diagnostic — the color split at the trough measures the board's intrinsic bias, while the aggregate at the peak measures the skill gap the learner has opened over its frozen self. Adding komi is on the list.

---

## A control: cranking entropy kills the sawtooth

To check that the sawtooth really reflects the policy committing to strong play, I raised the entropy bonus from `0.01` to `0.1` and reran. Training entropy now holds around `1.9` (≈7 effective moves considered per position) instead of collapsing — and the sawtooth essentially disappears, with win rate hovering near 0.5 the whole run.

That's the expected result, and it's reassuring. A policy kept deliberately diffuse can't decisively beat a frozen copy of itself, because both are spreading mass over the same set of plausible moves; there's no commitment for the learner to exploit. The sawtooth amplitude *is* the policy's willingness to sharpen. So there's a real tension here — sharpening is what produces visible improvement, but it's also what eventually drove the entropy-zero stall in the last generation above. Finding the right amount of exploration pressure is the open knob.

---

## What's next

This was the foundation. The roadmap toward an actual AlphaGo, in order:

1. **Convolutions.** Swap the flat-51 MLP for a small CNN / ResNet over the 5×5 board planes. The board has translation structure the MLP is throwing away.
2. **MCTS.** Add tree search at action-selection time, and train the policy toward the search's visit-count distribution rather than toward raw returns — the AlphaZero target. This is the piece that actually makes it "AlphaGo" rather than "PPO on Go."
3. **A joint head.** Collapse the separate actor and critic into one trunk with policy and value heads, the way the real architecture shares representation.

And one settled lesson I'm carrying forward: don't reintroduce a long-lived replay buffer or ε-greedy into a PPO loop. The on-policy rollout pattern isn't a workaround, it's the correct shape for the algorithm, and it's the difference between a clean sawtooth and a `NaN`.

---

*Code lives under [rl-testing/alphago_5x5](https://github.com/alexayvazyan/projects) in my projects repo — `go_env.py` (the board), `train.py` (the buffered version that blew up), `train_selfplay.py` (the on-policy version that works), and `web_play.py` if you want to click against a checkpoint. This page was drafted by Claude from my notes and run logs, then edited by me. Figures still to be generated from the TensorBoard runs.*
