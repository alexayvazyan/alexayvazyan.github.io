---
layout: default
---

# Hey, I'm Alex

I was a Quant Trader for almost 3 years. I've always had a strong interest in understanding why things happen, both in the realm of the physical but particularly in the realm of the mind. Naturally, I think this has fueled my predisposition towards machine learning. I've been fortunate that this skillset has been highly relevant at my work as a quant trader, but increasingly the world provides me signal that my skillset can directly be harnessed to contribute to models that will change the future, for better or for worse. 

As well as being an arguable moral prerogative, I also find the work in the field of "Artificial Intelligence" to be interesting, and these pages document my learning process and thoughts as I expand my horizons in this domain.


[GitHub](https://github.com/alexayvazyan) | [Site](https://alexayvazyan.github.io)

---

## Learning

- [Can we see features in a TFT placement prediction transformer?](/2026/02/19/tft-mechanistic-interpretability.html) — Intro project to just play around with some rough ideas of mech interp in a domain where I have a good intuition.
- [Softmax attention with mean pooling can it count](/2026/02/20/mean-pooling-cant-count.html) — A surprisingly insightful deep dive into whether soft max + mean pooling can actually count number of tokens.
- [How does a model's prismatic detection circuit evolve under signal pressure?](/2026/02/27/tft-oversampling-prismatic.html) — Oversampling a rare trait from 1x to 100x and watching the circuit sharpen.
- [Hyperparameter tuning in Cartpole and action persistence](/2026/03/20/action-persistence-cartpole.html) — Why gamma=0.99 outperforms gamma=0.9, and measuring how long a single action's causal influence persists.
- [DQN divergence without a frozen target network](/2026/04/14/dqn-divergence-target-network.html) — Three weights and two states is enough to see bootstrap-feedback divergence, and drawing the set of diverging initialisations turns out to have a clean geometric shape.

## Research

- [The ICL Heptagon — research thread](/icl-heptagon.html) — Synthesis page tying together the cross-model sweep, phase analysis, deduction-failure work, ring causal patching (PC1-PC2 carry 97% of donor-leak at onset L=16), and the closed +2 chain ceiling.
- [ICL can induce a cyclic manifold — but only in some models](/2026/04/22/icl-induced-cyclic-manifold-cross-model.html) — 13 base/IT runs across Gemma, Llama, Pythia, Qwen, GPT-2; ring forms in 4 of 7 base models; Qwen 2.5 3B solves at 100% with the cycle scrambled — task-solving doesn't imply ring geometry.
- [Linear ablation with rotating representations of refusal](/2026/04/09/refusal-direction-ablation.html) — Replicating refusal direction ablation on Gemma 4 and discovering that the refusal direction isn't stable across layers.
- [Geometry of speed and size across multimodal stimulus in Gemma 4 31B](/2026/04/29/geometry-speed-size-gemma4-31b.html) — Clean curved manifolds for physical scalars, causally steerable, but compositional use (computing momentum) collapses to chance — likely a single-pass bandwidth limit.
- [Dynamic scaling of hyperparameters during RL training of Pong](/2026/04/29/dynamic-hyperparameter-scaling-pong.html) — Reading convergence/divergence signals online and adjusting target-sync, learning rate, epsilon, and gamma on the fly instead of committing to a fixed schedule.

## Musings

- [Arbitrary thoughts](/arbitrary-thoughts.html) — A running log of whatever my brain chews on while waiting to fall asleep.
- [Representations on an entropy axis](/representations-entropy-tree.html) — A framing for where human concepts, LLM features, and mechanistic interpretability sit on the same compression tree.
