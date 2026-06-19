---
layout: post
title: "What exactly is MCTS?"
date: 2026-06-19
---

# What exactly is MCTS?

I'm writing this page partly to consolidate my own learning, but also because I feel like I struggled to find subject matter online when learning about this topic that managed to properly motivate MCTS. I think generally motivating concepts is very important to learning them, so I hope I'll be able to do that here.

Let's start with the situation. You just finished training your first attempt at a Chess AI bot. It's good, it can beat you, but you're no grandmaster. You want to make it better. Currently, you have simply trained a PPO actor and critic network by having the AI play itself repeatedly, and sample moves to train on from a replay buffer. It simply picks the best move as it sees fit given the state it's in and its actor policy network. It plays virtually instantly, there is no "thinking". It's akin to a human playing bullet off pure learned intuition from previous experience.

But classical chess games allow you to trade time to "think", come up with more accurate predictions of what the best move in the state is by doing some computation. A human typically acts by first coming up with several candidate moves, which look intuitively good. Then, simulate up to a certain depth each of these moves in a variety of different possible outcomes, and evaluate the positions, ultimately choosing the move with the best expected position.

We can naively implement this quite simply with our AI. At each state, we can select up to, let's say, 5 candidate moves, based on our top 5 logits for actions from our actor network. We can then sample from our actor policy X games (e.g. 15) with a max depth of 10 moves from our state. We then collect values using our critic network at these terminal nodes, and combine them in any way we want, let's say average.

This seems like a pretty good baseline! Let's see how it goes. We observe that our evaluation time skyrockets, each move before was almost instant but now takes a couple seconds. But we also see our AI get better! It's now able to beat the other AI 80% of the time. What's important to realize is that on top of this blueprint, there are so many choices and optimizations we can make. All MCTS is is a subset of some of these optimizations which are particularly good for training and inference!

There are so many different avenues to trying to make this algorithm better, maybe we should make a list of ideas?

- Current process is very slow due to recomputing moves at every instant and then discarding them after a move, only to calculate moves again. There is certainly a lot of redundant work here. Perhaps we can build a tree and add nodes to it over the course of the game? Trading memory for compute is a classic angle to improve our performance. We can document knowledge in the tree, like the number of times we have visited a node, the average payoff of being in that node, and any other information we want to preserve.

- We are currently averaging the critic win probabilities at each terminal node. Does that make sense? If a move is on average good, but can be punished catastrophically, should we make it if we assume we are playing a competent opponent? Perhaps we should look at the max of the min win probability?

- We sample just based on our actor policy, but looking at this reveals that our actor is quite sharp. That is, it has very little incentive to explore moves and much prefers based on its training to take the same moves repeatedly in similar states. Perhaps we can make our simulations take some off policy moves every now and then?

- We rollout equally from the top 5 moves at every state, but realistically a human is much more adaptive, both in depth and breadth. A human might see 2 or 3 good moves that need to be deeply calculated 20 moves in advance in an endgame, versus another move which only needs 2-3 depth to evaluate. Also, humans also somewhat weigh variance when assessing which moves to assess. A move might not look intuitively the best, but its "explosiveness" warrants an investigation. How can we try to build in these tendencies into our AI for selection and depth evaluation?

- Can we use this same process to train a better policy network?

MCTS is simply an answer to all of these. We preserve a tree between rollouts. We select which nodes to simulate by using a criterion known as PUCT, a simple formula that takes the quality of an action and additionally adds a term to punish nodes with high simulations already done through them, encouraging exploration. Instead of running to depth 16, we simply just let the selection criterion do the work to see where our simulations end up, terminating whenever we expand a new node. Then, we just throw a ton of simulations and let them go as they please, like watching balls go down the Galton board, weighted by the actor's priors, where the pegs are getting constantly reshaped based on what states we visit how often.

The second dot point answers itself empirically, where we see that most often taking a simple average does a decent job, as we are already weighting by how good a play is based on how many times we simulate down that path.

Now for the last dot point, how can we use this to help train better actor priors? Well, we just run the MCTS loop whenever we want during training, and run gradients to push our model towards producing the MCTS distribution of actions. That's it. We can run 5000 steps of PPO, then 5000 steps of MCTS, then 1000 steps of PPO, or we could just do the whole thing through MCTS.

As an aside, the MC in MCTS (Monte Carlo) doesn't actually apply to our algorithm here. It seems to be an artefact of the algorithm typically evaluating nodes through random rollout until termination. We have the luxury of having a value network trained through PPO, so we can use that deterministically. The responsibility falls on us to make sure that the value network is actually good, as we trade stochastic noise in the case of random rollouts for a potentially biased critic network.
