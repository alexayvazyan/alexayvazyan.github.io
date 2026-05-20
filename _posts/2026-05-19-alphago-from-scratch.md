---
layout: post
title: "AlphaGo from scratch"
date: 2026-05-19
---

# AlphaGo from scratch


---
The goal is to try and branch out from my previous RL projects (Cartpole, Pong, other Atari) and try to learn something new. Recreating a Go playing algorithm seems like it would be decently different to what I was familiar with.

Go is different in a few ways from what I've done before. It's state space is bigger. It's a multiplayer game, and I don't have a readily available teacher. In Pong, the gymnasium environment equipped me with an AI to verse who was already profficient in Pong. In Go, ill have to train a teacher and student simultaneously. 

To start with, we need to outline some specifications as to the environment we will create. We will terminate the game when a player has no legal moves, or after 100 moves have passed. Once that is done, we will tally the pieces to decide who wins. Suicide as a move will be prohibitted in order to encourage more diverse games and discourage some annoying loops.

The 100 move limit isnt great, as counting pieces is not surefire to name the winning player as the winner. Nonetheless, we will use it and keep track at how many games reach this move limit as an infererior data source metric.

We will start with something simpler, a 5x5 Go board, just to make sure our architecture and rules allow for learning to occur.

---
**Setup**

Lets go with something pretty vanilla for setup. A simple neural network with 4x256 layersxneurons to an output state corresponding to actions across the 25 squares. For the state space, we will encode the white pieces and black pieces as 2x25 vectors, as well as a bit for who's turn it is. Thus, the total environment will be summarized in a 2n^2 + 1 length vector, in the 5x5 case, length 51. 

We will use simple PPO as our gradient step method. Run a few games, collect trajectories for each game (sets of state, action, probabilities). Compute rewards via discounting the win at the end with a gamma, intuitive set high as moves far away from the end are if anything even more important in winning Go. Setup a value network which is identical to the main network, but only outputs a single logit for the expected reward given a state vector. Compare trajectory rewrads to expected rewards, and step both the value network and the actor policy in sequence, using standard PPO loss for the actor. Repeat for new trajectories.

Straightforward enough. A quick test run immediately raised two key questions which will be central to the remainder of the project 

1. How on earth am I meant to know if the policies are actually learning?
2. How can I prevent narrow tunnel vision?

For 1, this question arose out of simply having no idea how the actor network was improving over time. The value network is easy, we can just watch its loss go down over training. The actor was a new challenge. In Cartpole and Pong, I had a verifiable way to easily track progress, because I could just watch my policy get better at beating its environment. However, in this case, my AI is playing against itself, its winrate is always going to hover 50/50. 
