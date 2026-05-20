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

The first way to solve this came to mind as simply freezing the other teacher policy, only resyncing it occasionally, say every 5000 gradient steps. Doing this should ensure we get a sawtooth like pattern, smooth learning with learner winrate increasing from 50% to 100% before resetting to 50%. This worked well in practice, however, it still doesnt really solve the question of how strong a model is generally. Rather, it just lets us see model learning is occuring. One easy test was to see if it could beat me, which it did easily, but I am far from a good Go player.

After thinking about it for some time, it really seems like the only way to truly judge a models ability is just to have it play with opponents. Ideally, after many different training pipelines are executed, each fully trained checkpoint gets saved as a player, and each plays all other models trained. We can then construct an ELO based system scoring each model with its winrate against other models, effectively organising tournaments of AIs. There is no guarantee that a model that beats another model is "better" than it, as who knows, they might just be getting stuck on a single trajectory. But it does provide evidence - and as we collect more evidence with more and more 'players', we can hopefully become more confident that the model that wins the most is the "strongest".

As for 2, its super easy to imagine that an AI training against itself can quickly converge onto a singular set game that it just repeatedly plays against itself, learning nothing. We need a way in PPO to encourage exploration. There are a few ways I know how to do this. We can copy what we do in Pong and Cartpole, and have our policy take random steps with some probability, with this probability decreasing over training. Alternatively, we can try encouraging having more "candidate moves" directly in the loss function. Both seem worth exploring, but I went with the latter method initially because it appealed more to my intuition. When I was learning chess, I always learnt that after theory, a very standard approach in tactical situations was to come up with several candidate moves, and evaluate each before choosing the best (you may be able to notice foreshadowing for MCTS :) ). It seems ideal that our model never centres on exactly one move, so we can take the total action logits and add a term that punishes low entropy to the loss function. We can play around with this as much as we want, even capping the number of candidate moves encouraged to 5, by only summing the top 5 contributors in the action logits. It becomes clear that there is a boatload of exploration to do here, and the only way to know what works best is to test it! 
