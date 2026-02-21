---
layout: post
title: "Can we see features in a TFT placement prediction transformer?"
date: 2026-02-19
---

# Can we see features in a TFT placement prediction transformer?

*What does a tiny model actually learn about Teamfight Tactics — compositional rules, or statistical shortcuts?*

---

## Why This Project

- Data is easily available via Riot Developer APIs
- It's a domain I have good intuition in, which makes feature identification significantly easier
- It's fun

Teamfight Tacticts (TFT) is a game I played a lot in university. Very fun, and conceptually similar to chess, just more degrees of freedom and 8 players to one match. Interpretability is a lot easier with the lights on. The leading question that made me want to investigate this project was whether the model would learn synergistic 'trait' connections between different champions fielded, and if so, where would this be visible from an outsider looking in?

## The Model

**Task**: Predict TFT placement (1st–8th) from a board composition (up to 15 champions + 5 emblems).

More data was available, such as items for each champion, as well as the traits themselves. However, it mostly interested me if the model could learn the trait information without seeing it, given its all embedded in the champion information. The model chosen was intended to be relatively simple and low dimensional.

**Architecture**:
- **Unified token sequence**: Champions and emblems (items that add a trait) share a single vocabulary (124 tokens: 1 pad + 101 champions + 22 emblems) and attend to each other in the same sequence
- We take this 124 dimensional space for champions and emblems and project it into 8d vectors, one for each token.
- We run each token through a singular attention head to let the model discover champion 'synergies'.
- We then pool all the tokens together into a single vector intending to encapsulate the entire 'board state'.
- We then do a 32 neuron FFN over the board state output which directly unembeds to the logits, which then get softmaxed to probabilities of placing 1-8.
- Loss: CrossEntropy + MAE (ordinal-aware)
- **Total: 2,424 parameters**

**Data**: ~24,344 boards from Challenger players (OCE server, Set 16), scraped via Riot API. 50/50 train/test split.

**Training**: Up to 1000 epochs with early stopping (patience 50), Adam with weight decay 1e-4, ReduceLROnPlateau.

Given we have a totally uniform distribution of placements (each sample comes in a collection of 8 from the same match, with each match having exactly 1 placement from 1 to 8), we can benchmark against just guessing an average placement of 4.5 for each sample. This would have a MAE of 2.

As a first pass, lets try train and see what happens.

![Training curve showing train and test MAE converging around 1.48](/assets/images/training_curve_mean_pool.png)

MAE of 1.55! Not bad, the model is clearly working, and performing decently better than the benchmark. Lets investigate.
My prior intuition tells me that if I was training myself to look at this data, the first things I would take note of are:
- Average champion strength (some champions are just stronger and have higher average placement than others)
- Number of champions (the more champions you have on your final board (i.e. the more number of tokens passed), the more likely you made it later into the game)
Lets see if we can clearly see the model picking these up.


## Hypothesis 1: Champion Strength as a Feature

There are a few places where this feature might present itself, but most obvious would in the embedding of champions. One can expect that one dimension would be reserved for average placement, with dot products of two strong champions / two weak champions positive and likewise expecting that strong and weak champions point in opposite directions along this axis.

![Cosine similarity matrix showing strong champions clustered together](/assets/images/cosine_similarity_by_strength.png)
*Champions sorted by average placement. Strong champions (left/top) have high pairwise cosine similarity. Weak champions (right/bottom) are scattered.*

A simple heatmap lets us see this clearly, especially after sorting by average placement. However, the result is not exactly what I was expecting, though there is definitely something there. We can see strong champions very clustered and similar in the direction they point, but the relationship to weaker champions seems a lot less interpretable.
Numerically,
- **PC1 correlates with champion strength at r = 0.55–0.61**
- Strong champions (avg placement < 3.9) cluster tightly — mean pairwise cosine similarity 0.56–0.67
- Weak champions are scattered — mean pairwise cosine similarity 0.02–0.15

Intuitively, the model has learned a binary feature, rather than a scale. Its much more informative to ask whether a board has strong champions, rather than how strong on average is the board. 
Even though this was not my expectation entirely, it's not too hard to backsplain. Like more tokens, what the model is likely learning isn't that certain champions are stronger, but rather that they are harder to get to, and thus that they implicate you make it later into the game.

![Strength analysis showing binary clustering of champion embeddings](/assets/images/strength_analysis.png)
*The binary nature of the strength feature: strong champions form a tight cluster in embedding space, while weak champions scatter across multiple directions.*


## Hypothesis 2: Number of tokens presented is a dominant feature

Intuitively, this should obviously be a large feature. Where does our model learn it?

I guess before even asking these questions, we should ask, *does* the actually model learn it? When we look at test samples and group them by the number of champion tokens, does our model do a good job in aggregate in predicting these buckets?

![Mean pooling fit](/assets/images/pooling_mean_only.png)


We see that while we do okay at predicting the central cluster, we predict poorly in the tails. This is a pretty grim result, its not hard to predict the central cluser well even if you just use the benchmark with a couple adjustments. The tails is where this feature should really be activating to make a difference.

An early architectural choice actually turns out to have inhibited learning this feature. If we spot check some post pooled vectors, they actually look surprisingly similar to champions by themselves, even with attention before hand. 

This led to a fun little investigation where it turns out that a transformer (softmaxed attention + FFN) -> mean pooling cant count tokens! I'll do a quick writeup sometime to sumarize the findings there.

If we take the sum instead after the transformer, we allow the size of the vector would be able to easily represent how many champions are on a board. Lets retrain.

![Training curve showing train and test MAE converging around 1.48](/assets/images/training_curve.png)

MAE 1.55 -> 1.45.

![Sum vs Mean pooling fit](/assets/images/pooling_sum_vs_mean.png)

A much nicer fit. Lets continue with this.

Great. Our model is now correctly handling the two most important features directly. 

## Hypothesis 3: Prismatic Traits as Sparse Features

The MLP head has 32 hidden neurons, each contributing to the final placement prediction. Most encode diffuse features, but one stands out.

**Neuron 13** pushes the prediction toward 1st place, and its activation pattern is remarkably clean:

| Board condition | N13 activation |
|----------------|---------------|
| 8 BG + 2 BG emblems | +0.98 |
| 8 Yordle + 2 Yordle emblems | +0.95 |
| 8 Ionia + 2 Ionia emblems | +1.00 |
| Control board 1 + 2 BG emblems | **0.00** |
| Control board 2 + 2 BG emblems | **0.00** |

N13 fires when matching emblems appear on trait-heavy boards, and *does not fire* when the same emblems appear on control boards. This is the most interpretable finding in the model: a single neuron has learned to detect "emblems on a synergy-heavy board" — a genuine contextual feature, not just emblem presence.

Other MLP neurons show more trait-specific patterns:
- **N8** detects BG emblem presence (fires regardless of board context)
- **N26** is Noxus-specific (suppressed by Noxus emblems on Noxus boards)
- **N16** and **N15** are suppressed by Yordle and Ionia emblems respectively on matching boards

The MLP uses a mix of generic and trait-specific neurons. The generic ones (N13) capture "emblems help trait boards." The trait-specific ones respond to particular emblem types but haven't fully learned that the emblem *must* match the board's trait.

*The code for this project is on [GitHub](https://github.com/alexayvazyan/projects).*
