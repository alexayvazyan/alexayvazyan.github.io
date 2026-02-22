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

We can now begin looking to see if our model has learnt 'trait' features. To begin with, one of the most obvious cases that would stand out to a human would be 'prismatic traits', a rare occurance of a constellation of champions/emblems that normally guarantees a first placement.

  ┌────────────┬──────────┬────────┬──────┬────────┬───────────┬────────────┐                                                                    
  │   Trait    │ # boards │ Actual │ Pred │ No emb │ Wrong emb │ Wrong type │
  ├────────────┼──────────┼────────┼──────┼────────┼───────────┼────────────┤                                                                    
  │ Bilgewater │ 23       │ 1.09   │ 1.55 │ 3.93   │ 5.21      │ Noxus      │                              
  ├────────────┼──────────┼────────┼──────┼────────┼───────────┼────────────┤                                                                    
  │ Noxus      │ 4        │ 1.00   │ 2.27 │ 2.90   │ 2.12      │ Bilgewater │                              
  ├────────────┼──────────┼────────┼──────┼────────┼───────────┼────────────┤
  │ Ionia      │ 8        │ 1.00   │ 2.70 │ 2.93   │ 3.23      │ Bilgewater │
  ├────────────┼──────────┼────────┼──────┼────────┼───────────┼────────────┤
  │ Yordle     │ 12       │ 1.00   │ 2.28 │ 2.90   │ 3.48      │ Noxus      │
  ├────────────┼──────────┼────────┼──────┼────────┼───────────┼────────────┤
  │ Shurima    │ 6        │ 1.33   │ 2.27 │ 2.27   │ 2.27      │ Bilgewater │
  └────────────┴──────────┴────────┴──────┴────────┴───────────┴────────────┘

Here is a summary of the prismatic trait boards present in the test set, and what is predicted of them. As can be seen, these are incredibly sparse. Of the ~12,000 boards in the test set, the most represented prismatic board is that of 10 Bilgewater, with only 23 samples (0.2%).
We can see some signs of life in these predictions, they are mostly skewed towards first place when compared to the average 10 unit board placement (2.9), and they generally get worse when you take out the emblems / replace them with another trait's emblems, with the exception of replacing Noxus.
The reality is probably that these features are just too sparse to be worth taking up an entire dedicated monosemantic neuron. 

If we actually look at the neurons just before the logits, we can see some interesting findings. In particular, #13 seems to capture a lot of what we are looking for.
  ┌───────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
  │ Placement │  1st  │  2nd  │  3rd  │  4th  │  5th  │  6th  │  7th  │  8th  │
  ├───────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
  │ Weight    │ +0.34 │ -0.21 │ -0.15 │ -0.11 │ -0.04 │ +0.02 │ +0.09 │ +0.08 │
  └───────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
N13 points mostly to 1st place, and away from 2nd or third, pretty flat for everything else. Looking at its activations, its pretty clearly encoding something about high trait density.
  ┌───────────────────────┬────────────────┬────────┬───────┐                                                                                    
  │         Board         │ + Matching Emb │ No Emb │ Delta │                                                                                    
  ├───────────────────────┼────────────────┼────────┼───────┤                                              
  │ 8 Bilgewater + 2 fill │ 1.37           │ 0.46   │ +0.91 │
  ├───────────────────────┼────────────────┼────────┼───────┤
  │ 8 Yordle + 2 fill     │ 1.84           │ 0.95   │ +0.88 │
  ├───────────────────────┼────────────────┼────────┼───────┤
  │ 8 Ionia + 2 fill      │ 0.89           │ 0.00   │ +0.89 │
  ├───────────────────────┼────────────────┼────────┼───────┤
  │ 8 Noxus + 2 fill      │ 0.00           │ 0.00   │ +0.00 │
  ├───────────────────────┼────────────────┼────────┼───────┤
  │ 8 Void + 2 fill       │ 1.22           │ 0.63   │ +0.59 │
  ├───────────────────────┼────────────────┼────────┼───────┤
  │ Control 1 (mixed)     │ 0.00           │ 0.00   │ +0.00 │
  ├───────────────────────┼────────────────┼────────┼───────┤
  │ Control 2 (mixed)     │ 0.10           │ 0.00   │ +0.10 │
  └───────────────────────┴────────────────┴────────┴───────┘
On relatively mixed boards that still place well, this neuron does not activate. But on any heavy trait board (except seemingly 8 Noxus, only 4 samples so not surprising), it activates, and with increasing strength if that board state is paired with matching emblems.
If we sweep across the most represented axis (Bilgewater), we can see this clear phenomena.
  ┌───────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────┬──────┬──────┬──────┐
  │ BG traits │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7   │  8   │  9   │  10  │
  ├───────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼──────┼──────┼──────┼──────┤
  │ N13       │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │ 0   │ 0.61 │ 0.83 │ 0.70 │ 1.47 │
  └───────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────┴──────┴──────┴──────┘

Because the model is relatively straightforward (1 layer of neurons), we can go backwards to see what dimensions actually encode this information by looking at the weights that point to neuron 13. There are only two such weights from our 8d board state that feed into this neuron with non zero weights, dimension 2 and 5. A quick look shows that neither of these dimensions are correlated with general champion strength, but rather represent something about the composition. Dim 2 seemingly measure how 'standard' a board is, with general high sample control groups scoring highly here. Dim 5 on the other hand seems to correlate a lot more with boards that have high 'trait' champions + emblems. A high presence of Dim 5 and low presence of Dim 2 is thus what ultimately activates neuron 13.

As a slight side note, I actually managed to find another rather monosemantic neuron, N20. This neuron basically only fires on boards with <3 champions and has a very high weight to 1st and 2nd place. Hilariously, the model has clearly learnt the behavior of people selling their boards when they are in 2nd place and resigning themselves, or if they are in first place with a 3 star 5 cost champion, with these gamestates occuring with roughly 100 samples, or 1% frequency.

It definitely begs the question as to whether we will see clearer delegation of the trait feature to N13 if we just upped the samples by a factor of lets say 3x for each of the prismatic traits. This is certainly worth investigating, but for now we can turn our eyes to the transformer block pre sum pooling to see whats going on there.








*The code for this project is on [GitHub](https://github.com/alexayvazyan/projects).*
