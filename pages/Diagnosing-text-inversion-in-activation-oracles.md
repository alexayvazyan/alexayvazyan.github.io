---
layout: post
title: "Diagnosing text inversion in activation oracles"
permalink: /Diagnosing-text-inversion-in-activation-oracles.html
date: 2026-09-11
---

# Diagnosing text inversion in activation oracles

So a couple weeks ago I did a 24 hour research project as part of Neel Nanda's MATS application, a writeup of which can be found here <>. The topic I chose to investigate was that of metamodels, specifically whether Natural Language Autoencoders (NLAs) could benefit from improved performance on recovering specific information via supervised training. For example, we could SFT a pretrained NLA on medical diagnosis prompt activations of a target model and get it to return only (symptoms, diagnosis) tuples. In implementing this I achieved moderate success against the evaluation metrics I had set out, but I had achieved more than moderate success in frustrating myself with understanding whether the evaluation metrics were actually well set for the insights I was trying to extract. 

Yes we got the model to produce (symptom, diagnosis) pairs, yes they were often correct as judged by an independent model based on the prompt, and yes, they even generalized sometimes. But how do I know if I actually made the model better at extracting specific information from the target model, which is the intended AI safety contribution, versus just making a new transformer machine learning model that takes in the general context of the problem from the activations and postulates its own solution? Playing around with it a bit, my statistical intuition was certainly pointing towards the latter. The worst part is the uncertainty, I don't know whether its trustable, whether its providing anything useful or how it could ever be pragmatic. I suspected that the same issues would be present in the application of Activation Oracles (AOs). NLAs (the unsupervised true autoencoders) are at least somewhat grounded in the fact that we can have confidence that all they are trying to do is perform a faithful translation of any representations of activations into text. It was with this feeling of resignation and disappointment that I submitted my application. 

A week later, I stumbled upon a few pieces of literature that, as per usual, I wish I had read before embarking on the aforementioned project. Namely, <building better activation oracles> and <lesswrong post about how AOs are hard to evaluate> both express very similar frustrations to what I was experiencing, just with AOs instead of trained NLAs. To quote from the lesswrong post, the crux of the issue that all of us identified was that - 

"Evaluation is hard:
1. Text inversion: A large portion of AO training data involves identifying words that appear before or after a given position, so AOs are good at reconstructing nearby text from activations. This means correct answers don't prove the AO is reading deeper signal — if the text before a backtracking token mentions "modular arithmetic" and the AO says "the model is uncertain about modular arithmetic," it may just be recovering nearby tokens rather than reading internal model state. In these cases there's little reason to use an AO over a blackbox method that just reads the text. 
Note that text inversion is a confounder for any technique that reads activations, not just AOs. It's worth flagging specifically for AOs because, unlike SAEs, they're explicitly trained to reconstruct nearby text - so text inversion is a more expected failure mode rather than an impressive capability.
2. It's an LLM: When the AO gives a correct answer, there's no guarantee it extracted that information from the subject model's activations. The AO is itself a capable language model - it may simply be reasoning to the correct answer using its own weights, with the activations playing no role. We saw this directly in the Chinese models experiments: running the AO without any activations injected produced comparable or better answers than running it with activations."

I didn't want to give up on metamodels though, they still seem like a powerful tool and I agree with the fundamental concepts underlying them. One path forward seemed to be a mechanistic analysis into whether text inversion is happening or not inside the AO. My hypothesis is that it's not like the metamodel is actively making a decision at a fork to go and reconstruct the prompt and then solve it versus just reading out a simple answer from the activations. More likely, the context just blends in as useful information to the AO containing some signal for as learnt during training, potentially interfering with its readout. 


Brief summary of methodology:


- Arithmetic questions seem to be a relatively simple setting to examine our model organism.
- We could come up with two groups consisting of AO pass throughs that likely text inverted vs likely read from activation output. We can then contrast the model internals in each case to hopefully arrive at a fingerprint for whether a model is text inverting.
  - It may be of interest to examine cases where the target model gets an arithmetic question wrong, while the AO receiving its activations produces the correct answer. Such cases may be good candidates for "likely to be doing something akin to text inversion".
  - We ideally compare these to cases where the activation stream is missing a key piece of context to arrive at the answer for our non-inversion group.
-  Alternatively, the AO itself if prompted may be able to answer simple arithmetic without any activations. We could compare its activations while trying to solve the problem on its own vs from activations and look for any similarities as evidence of text inversion.


Results:
