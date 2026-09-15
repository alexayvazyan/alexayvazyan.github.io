---
layout: post
title: "Diagnosing text inversion in activation oracles"
permalink: /Diagnosing-text-inversion-in-activation-oracles.html
date: 2026-09-11
description: "Does an Activation Oracle read a model's thoughts, or rebuild the prompt and solve it itself? Model organisms that separate the two."
image: /assets/images/ao/ao_reading_vs_inversion.png
---

# Diagnosing text inversion in activation oracles

So a couple weeks ago I did a 24-hour research project as part of Neel Nanda's MATS application, a writeup of which can be found [here](/training-nlas.html). The topic I chose to investigate was that of metamodels, specifically whether Natural Language Autoencoders (NLAs) could benefit from improved performance on recovering specific information via supervised training. For example, we could SFT a pretrained NLA on medical diagnosis prompt activations of a target model and get it to return only (symptoms, diagnosis) tuples. In implementing this I achieved moderate success against the evaluation metrics I had set out, but I had achieved more than moderate success in frustrating myself with understanding whether the evaluation metrics were actually well set for the insights I was trying to extract. 

Yes we got the model to produce (symptom, diagnosis) pairs, yes they were often correct as judged by an independent model based on the prompt, and yes, they even generalized sometimes. But how do I know if I actually made the model better at extracting specific information from the target model, which is the intended AI safety contribution, versus just making a new transformer machine learning model that takes in the general context of the problem from the activations and postulates its own solution? Playing around with it a bit, my statistical intuition was certainly pointing towards the latter. The worst part is the uncertainty, I don't know whether it's trustable, whether it's providing anything useful or how it could ever be pragmatic. I suspected that the same issues would be present in the application of Activation Oracles (AOs). NLAs (the unsupervised true autoencoders) are at least somewhat grounded in the fact that we can have confidence that all they are trying to do is perform a faithful translation of any representations of activations into text. It was with this feeling of resignation and disappointment that I submitted my application. 

A week later, I stumbled upon a few pieces of literature that, as per usual, I wish I had read before embarking on the aforementioned project. Namely, Building Better Activation Oracles [(Bauer et al. 2026)](https://arxiv.org/abs/2606.02609) and the LessWrong post [Current activation oracles are hard to use](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use) both express very similar frustrations to what I was experiencing, just with AOs instead of trained NLAs. To quote from the LessWrong post, the crux of the issue that all of us identified was that - 

> "Evaluation is hard:
> 1. Text inversion: A large portion of AO training data involves identifying words that appear before or after a given position, so AOs are good at reconstructing nearby text from activations. This means correct answers don't prove the AO is reading deeper signal — if the text before a backtracking token mentions "modular arithmetic" and the AO says "the model is uncertain about modular arithmetic," it may just be recovering nearby tokens rather than reading internal model state. In these cases there's little reason to use an AO over a blackbox method that just reads the text. 
> Note that text inversion is a confounder for any technique that reads activations, not just AOs. It's worth flagging specifically for AOs because, unlike SAEs, they're explicitly trained to reconstruct nearby text - so text inversion is a more expected failure mode rather than an impressive capability.
> 2. It's an LLM: When the AO gives a correct answer, there's no guarantee it extracted that information from the subject model's activations. The AO is itself a capable language model - it may simply be reasoning to the correct answer using its own weights, with the activations playing no role. We saw this directly in the Chinese models experiments: running the AO without any activations injected produced comparable or better answers than running it with activations."

I didn't want to give up on metamodels though, they still seem like a powerful tool and I agree with the fundamental concepts underlying them. One path forward seemed to be a mechanistic analysis into whether text inversion is happening or not inside the AO. My hypothesis is that it's not like the metamodel is actively making a decision at a fork to go and reconstruct the prompt and then solve it versus just reading out a simple answer from the activations. More likely, the context just blends in as useful information to the AO containing some signal as learnt during training, potentially interfering with its readout. 

<figure class="research-figure research-figure--wide">
  <a href="{{ '/assets/images/ao/ao_reading_vs_inversion.png' | relative_url }}">
    <img src="{{ '/assets/images/ao/ao_reading_vs_inversion.png' | relative_url }}" alt="Side-by-side diagram: an Activation Oracle intercepting a model's thoughts and decoding them (reading), versus rebuilding the prompt and re-solving it (text inversion)" loading="eager">
  </a>
  <figcaption>Figure 1. Reading vs. text inversion. Reading decodes what the model computed; text inversion rebuilds the prompt from the activations and answers it with the AO's own weights. Click to open the full-resolution figure.</figcaption>
</figure>


## Methodology Ideas

- Arithmetic questions seem to be a relatively simple setting to examine our model organism.
- (1) We could come up with two groups consisting of AO pass-throughs that likely text inverted vs likely read from activation output. We can then contrast the model internals in each case to hopefully arrive at a fingerprint for whether a model is text inverting.
  - It may be of interest to examine cases where the target model gets an arithmetic question wrong, while the AO receiving its activations produces the correct answer. Such cases may be good candidates for "likely to be doing something akin to text inversion".
  - We ideally compare these to cases where the activation stream is missing a key piece of context to arrive at the answer for our non-inversion group.
- (2) Alternatively, the AO itself if prompted may be able to answer simple arithmetic without any activations. We could compare its activations while trying to solve the problem on its own vs from activations and look for any similarities as evidence of text inversion.
- (3) Another interesting idea is to take the base model and fine tune it on a specific task that has a pre-existing meaning, for example, relearning multiplication into a different operation. We can pass activations from this model into our AO and see if it outputs the original calculation or the secret calculation.

## Results

### Method (1)

We first began with method (1) with rather uninspiring results. The problems with this approach are broadly summarized as
- Cases where AOs get arithmetic problems right after the original model gets it wrong are rather thin. It is also not entirely clear these are examples of text inversion rather than just ordinary operation of the AO and an "unlucky" misstep in the original model.
  - As a result, the amount of inversion signal in these candidates was quite low.
- Comparing internals of the two sets under this method was also problematic.
  - Cosine similarities were very high with both direct computation and each other. Models, when doing a task like arithmetic, can have very similar activation vectors but produce substantially different numerical logit vectors.
  - The next approach was thus to use methods attributing direct logit contributions to each attention head / MLP layer and contrast these.
    - These identified several heads which failed to generalize across problem sets and non arithmetic examples.

<figure class="research-figure research-figure--portrait">
  <a href="{{ '/assets/images/ao/ao_method1_grouping.png' | relative_url }}">
    <img src="{{ '/assets/images/ao/ao_method1_grouping.png' | relative_url }}" alt="Method 1 steps: grouping AO passes into likely inversion and likely reading, cosine similarity comparison, and direct logit attribution" loading="lazy">
  </a>
  <figcaption>Figure 2. Method (1): grouping AO passes by their likely route, then comparing internals. Click to open the full-resolution figure.</figcaption>
</figure>

### Method (2)

For this method, we first check to see whether the AO can actually solve arithmetic problems without any activations supplied. Indeed it can, great! 
- We run through 500 arithmetic equations via the direct solve and activation paths. We attempt to check whether the activation path solves mirror the direct solve path, as evidence of inversion.
  - When we compare the activation vectors of the AO, there is very little problem-specific similarity to the direct solve until the later layers. The fact that they are similar at later layers is not that surprising as the models are working to produce the same logit. Although as we noted in method (1), logit production and similarity of activation vectors are not necessarily paired facts, so this is still somewhat significant. The lack of similarity in the earlier layers is also not too surprising, given we are starting from entirely different latent vectors.
- 

<figure class="research-figure research-figure--wide">
  <a href="{{ '/assets/images/ao/ao_method2_direct_vs_activation.png' | relative_url }}">
    <img src="{{ '/assets/images/ao/ao_method2_direct_vs_activation.png' | relative_url }}" alt="Method 2: the AO solving a prompt directly vs. from activations, with same-problem retrieval of its residual stream by layer" loading="lazy">
  </a>
  <figcaption>Figure 3. Method (2): the AO's direct solve vs. its activation solve, and how closely their internal states match by layer. Click to open the full-resolution figure.</figcaption>
</figure>

### Method (3)

In this method, we start by training our base model with a secret arithmetic rule. In particular, we finetune so that a × b becomes a + 3b − 7. We train on 2,784 pairs with operands from 2 to 60, and see that our new target model is able to pick up the rule and also able to somewhat extrapolate this to outside the range. Great. We did this fine tuning through a LoRA on layers 0-20, with us eventually taking the activations to pass to the AO from layers 21-25.

Now that we have a model that treats multiplication with this secret algorithm, the first question to ask is whether the 'true' answer is actually in the activations at layers 21-25. Linear probes turn out to be inconclusive here, but we can examine this through activation patching and indeed, it's there.

So now the question becomes, does the AO use this computed value and read out this, or does it re-solve the original multiplication problem? The answer was unambiguous, the original multiplication answer was presented by the AO 60-67% of the time, and the secret answer just once in 348 problems. Further analysis looking at log probabilities of answers across candidates revealed no extra probability on the secret answer, and that the trained rule at best reaches the AO with some distortion (a × b is partly read as a + b). This is the strongest evidence that AOs, at least for relatively simple arithmetic, are largely just models solving questions deduced from target activations themselves.

<figure class="research-figure research-figure--portrait">
  <a href="{{ '/assets/images/ao/ao_method3_secret_rule.png' | relative_url }}">
    <img src="{{ '/assets/images/ao/ao_method3_secret_rule.png' | relative_url }}" alt="Method 3 steps: training the secret-rule LoRA, verifying the answer with patching, AO results, and follow-up checks" loading="lazy">
  </a>
  <figcaption>Figure 4. Method (3): the secret-rule target, the check that its answer is present where the AO reads, and what the AO reports. Click to open the full-resolution figure.</figcaption>
</figure>
