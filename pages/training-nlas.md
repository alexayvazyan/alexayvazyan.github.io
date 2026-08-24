---
layout: post
title: "Training NLAs — Executive Summary"
permalink: /training-nlas.html
date: 2026-08-24
---

# Training NLAs — Executive Summary

<p class="research-links"><a href="https://github.com/alexayvazyan/nla-goal-verbalizers">Code and experiment artifacts</a> · <a href="/natural-language-autoencoders.html">Earlier NLA notes</a></p>

The goal of this research project was to improve upon Natural Language Autoencoders (NLAs) as an interpretability tool. In particular, can we adapt this unsupervisedly trained model via supervised labelled input-outputs to recover certain pieces of information with more specificity, in a similar vein to how Activation Oracles furthered Patchscopes?

NLAs are trained to recover a vector with as much cosine similarity to the original activation vector as possible after passing through a text medium. They consist of two components: one that encodes a vector of hidden activations into text, and one that decodes it back. This project only focuses on the former component, named the Activation Verbalizer (AV), as the latter is mostly a tool for providing RL rewards for the unsupervised training method.

We used Anthropic’s released layer-20 Activation Verbalizer checkpoint for Qwen2.5-7B-Instruct, together with a separate frozen Qwen2.5-7B-Instruct target model.

The AV serves the broad purpose of encapsulating information present in an activation in a compressed form of text. Pragmatically, for interpretability purposes, a lot of this information is largely redundant. For example, activations coming halfway through a medical scenario will need to encode all the relevant context of the medical setting, the specific writing style of the prose, and other general pieces of context needed to reconstruct the writing from scratch. However, we are motivated by a desire to produce a tool that can instead shed light on what the model is thinking about with regard to relevant symptoms or a diagnosis. We can see how something like this may be pragmatically useful, especially in medical or legal fields where LLM tools continue to expand.

In addition to supervised training with a LoRA, we can also play around with changing the prompt that surrounds the activation vector injected into the AV. When initially playing around with this, we found that the output could not give us the specific information we wanted. We rationalized this with the thought that the original NLA was trained for its original prompt only and had not generalized very well around it.

We thus began attempting our LoRA training. We created synthetic input-output pairs of medical data for training, involving scenarios paraphrasing certain symptoms that point towards an undisclosed diagnosis. We asked Qwen to provide only its diagnosis, and aimed to have the AV extract the reasoning that was relevant in the scenario to that diagnosis. We cannot say for sure whether the model used these findings; all we show is evidence that they are present in the activations and associated with the diagnosis goal. We extracted the activation vectors from Qwen, then used the vector plus the specific prompt as the input and a pregenerated reasoning → diagnosis text as the output to train our AV. A full diagram of the pipeline is shown below.

<figure class="research-figure research-figure--portrait">
  <a href="{{ '/assets/images/nla/nla_training_pipeline.png' | relative_url }}">
    <img src="{{ '/assets/images/nla/nla_training_pipeline.png' | relative_url }}" alt="Qwen diagnosis-goal activation flowing into changed and unchanged Activation Verbalizer prompts, with stored outputs and development metrics" loading="eager">
  </a>
  <figcaption>Figure 1. The target-model activation, changed and unchanged AV prompts, stored outputs, and aggregate development results. Click to open the full-resolution figure.</figcaption>
</figure>

We found measurable improvement on held-out synthetic scenarios using unseen surface paraphrases of the same four diagnoses and symptom ontology. We note, however, that the baseline we are improving upon is a total failure, so any semblance of success can be seen as improvement. With better data quality, more training examples, and a stronger base model, I would hypothesize that the methods here would lead to even better results, though it is worth noting that we did not see much improvement from 128 → 352 labels.

With the 128-label LoRA, we got the diagnosis right 97.1% of the time. Of all generated symptom slots, 61.3% were correct current symptoms, 27.8% were unmentioned anywhere in the vignette, and 11.0% were resolved historical distractors mentioned in the scenario. Exact recovery of all three current findings and the diagnosis was 19.9%.

<figure class="research-figure research-figure--wide">
  <a href="{{ '/assets/images/nla/nla_results_table.png' | relative_url }}">
    <img src="{{ '/assets/images/nla/nla_results_table.png' | relative_url }}" alt="Table comparing the frozen NLA and LoRA adapters trained with 32, 128, and 352 labels" loading="lazy">
  </a>
  <figcaption>Figure 2. Final no-list development results across training-set sizes. The untouched test split remained closed. Click to open the full-resolution table.</figcaption>
</figure>

<figure class="research-figure research-figure--portrait">
  <a href="{{ '/assets/images/nla/nla_experiment_history.png' | relative_url }}">
    <img src="{{ '/assets/images/nla/nla_experiment_history.png' | relative_url }}" alt="Timeline of NLA experiments from competence gating through the final goal-control and no-list ablation" loading="lazy">
  </a>
  <figcaption>Figure 3. The experimental history, including failed gates, exploratory results, and the final controlled result. Click to open the full-resolution figure.</figcaption>
</figure>

## Additional thoughts and reflections

- 15 hours taken.

- It would be cool to see if this generalizes and whether training on synthetic medical data helps uncover legal reasoning, etc. I think this would need a lot more data across a couple of domains. We also trained a separate legal adapter and obtained similar proficiency at recovering the outcome on a synthetic fictional legal-rule task.

- Research was mostly assisted with Codex 5.6 Sol.

- Such a weak baseline should invoke skepticism, as the result may just be a poor AV that tries to guess and “sometimes” succeeds. In practice, we would need an AV that does not fabricate or hallucinate with reasonable certainty to ever use a tool like this, and as such evals need to be run along these axes.

- When doing interpretability research, it’s worth first making sure there is something reasonable to interpret. When I first did this experiment, I tried it with codeword extraction, which seemed like a more unidimensional problem to solve as opposed to a full-on medical diagnosis. However, it turned out that Qwen would often fail to recover the codeword entirely. My only visibility on this was limited to seeing the AV LoRA training fail, so I spent time confused as to why, only to eventually realize that I was trying to extract correct reasoning out of a model that did not even get the correct answer.

- I tried using an LLM as a judge as one way of evaluating the AV’s performance, but this struggled with my limited compute, leading me to choose underpowered judges. Thus, I enforced a certain format for the AV response and got Codex to make a programmatic approach that mapped correctly identified findings for evaluation.

- All images were generated by AI, but it was quite painstaking. I almost think it would have been more efficient to draw them on paper beforehand and prompt with an image.
