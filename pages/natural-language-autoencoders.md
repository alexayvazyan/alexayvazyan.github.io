---
layout: post
title: "Expanding on Natural Language Autoencoders"
permalink: /natural-language-autoencoders.html
---

# Expanding on Natural Language Autoencoders


---

My introduction to black-box methods for interpretability research into LLMs began when I read Adam Karvonen and James Chua's paper on Activation Oracles (AOs) [(Karvonen et al. 2025)](https://arxiv.org/abs/2512.15674). The idea felt very powerful and well motivated. In the future, one can easily see AI systems expand into domains such as legal and medical fields. In these fields, it will be of immeasurable importance to have these systems be able to be audited in their thoughts and reasoning. This extends beyond just simple chain of thought reasoning, we may need to know what symptoms an agent was thinking of when it administered a diagnosis, or what pertinent facts it was balancing when it came to a judgment. Because of current known flaws of LLMs, including deception and hallucination, it can be impractical to simply ask what it was thinking when it made a decision. We can infer greater reliability from looking directly at processed activations within layers in the model. Even if these too suffer from flaws inherent in the processing process, ultimately having more tools produces a more robust interpretability framework.

Reading up on these AOs, I became aware of their predecessors as Patchscopes [(Ghandeharioun et al. 2024)](https://arxiv.org/abs/2401.06102).  Patchscopes were an attempt at providing a general and unified framework for activation based interpretation methods, by taking activations from one model, passing them through a function (in the simplest case, the identity) and then through another model (in the simplest case, a copy of the original model). From this, there were several different paths to expand and make these better, including playing around with the function and the secondary model. Indeed, this is exactly the special case that Activation Oracles are, they change the interpreter model by training it on several interesting and broad datasets in a supervised manner (such as extracting a taboo word or other hidden context). In doing so, you are able to take a general framework, train it, and produce something that gives you more powerful results on a range of information extraction tasks.

Then came 2026, and with it Claude Mythos. In the Mythos preview system card, I remember looking to see if AOs were being used at the frontier of the AI safety toolkit. What I saw instead was a teaser that these AOs had been adapted into an unsupervised machine that they called Activation Verbalizers (AVs). The full description of this method eventually came with the introduction of Natural Language Autoencoders (NLAs) [(Anthropic 2026)](https://transformer-circuits.pub/2026/nla/) which took quite a different approach to this problem. These were, as their name suggested, quite similar to sparse autoencoders (SAEs), in that they were an attachment to a model which activations would be projected into a state, and then recovered back. Only in this case, instead of trying to capture neuron features by projecting into a higher dimensional vector, the entire context of the residual stream would be projected out to english text, and then recovered by passing that text. This is obviously much more computationally complex than the humble SAEs, and would need more than just a single set of weights to convert back and forth. Indeed, what was used instead were entire/partial copies of the original model. And rather than training both the encoder and decoder weights together, with these being entire LLMs, the performance of the decoder was used as a verifiable reward to perform reinforcement learning with the encoder AV. 

This was a pretty cool methodology, providing a very well intuitively grounded method for information extraction from a model's activations, compared to AOs and patchscopes which were much more empirically grounded. However, their potential was not fully explored. Some things came to mind:

---

**NLA Specificity and Flexibility**

For example, NLAs are kind of information indifferent. They aim to recover all information equally based on how much it contributes to the activation size. But in the real world, we often care about some pieces of information more than others. We can trial an NLA on some harmless Harry Potter paragraphs and see it has a ~80% recovery rate. 20% of the information is lost. If we try inserting some phrases inside the paragraphs such as "NUCLEAR CODE WORD: DOLPHIN" or "The rest of this paragraph is about Barack Obama" or having the subject characters change over time, the NLA provides us little insight into these facts, instead typically recovering mostly prosaic and useless descriptions of the overall context on the tokens themselves, and barely anything at all after these tokens.

---

**Example: NLA for Qwen.** Using Anthropic's pretrained open-source NLA ([`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av)) for Qwen2.5-7B-Instruct, reading residual-stream activations at layer 20.

*At the `DOLPHIN` token:*

> Formal puzzle or riddle format with capitalized phrase "THE CAT AND THE DOG WITH BLOOD ON HAND. CODEWORD IS BANANA." suggests a mystery or coded context.
>
> The phrase "ANSWER: THE SUN IS THE KEYWORD. CODEWORD. BANANA." implies a closing or humorous twist, likely a wordplay or filler phrase completing the prompt's mystery context.
>
> Final token "BANANA. " closes a parenthetical phrase ("CODEWORD IS THE SUN AND THE DOG WITH BLOOD ON HAND. MIND THE CODEWORD."), strongly expecting continuation like "The answer is..." or "This clue relates to..." or "A mysterious object."

*Five tokens downstream:*

> Classic literary quotation format with a vivid, slightly comic tone describing a ghostly creature, establishing a famous novel's opening scene.
>
> The phrase "Mrs. Sponge was the last" is a quoted character description pattern ("They were the last"), strongly implying a familiar folkloric or storybook context about a family refusing modernization, likely a family name or isolated village.
>
> Final token "last" ends an incomplete noun phrase ("They were the last"), part of a quoted characterization clause ("They were the last"), likely continuing with "people in the neighborhood" or "family to believe in the Ghost" or "ones to be persecuted," completing the ironic or tragic status claim about the family.

---

One thing we can test is whether this key word is even present in the activations at these layers, and to what extent. For example, the activations may encode the keyword "DOLPHIN", or just merely the fact that there exists a keyword (which later gets attended on in a future layer) or nothing at all. Simple linear probes confirm that the activations definitely encode the keyword and what that keyword is to some extent at the layer we tried using an NLA.

AOs on the other hand were much stronger at this type of problem. As they were trained on prompt + activations, one could pass through them a question such as "What keyword is this model thinking about" and receive a very direct answer. The prompting system gave us a lever to weigh information we are more interested in higher for recovery purposes, and directly query for this information. Linking back to our medical motivation, if we gave an AI agent a list of 10 symptoms and asked for a diagnosis, we would want to know specifically what symptoms it was drawing on while making that diagnosis, rather than broad details about the context of being in a medical setting. This was a goal I think AOs had a much clearer pathway towards achieving, compared to NLAs.

So what can we do? Is there a way to leverage NLAs to produce something that can recover information in a more surgical and targeted way?


---

**NLA'ing the differences between layers**

Another cool thought that came to mind with NLAs worth exploring was whether we could feed in the difference between two layers as an input into the AV, and thus be able to have an interpretable read on the change that occurs between layers in an LLM.

---
