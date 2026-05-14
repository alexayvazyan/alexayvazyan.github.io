---
layout: post
title: "Representations on an Entropy Axis"
permalink: /representations-entropy-tree.html
---

# Representations on an Entropy Axis

---

Compression and tractability are the through-lines for almost everything I work on, and increasingly the vocabulary that contemporary learning-related research seems to be settling into. Papers like ["Compression is all you need"](https://arxiv.org/html/2603.20396v1), and the multidimensional embeddings learned by large language models, both point at the same thing: human understanding and human abstractions are a small subset of the ways information can be carried and consumed. Mechanistic interpretability is, in some sense, the study of building bridges between these different compressions — a translation layer through which we hope to understand, communicate with, and guide models.

## The root

Start the mapping of this tree of abstraction at the root: the singular point that contains all information about the universe with no compression. The being capable of holding such a representation was dubbed Laplace's Demon, precisely because processing it would require a divine level of computational intelligence. If we reduce any system to this atomic level — every position, every momentum, every quantum amplitude — we know everything available about the current state of the universe. If someone asks us why a model produces a certain output, we could in principle answer by tracing the flow of charge through its transistors. The answer would be correct and entirely useless. One step higher in abstraction, we could imagine reasoning about the output by tracing the stochastic updates the weights underwent during training. For simple linear or decision-tree models with a handful of parameters, that's a perfectly valid methodology, sometimes even an engineering one. For large language models, it isn't — the information has to be further compressed, into human-readable features, before any useful deductions can be made.

So the root of the tree is information-rich and use-poor. It is the limit case against which every other representation is a compromise.

---

## Axis - Compression and Tractability

Every representation we actually use is a compression. We discard the position of individual gas molecules and keep temperature and pressure. We discard the trajectory of every photon and keep "the room is lit." We discard the firing pattern of every neuron and keep "she's smiling." Each step throws away orders of magnitude of information and gains, in exchange, the ability to think about the system in a finite amount of time with a finite-sized brain.

But alongside compression it can be useful to think along another axis as well, that of tractability. These are not the same, but they are related. We can define tractability in the context of human understanding, as the ability of some information to be used, consumed, modified or interpreted.

At any slice of information level, there can be many ways to represent information, with some representations more tractable than others. We can say though that at the Laplacian information root, all ways of representing this have the same tractability - zero. Furthermore, we can loosely argue the same is true at the other end of the spectrum, when information is maximally compressed and lost into a singular number let's say - again zero tractability. In the middle ground though, different representations of the same information content can exist with varying levels of usability.  

Starting from the root at the most information dense point, compression generally allows for more tractability. This is a byproduct of the human brain. We are not built to handle representations that are too information-dense — we simply do not have the neuron capacity to be able to efficiently compute from such representations in a reasonable amount of time. 

This has a limit, however. Compress too much and again, you end up reducing the degree to which information can be tractable. Why? Noise. Think about a video, or a piece of text. What happens as you keep compressing it? Generally, it stays interpretable up to a point (for a video this limit is ~99.9% bit compression, for English language it's typically quoted at around ~50%), after which it becomes somewhat indistinguishable from random noise - there is no signal that can be recovered. 

![A right triangle whose hypotenuse is the vertical axis: the y-axis is compression / information loss and horizontal width is tractability. Tractability is zero at the Laplacian root at the bottom and at maximal compression at the top, peaking in the middle. An arrow along the upper leg points toward pure noise; an arrow along the lower leg points toward too information dense.]({{ site.baseurl }}/assets/images/representations-entropy-tree_tractability-triangle.png)

And every theoretical physicist that's any good knows six or seven different theoretical representations for exactly the same physics, and knows that they're all equivalent, and that nobody's ever going to be able to decide which one is right at that level. But he keeps them in his head, hoping that they'll give him different ideas for guessing. - Richard Feynman

---

## Where humans live

Human cognition lives at a specific, somewhat narrow band of compression depth. Deep enough that we can think in concepts (justice, momentum, family) rather than configurations of cells. Shallow enough that the concepts retain enough structure to support causal reasoning about the world.

![The same compression-and-tractability triangle with a yellow wedge around the apex marking where human understanding sits, at peak tractability.]({{ site.baseurl }}/assets/images/representations-entropy-tree_tractability-triangle-human.png)

Crucially, the *shape* of human compression is constrained by what humans need to do: communicate with each other, predict each other's behavior, coordinate joint action over time. So our concepts are biased toward the kinds of features that are stable across observers, expressible in linear sequences of words, and learnable from a finite number of examples.

This is why human concepts have the structure they do. "Cause," "agent," "promise," "guilt," "left and right" — these are not arbitrary categories cut out of the world. They are the categories that survive the joint constraints of being mostly true, mostly communicable, and mostly useful for predicting what other humans will do next. The set of concepts that meets all three constraints is much smaller than the set of concepts that just meets the first.

Mathematics and physics are interesting in this picture because they are the project of pushing as far up the tree (toward the root) as humans can manage while still preserving communicability. A theorem is an abstraction that retains so much structure that it generalizes across nearly all branches at lower depths. The price is that it takes a lot of training to use one, and even then most people can't.

Any science that tries to understand how something works is, in this framing, building a bridge from the set of human concepts to something outside it. The standard move is to go up the tree, where a higher-information state is recoverable — though it isn't always. To bring quantum mechanics into the domain of human understanding, experimentalists gathered information about particles under various conditions, theorists fit mathematical structures to those observations, and the field tied the structures back into communicable form. Each of those steps is a bridge.

---

## Where language models live

A language model trained on the entire internet has built a representation that is shaped by completely different constraints. It does not need its concepts to be communicable to other agents — it only needs them to predict the next token. It does not need them to be stable across observers — it only needs them to compress its training distribution. It does not need them to support causal reasoning — only correlational fluency.

![The compression-and-tractability triangle with its horizontal axis relabelled machine tractability, or learned readout efficiency, and a blue wedge around the apex marking where machine understanding sits, at peak machine tractability.]({{ site.baseurl }}/assets/images/representations-entropy-tree_tractability-triangle-machine.png)

So we should expect the model's representation to live in a different region of the tree than humans do. Some of its concepts will overlap with ours, because human language is the substrate of its training data, and the structure of language partially encodes the structure of human concepts. But the overlap is partial. The model has features for things humans don't bother to name (the difference between two formatting conventions in academic writing, or some superposition of "this is a question" and "the writer is mildly annoyed"), and humans have concepts the model has only a thin handle on (embodied skills, the felt sense of effort, the difference between knowing something and remembering it).

Polysemanticity is the model's response to exactly this kind of pressure. Models are incentivised to encode information as efficiently as possible given the bandwidth of their attention blocks and feed-forward layers, and the cheapest way to do that is to overload single neurons with multiple features. The linear representation hypothesis is a direct corollary: given how linear the architecture is, it's no surprise that features which *can* be expressed as a single direction in activation space are. It's optimal packing inside a fixed-size information container.

Mechanistic interpretability work over the last few years has been steadily clarifying this picture. Superposition tells us that models pack many more features into their residual stream than they have dimensions, by exploiting the sparsity of feature co-activation. Sparse autoencoders have been finding features that are interpretable as human concepts — but also features that don't correspond to any concept we have a name for, and that we have to invent labels for ad hoc by inspecting their activation patterns. The features are real, in the sense that they're causally functional in the model's computation. They just don't sit in the same neighborhood of the tree as our concepts do.

The closer you look, the more the model's representation looks like a *different compression* of the same root world, with branches that mostly don't line up with ours. Sometimes a branch of the model's tree happens to terminate near a branch of ours — that's when probing finds a clean direction for "truth" or "refusal" or "size." Other times it terminates somewhere we have no name for, and we either invent a name or shrug and move on.

---

## Mechanistic interpretability as a bridge

If you take this framing seriously, mech interp is the project of building translation infrastructure between two regions of the compression tree that don't natively share vocabulary. Not a dictionary — a dictionary assumes the concepts on each side already correspond — but something more like a topographer's map of the contact surface. Where do the model's branches reach close enough to ours that we can identify them with a human concept? Where do they reach close to each other but not to anything we have a name for? Where do they not reach at all?

The honest answer is that the contact surface is small. Most of the model's representation is, from our standpoint, illegible — not because it's encrypted, but because it's a different compression of the same world, optimized for a task we don't share. The features we successfully label are the lucky cases where the model's compression happened to converge on something close to ours.

![The triangle on the human-tractability axes showing both human and machine representations. Human representations bulge to the apex at peak human tractability; machine representations hug the opposite low-tractability edge, spanning a wide range of compression. Neither leaves the triangle, since the boundary is the definitional edge of the plane. The small gap between the machine region and the axis is the human tractability that interpretability has recovered.]({{ site.baseurl }}/assets/images/representations-entropy-tree_tractability-triangle-representations.png)

This is why interp is hard, and why it's important. It's hard because we are trying to read a representation that wasn't designed to be read. It's important because as models get more capable, the fraction of their cognition that is illegible grows, and the fraction of consequential decisions made in that illegible region grows with it. Bridging more of the tree — finding more contact points, building more translation surfaces — is the only path I can see toward being able to say, with any confidence, what these systems are actually doing.

---
