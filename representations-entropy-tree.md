---
layout: post
title: "Representations on an Entropy Axis"
permalink: /representations-entropy-tree.html
---

# Representations on an Entropy Axis

*A musing. The framing I've been carrying around is that any representation of the world sits somewhere on an axis of entropy. The root of that axis — call it the Laplacean point — is the position of every particle and every field value at every instant. From there, abstractions branch outward: each one throws away information in exchange for tractability. Humans live deep in the branches. Language models live in a different region of the same tree, with a topology that mostly doesn't line up with ours. Mechanistic interpretability is the project of finding the places where the two regions touch.*

---

Compression and tractability are the through-lines for almost everything I work on, and increasingly the vocabulary that contemporary learning-related research seems to be settling into. Papers like ["Compression is all you need"](https://arxiv.org/html/2603.20396v1), and the multidimensional embeddings learned by large language models, both point at the same thing: human understanding and human abstractions are a small subset of the ways information can be carried and consumed. Mechanistic interpretability is, in some sense, the study of building bridges between these different compressions — a translation layer through which we hope to understand, communicate with, and guide models.

## The root

Start the mapping of this tree of abstraction at the root: the singular point that contains all information about the universe with no compression. The being capable of holding such a representation was coined Laplace's Daemon, precisely because processing it would require a divine level of computational intelligence. If we reduce any system to this atomic level — every position, every momentum, every quantum amplitude — we know everything available about the current state of the universe. If someone asks us why a model produces a certain output, we could in principle answer by tracing the flow of charge through its transistors. The answer would be correct and entirely useless. One step higher in abstraction, we could imagine reasoning about the output by tracing the stochastic updates the weights underwent during training. For simple linear or decision-tree models with a handful of parameters, that's a perfectly valid methodology, sometimes even an engineering one. For large language models, it isn't — the information has to be further compressed, into human-readable features, before any useful deductions can be made.

So the root of the tree is information-rich and use-poor. It is the limit case against which every other representation is a compromise.

---

## Branches as compression

Every representation we actually use is a compression. We discard the position of individual gas molecules and keep temperature and pressure. We discard the trajectory of every photon and keep "the room is lit." We discard the firing pattern of every neuron and keep "she's smiling." Each step throws away orders of magnitude of information and gains, in exchange, the ability to think about the system in a finite amount of time with a finite-sized brain.

The branching structure matters. Two different compressions of the same root system are not interchangeable. Statistical mechanics and fluid dynamics both compress molecular motion, but they compress it differently and they're useful for different questions. You can convert between them with effort, but the conversion is lossy in both directions because each compression has decided which features are signal and which are noise — and they disagreed.

The tree has another property worth naming: **branches at similar depths are not necessarily nearby in compression-space.** "Temperature" and "the dollar's exchange rate" are both extremely compressed summaries of physical configurations of matter, but their nearest common ancestor on the tree is most of the way back to the root. Compression is not metric. Two abstractions can be equally distant from the ground truth and yet have nothing to say to each other.

Feynman is supposed to have said that a good theoretical physicist should keep seven different representations of the same phenomenon at hand. There are many ways to represent the same object even at the same level of compression — and a sign of mastery is being fluent across them.

---

## Where humans live

Human cognition lives at a specific, somewhat narrow band of compression depth. Deep enough that we can think in concepts (justice, momentum, family) rather than configurations of cells. Shallow enough that the concepts retain enough structure to support causal reasoning about the world.

Crucially, the *shape* of human compression is constrained by what humans need to do: communicate with each other, predict each other's behavior, coordinate joint action over time. So our concepts are biased toward the kinds of features that are stable across observers, expressible in linear sequences of words, and learnable from a finite number of examples.

This is why human concepts have the structure they do. "Cause," "agent," "promise," "guilt," "left and right" — these are not arbitrary categories cut out of the world. They are the categories that survive the joint constraints of being mostly true, mostly communicable, and mostly useful for predicting what other humans will do next. The set of concepts that meets all three constraints is much smaller than the set of concepts that just meets the first.

Mathematics and physics are interesting in this picture because they are the project of pushing as far up the tree (toward the root) as humans can manage while still preserving communicability. A theorem is an abstraction that retains so much structure that it generalizes across nearly all branches at lower depths. The price is that it takes a lot of training to use one, and even then most people can't.

Any science that tries to understand how something works is, in this framing, building a bridge from the set of human concepts to something outside it. The standard move is to go up the tree, where a higher-information state is recoverable — though it isn't always. To bring quantum mechanics into the domain of human understanding, experimentalists gathered information about particles under various conditions, theorists fit mathematical structures to those observations, and the field tied the structures back into communicable form. Each of those steps is a bridge.

---

## Where language models live

A language model trained on the entire internet has built a representation that is shaped by completely different constraints. It does not need its concepts to be communicable to other agents — it only needs them to predict the next token. It does not need them to be stable across observers — it only needs them to compress its training distribution. It does not need them to support causal reasoning — only correlational fluency.

So we should expect the model's representation to live in a different region of the tree than humans do. Some of its concepts will overlap with ours, because human language is the substrate of its training data, and the structure of language partially encodes the structure of human concepts. But the overlap is partial. The model has features for things humans don't bother to name (the difference between two formatting conventions in academic writing, or some superposition of "this is a question" and "the writer is mildly annoyed"), and humans have concepts the model has only a thin handle on (embodied skills, the felt sense of effort, the difference between knowing something and remembering it).

Polysemanticity is the model's response to exactly this kind of pressure. Models are incentivised to encode information as efficiently as possible given the bandwidth of their attention blocks and feed-forward layers, and the cheapest way to do that is to overload single neurons with multiple features. The linear representation hypothesis is a direct corollary: given how linear the architecture is, it's no surprise that features which *can* be expressed as a single direction in activation space are. It's optimal packing inside a fixed-size information container.

Mechanistic interpretability work over the last few years has been steadily clarifying this picture. Superposition tells us that models pack many more features into their residual stream than they have dimensions, by exploiting the sparsity of feature co-activation. Sparse autoencoders have been finding features that are interpretable as human concepts — but also features that don't correspond to any concept we have a name for, and that we have to invent labels for ad hoc by inspecting their activation patterns. The features are real, in the sense that they're causally functional in the model's computation. They just don't sit in the same neighborhood of the tree as our concepts do.

The closer you look, the more the model's representation looks like a *different compression* of the same root world, with branches that mostly don't line up with ours. Sometimes a branch of the model's tree happens to terminate near a branch of ours — that's when probing finds a clean direction for "truth" or "refusal" or "size." Other times it terminates somewhere we have no name for, and we either invent a name or shrug and move on.

---

## Mechanistic interpretability as a bridge

If you take this framing seriously, mech interp is the project of building translation infrastructure between two regions of the compression tree that don't natively share vocabulary. Not a dictionary — a dictionary assumes the concepts on each side already correspond — but something more like a topographer's map of the contact surface. Where do the model's branches reach close enough to ours that we can identify them with a human concept? Where do they reach close to each other but not to anything we have a name for? Where do they not reach at all?

The honest answer is that the contact surface is small. Most of the model's representation is, from our standpoint, illegible — not because it's encrypted, but because it's a different compression of the same world, optimized for a task we don't share. The features we successfully label are the lucky cases where the model's compression happened to converge on something close to ours.

This is why interp is hard, and why it's important. It's hard because we are trying to read a representation that wasn't designed to be read. It's important because as models get more capable, the fraction of their cognition that is illegible grows, and the fraction of consequential decisions made in that illegible region grows with it. Bridging more of the tree — finding more contact points, building more translation surfaces — is the only path I can see toward being able to say, with any confidence, what these systems are actually doing.

---

## A few corollaries I find interesting

**Interpretability has an asymptote, not a finish line.** If the model's compression is genuinely different from ours, there is no point at which we will have "fully understood" it. We can build more bridges, label more features, identify more circuits, but the regions of its representation that don't admit clean human translation will remain. The right metric is not "understood vs not understood" but "what fraction of consequential computation is reachable from a human-understandable handle."

**Some abstractions are easier to bridge than others.** Physical scalars (size, speed, distance) are easy because the root-level structure is simple and most reasonable compressions converge on similar features. Social abstractions (intent, blame, status) are hard because human concepts in this space are heavily shaped by the communicability constraint, and the model has no analogous pressure. I'd predict, and the empirical interp record so far is consistent with this, that we'll find clean linear directions for the former and messy distributed representations for the latter.

**Models can move along the axis.** Training pushes a model's representation toward whatever compression best predicts its data. Fine-tuning, RLHF, instruction tuning — each shifts the location of the model's branches. Some of these shifts move the model's representation closer to ours (because human feedback is itself a signal about which features humans care about), and some move it further (because they reward fluency in a particular sub-distribution that may not map to our normal cognitive landmarks). Tracking these shifts is its own mech interp project.

**The tree framing is not the territory.** A tree is a particular shape. Real compressions probably don't form a tree — they form something messier, with reconvergences and shortcuts. The tree is the simplification I find useful for thinking about the relationship between human and model concepts; it would be a mistake to insist that the actual structure has to be tree-shaped. If a future, sharper picture turns out to look more like a graph or a fiber bundle, fine. The point is the axis of compression depth, and the observation that two agents at similar depth can still inhabit different neighborhoods.

---

## Why this matters to me

The reason I find this framing useful, beyond as a way to organize my own thinking about the field, is that it suggests where to look. If interp is bridge-building, then high-value bridges are in places where (a) the model has a feature we can localize, (b) that feature plausibly corresponds to a human concept, and (c) understanding the correspondence has consequences — for safety, for capability, for whether we can trust the model's behavior in distribution shift.

A clean refusal direction, when you find one, is a high-value bridge: model-internal feature, human-meaningful concept, safety-consequential. The fact that the bridge breaks on Gemma 4 — that the refusal direction rotates partway through the network — is itself information about the topology of the contact surface, and about how brittle our assumptions of correspondence are.

The momentum failure I wrote about [in the Gemma 4 size/speed post](/2026/04/29/geometry-speed-size-gemma4-31b.html) is an instance of the same thing from the other direction: the model has a clean size manifold (a feature that corresponds well to a human concept), but the *use* of that feature in a downstream computation breaks. We have a bridge to the static representation, and no bridge to whatever circuitry was supposed to consume it. That gap is exactly the territory worth charting.

I expect to come back to this framing a lot. It is the loosest possible commitment I can make about how to think about model representations while still committing to anything at all. I find that useful — it constrains how I think without constraining what I find.
