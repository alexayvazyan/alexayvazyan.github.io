---
layout: post
title: "Arbitrary Thoughts"
---

# Arbitrary Thoughts

*A collection of various thoughts, normally formulated in bed while trying to sleep*

---

## 1 - Mean Variance Optimization

In the course of my work as a quant trader I encountered the optimization function for mean variance:

O(h) = h_t * mu - gamma * h_t * sigma * h

where h is the proportion vector of holdings across various securities, mu and sigma are the mean and covariance matrix of those securities, and finally gamma is a hyperparameter.

Initially it wasn't obvious to me that such a formulation can be equivalent to maximizing sharpe, even though I was told that it was equivalent. I knew that the reason it was normally formulated like this was due to a desire for linear and fast differentiation.

Thinking about it more though, if one is to fix the covariance matrix or the mean return vector, optimization maximizes/minimizes the other. In this way, optimizing this function guarantees that you are on the 'efficient frontier' as described in markowitz portfolio theory.

With gamma = 0, you lie on the point of the frontier furthest from the origin. As gamma approaches infinity, one appraoches the basin of minimum variance. Moving gamma up thus moves you across the efficient frontier, at some point encountering the optimal sharpe point. This intuitively also gives an understanding why gamma should scale hyperbolically. 

## 2 - 12 balls and 3 weighs

A classic puzzle/brainteaser. You have 12 balls, one of which is a fake that either weighs slightly more or less than all the others. What is the least amount of times you need to use a scale to determine which ball is fake? The answer turns out to be 3.

The standard way to solve this that almost anyone would take would be to split into cases and just work through all possible paths. If thats all there was to it, it wouldn't be that interesting of a problem. Looking at just the first weighing though, we can see some logic in how we might want to weigh at first. 

To begin, weighing an uneven amount of balls on either side clearly makes no sense. Our choices at the start are thus either 1-1, 2-2 ... 6-6. It might not be too hard to deduce that 4-4 turns out to be the best, and we can formalize why by taking an information theoretic perspective. 

Originally, there are 24 possible states of our system. 12 for whichever ball is fake, doubled by whether it is heavier or lighter. This is equivalent to saying we have about ~4.6 bits (log_2(24)) of entropy in our system. Weighing 4 balls against 4 effectively removes 1/3 of the possible states, bringing the entropy down by ~1.6 bits.

It's worth unpacking why 4-4 is the sweet spot, because the shape of the answer ends up generalising. For the first weighing with n=12 the three scale outcomes have state counts (2k, 2k, 24-4k) where k is the number of balls on each side. We want those three numbers as close to equal as possible, because a three-way toss is most informative when it's a 1-in-3-each toss. Plotting entropy reduction against k for n=12, 40, 100 gives a clean concave curve that peaks right around k = 2n/3 balls on the scale:

![First-weighing option landscape for n=12, 40, 100 — entropy reduction against the number of balls placed on the scale, red dot marks the greedy pick](/assets/images/balls_scale_initial_option_landscape.png)

What intrigued me about this problem was that the information theory perspective seems deceptively useful, as it doesn't really tell us a lot about how to solve the problem. Sure it gives us a greedy algorithm (at each step, do the weighing that reduces the number of possible states by a maximum amount), but this provides pretty minimal insight over just looking at cases, which is essentially what we are still doing.

It would be excellent if we could deduce some upper or lower bounds for the expected entropy reduced from a single weighing. In our example, if we are to keep making moves as good as the first weighing, we would be able to do the puzzle in indeed 3 weighs, as 1.6 * 3 >  total starting entropy of the system. We have no guarantee that this is possible though.

A good first step might be to empirically study the system for n balls, by writing a simple computer program. We want to model an algorithm that takes the choice that maximizes expected information gain, and then give it the worst possible outcome in terms off true information gained. We always assume that there are an equal number of balls on each side, as not doing this gives us 0 information.

The first thing we can notice is that it doesn't seem to be possible to reduce the entropy by more than log_2(3) bits at any given point in the worst case scenario, regardless of ball count. We can convince ourselves that this is true by contradiction. Suppose that the opposite was true, that there exists a weighing at some point with K possible states, such that the worst case scenario gives more than log_2(3) bits of information. This is equivalent to saying that after weighing, there are strictly less than K/3 surviving states for each outcome. Because there are three possibilities when weighing (left side down, right side down, balanced), even if all post-weigh states are mutually exclusive, they would still only be able to cover strictly less than 3*K/3 = K possible states. I.e., there must be less than K possible states to begin with, and we have a contradiction.

There's a second way to see the same fact that I like a bit better. The scale outcome O is a deterministic function of the hidden state S — once you know which ball is fake and whether it is heavy or light, the pan direction is fixed. So the information you pull out of a weighing is exactly the mutual information I(S;O) = H(O) - H(O|S) = H(O). The entropy of a three-valued random variable tops out at log_2(3), and only hits that ceiling when the three outcomes are genuinely equiprobable. Every bound we're going to get is really a statement about H(O). The greedy policy is then mechanically clear: pick the weighing whose outcome is as close as possible to a uniform three-way toss, i.e. whose three state-count piles are as close to equal as the integers allow.

Great. We now have a minimum weigh function of W_min(n) = ceil(log(2n) / log(3)), or ceil(log_3(2n)). Running the greedy against a worst-case branch for every n from 4 to 100 shows we sit on this bound almost everywhere, stepping up exactly where log_3(2n) crosses an integer:

![Greedy weighings versus n with the log_3(2n) lower bound overlaid, showing the staircase match for n=4 to 100](/assets/images/balls_scale_weighings_vs_n.png)

The per-step picture is arguably prettier. If you record how many bits the chosen weighing actually buys at each step, the first few steps of any run sit right at the log_2(3) ceiling — we are performing near-perfect ternary splits early — and only the final step falls short, because by then the residual state count isn't a multiple of 3 and a clean third-way split is impossible:

![Per-step entropy reduction for selected n, dashed line at log_2(3); early steps hug the ceiling, the last step drops](/assets/images/balls_scale_per_step_reduction.png)

There are occasional exceptions — some n need one more weigh than the bound suggests, the smallest interesting example being 13 balls. The reason is concrete: with no already-known-genuine ball to pad either pan, the first weighing can only produce state-count triples of the form (2k, 2k, 2n-4k), and for some n no k gives all three piles small enough for the remaining weighings to handle. So the lower bound isn't always achievable, but it's a remarkably good guide — the greedy hits it at all but a handful of values up to 100.
