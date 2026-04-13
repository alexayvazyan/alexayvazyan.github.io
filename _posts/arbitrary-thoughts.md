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

What intrigued me about this problem was that the information theory perspective seems deceptively useful, as it doesn't really tell us a lot about how to solve the problem. Sure it gives us a greedy algorithm (at each step, do the weighing that reduces the number of possible states by a maximum amount), but this provides pretty minimal insight over just looking at cases, which is essentially what we are still doing.

It would be excellent if we could deduce some upper or lower bounds for the expected entropy reduced from a single weighing. In our example, if we are to keep making moves as good as the first weighing, we would be able to do the puzzle in indeed 3 weighs, as 1.6 * 3 >  total starting entropy of the system. We have no guarantee that this is possible though. 

A good first step might be to empirically study the system for n balls, by writing a simple computer program.
