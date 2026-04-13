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

## 2 - 
