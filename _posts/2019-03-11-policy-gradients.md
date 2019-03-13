---
layout: post
title: Entropy and KL-divergence
mathjax: true
---

In this post I will present a short introduction to the information-theoretical concepts of entropy and KL-divergence. 

The entropy of a random variable X with probability distribution $p(x)$ is defined as 
$$H(p) = \mathbb{E}\log(\frac{1}{p(x)})$$.

The logarithm above is the logarithm of base 2. Sometimes the entropy is defined using the natural logarithm. The definitions are equivalent under multiplication by a constant. 

As an example consider the Bernoullui random variable X that can take values 0,1 with probabilities $p(0) = p$ and $p(1) = 1-p$. The entropy of X, $H = -p\log(p) - (1-p)\log(1-p)$. If we graph the entorpy of a function of $p$ it would look like: 


We can see that the more "random" the random variable is the higher its entropy. One can think about it in the following way. The harder to predict the random variable is, the higher its entropy, or the larger the entropy, the less a priori information one has on the value of the random variable. 

The next concept we will discuss is the KL-divergence. The KL-divergence between two probability distributions $p(x)$ and $q(x)$ over the same space is defined as 
$$D(p||q) = \mathbb{E}_{p(x)}\log(\frac{p(x)}{q(x)})$$.

KL-divergence is used to characterize how different two distributions are. We'll show that the KL-divergence is greater than or equal to 0 with equality only if $p=q$ almost everywhere. We know from Jensen's inequality that if $\Phi$ is a convex function then $\Phi(\mathbb{E}(X)) \leq \mathbb{E}(\Phi(X))$. $\log(x)$ is a concave function (the second derivative is $-\frac{1}{x^2}$. So Jensen's inequality is inverted. Thus we have:
$$-D(p||q) = \mathbb{E}_{p(x)}\log(\frac{q(x)}{p(x)}) \leq \log(\mathbb{E}_{p(x)}\frac{q(x)}{p(x)} = \log(1) = 0$$

Thus, the KL-divergence looks like a distance between two probability distributions, altho it is not symmetric. 

One can also notice that we can apply Jensen's inequality directly to $H(p)$ and obtain that $H(p)\geq 0$ with equality only if X is certain (has one outcome with probability 1). 

Let's look at discrete distributions over a set of $n$ elements. We can show that the entropy is maximized when every element is equally probable. To show this let, p(x) be the uniform distribution over the set of $n$ elements and let q be another distribution. Consider $D(q||p)$. We know it is greater than or equal to 0. We'll also show that when $p$ is uniform it is just equal to $H(p) - H(q)$. 
$$D(q||p) = \sum_{i=1}^n q(i)\log\frac{q(i)}{p(i)} = -H(q) - \sum_{i=1}^n q(i)\log p(i)$$
However, $p(i) = \frac{1}{n} ,\forall i$, so
$$\sum_{i=1}^n q(i)\log p(i) = \log p(i) \sum_{i=1}^n q(i) = \log p(i) = \log p(i) \sum_{i=1}^n p(i) = -H(p) $$

Thus, $0 \leq D(q||p) = H(p) - H(q)$. So entropy is maximized by the uniform distribution.

Now we will look at when entropy is maximized among continous distributions with equal variance. The distribution with the highest entropy will again be an usual suspect - the normal distribution. The argument will be analogous with the one above. Let $p$ be the Gaussian with variance $\sigma$ and let $q$ be another distribution with the same variance. We can assume $q$ has the same mean $\mu$ as $p$ since entropy stays constant under translations. 

$$D(q||p) = \mathbb{E}_{q(x)}\log(\frac{q(x)}{p(x)}) = -H(q) - \mathbb{E}_{q(x)}\log p(x) $$

The second term on the RHS is equal to:
$$\mathbb{E}_{q(x)}\log p(x) = \int_{-\inf}^{\inf} q(x) \log (\frac{1}{\sqrt(2\pi\sigma^2)} e^(-\frac{(x-\mu)^2}{2\sigma^2})dx$$
$$\mathbb{E}_{q(x)}\log p(x) = \int_{-\inf}^{\inf} q(x) \log (\frac{1}{\sqrt(2\pi\sigma^2)}) dx - \log(e) \int_{-\inf}^{\inf}q(x) \frac{(x-\mu)^2}{2\sigma^2})dx$$

One can see that the second term on the RHS is 
$$\frac{1}{2\sigma^2} \mathbb{E}_{q(x)}(X-\mu)^2 = \frac{1}{2\sigma^2}Var(q(x)) = \frac{1}{2\sigma^2}Var(p(x))$$. 
The first term on the RHS is 
$$\log (\frac{1}{\sqrt(2\pi\sigma^2)}) \int_{-\inf}^{\inf} q(x) dx = \log (\frac{1}{\sqrt(2\pi\sigma^2)}) = \log (\frac{1}{\sqrt(2\pi\sigma^2)}) \int_{-\inf}^{\inf} p(x) dx $$

Thus, 
$$\mathbb{E}_{q(x)}\log p(x) = \mathbb{E}_{p(x)}\log p(x) = -H(p)$$

Again we have $0 \leq D(q||p) = H(p) - H(q)$. So entropy is maximized by the Gaussian distribution.
