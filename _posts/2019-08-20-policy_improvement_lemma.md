---
layout: post
title: Policy Improvement Lemma
mathjax: true
---
This will be the first post out of a series of posts on theoretical results in reinforcement learning. This post will present the proof of the policy improvement lemma. 

**Policy Improvement Lemma**

Let $\pi$ and $\pi'$ be two policies. Let $s_0$ be a state. Then:

$$V^{\pi'}(s_0) -  V^\pi (s_0) = \mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^tA^{\pi}(s_t,a_t) $$

Proof:

The proof follows from a telescopic sum argument. 

Let's expand the RHS using the definition of the advantage. 

$$\mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^tA^{\pi}(s_t,a_t) =  \mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^t (Q^\pi(s_t,a_t)-V^\pi(s_t))$$

Now using the definition of the value function we obtain:

$$RHS = \mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^t (r(s_t,a_t)+\gamma V^\pi(s_{t+1})-V^\pi(s_t))$$

Breaking it up into two sums we have that.

$$RHS = \mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^t r(s_t,a_t) + \mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^{t+1}V^\pi(s_{t+1} )- \gamma^t V^\pi(s_t)$$

The first sum is just equal to $V^{\pi'}(s_0)$ while the second sum telescopes to $-V^\pi(s_0)$.

Thus we have proved that,

$$V^{\pi'}(s_0) -  V^\pi (s_0) = \mathbb{E}_{(s_0,a_0,...)\sim \pi'}\sum_{t=0}^\infty\gamma^tA^{\pi}(s_t,a_t). $$


We will define the discounted state visitation distribution by

$$d_{s_0}^\pi(s) = (1-\gamma)\sum_{t=0}^\infty \gamma^t\mathbb{P}(s_t = s\vert s_0 = s_0),\forall s\in S.$$

Using this distribution we can rewrite every expectation over trajectories of a sum over timesteps as an expectation over the discounted state visitation distribution and the policy distribution, i.e.:

 
$$\mathbb{E}_{(s_0,a_0,...)\sim \pi}\sum_{t=0}^\infty\gamma^tf(s_t,a_t) = \frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{s_0}^\pi}\mathbb{E}_{a\sim \pi(\cdot\vert s)}f(s,a)$$

This is obtained by just rearranging the sums after writing out the expectations. Using the discounted state visitation distribution we get that:

$$V^{\pi'}(s_0) -  V^\pi (s_0) = \frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{s_0}^\pi}\mathbb{E}_{a\sim \pi'(\cdot\vert s)}A^{\pi}(s,a).$$

Notice that the low variance gradient of the policy gradient algorithm was 

$$\nabla_\theta J(\theta) =\mathbb{E}_{(s_0,a_0,...)\sim \pi}\sum_{t=0}^\infty \gamma^t \nabla_\theta\log\pi(a_t|s_t)A^\pi(s_t,a_t)$$

which we can now rewrite using the discounted state visitation distribution as:
$$\nabla_\theta J(\theta) = \frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{s_0}^\pi}\mathbb{E}_{a\sim \pi(\cdot\vert s)}\nabla_\theta\log\pi(a|s)A^\pi(s,a).$$

The equation above is recognized as the policy gradient theorem.

The policy improvement lemma is useful because it gives us a way to measure how much better a policy is compared to another. For example if for all states we have $\mathbb{E}_{a\sim \pi'(\cdot\vert s)}A^{\pi}(s,a)\geq 0$ then policy $\pi'$ is at least as good as policy $\pi$.  The lemma is used in the theoretical justifications of a few reinforcement learning algorithms such as conservative policy iteration, natural policy gradient or TRPO. 
