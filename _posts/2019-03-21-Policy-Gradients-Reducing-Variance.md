---
layout: post
title: Policy Gradients - Reward-to-Go
mathjax: true
---
 
 In this post I will introduce how to compute gradients of lower variance in the REINFORCE algorithm. 

In the previous post we have computed that the gradient of the reward function with respect to the parameters is:

$$\nabla_\theta J(\theta)  = \mathbb{E}_{p(\tau)}[ \nabla_\theta \ln p(\tau)R(\tau)]\approx \frac{1}{N}\sum_{i=1}^N R(\tau_i)\nabla_\theta \ln p(\tau_i).$$

We can substitute
$$R(\tau_i) = \sum_{t=0}^T r(s_t,a_t)$$ 

and 

$$\nabla_\theta \ln p(\tau_i) = \sum_{t=0}^T \nabla_\theta \ln\pi(a_t\vert s_t).$$

We can write the second equation because the transition dynamics are independent of $\theta$.

We get
$$\nabla_\theta J(\theta)  \approx \frac{1}{N}\sum_{i=1}^N \big( \sum_{t=0}^T r(s_t,a_t)\big) \big(\sum_{t=0}^T \nabla_\theta \ln\pi(a_t\vert s_t)\big)$$

The first way to reduce variance will be to show that we do not have to sum rewards from the beginning to the end, but just from the current time step to end, i.e.:
$$\nabla_\theta J(\theta)  \approx \frac{1}{N}\sum_{i=1}^N  \big(\sum_{t=0}^T \nabla_\theta \ln\pi(a_t\vert s_t)\big( \sum_{t'=t}^T r(s_{t'},a_{t'})\big)\big)$$

It is easy to see that by summing up fewer numbers the gradient will be smaller and thus have smaller variance. We can prove the equation above by just being slightly more careful in the way we compute the gradient.

We have $J(\theta) = \mathbb{E}_{p(\tau)} R(\tau)$. By linearity of expectation, this is equal to $J(\theta) = \sum_{t=0}^T\mathbb{E}_{p(\tau)} r(s_t,a_t)$.  Let $\tau_t$ be the trajectory until time step $t$ and $\tau'_t$ be the trajectory after $t$. Thus,

$$J(\theta) = \sum_{t=0}^T\mathbb{E}_{p(\tau_t)p(\tau_t'\vert\tau_t)} r(s_t,a_t) = \mathbb{E}_{p(\tau_t)}r(s_t,a_t)\mathbb{E}_{p(\tau_t'\vert\tau_t)} (1)$$ 

Thus,

$$J(\theta) = \sum_{t=0}^T\mathbb{E}_{p(\tau_t)}r(s_t,a_t).$$
When taking the gradient, and after applying the log trick as before the transition dynamics cancel and we will be left with what we need.

$$\nabla_\theta J(\theta) = \sum_{t=0}^T\nabla_\theta\mathbb{E}_{p(\tau_t)}r(s_t,a_t)$$
By writing out the expectation we have:
$$\nabla_\theta J(\theta) = \sum_{t=0}^T\nabla_\theta\int_{\tau_t}r(s_t,a_t)p(\tau_t) d\tau_t=\sum_{t=0}^T\int_{\tau_t}r(s_t,a_t)\nabla_\theta p(\tau_t) d\tau_t$$

Applying the log trick as before we get:

$$\nabla_\theta J(\theta) = \sum_{t=0}^T\int_{\tau_t}r(s_t,a_t)\nabla_\theta \ln p(\tau_t) p(\tau_t) d\tau_t$$

We can multiply every integral by $1 = \int_{\tau_t'}p(\tau_t'\vert\tau_t)d\tau_t'$ and we get:

$$\nabla_\theta J(\theta) = \sum_{t=0}^T\int_{\tau}r(s_t,a_t)\nabla_\theta \ln p(\tau_t) p(\tau) d\tau = \sum_{t=0}^T\mathbb{E}_{p(\tau)}r(s_t,a_t)\nabla_\theta \ln p(\tau_t)$$

By getting rid of the transition dynamics which are independent of our policy and by rearranging the sum we get:

$$\nabla_\theta J(\theta) =\mathbb{E}_{p(\tau)} \sum_{t=0}^T \nabla_\theta \ln \pi(a_t\vert s_t)\sum_{t'=t}^Tr(s_{t'},a_{t'})$$

which is exactly what we set out to obtain. 

I implemented this "reward-to-go" policy gradient algorithm [here](https://github.com/alexandrumilu/rl/blob/master/policy_gradient_algorithms/policy_gradient_reward_to_go_agent.py). These are the results after running on the CartPole environment. 

![Results](/assets/PG_rtg_on_CartPole.png)


We can see it does much better than the previous version. 


![Results](/assets/PG_on_CartPole.png)

In a future blog post, I will show how we can reduce variance further by subtracting a function of the state from the sum of "rewards to go".
