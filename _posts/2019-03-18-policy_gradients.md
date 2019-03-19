---
layout: post
title: Policy Gradients
mathjax: true
---


In this post I will introduce reinforcement learning and one of the basic algorithms - REINFORCE. I will also show some results of my own implementation on the OpenAI provided CartPole environment. 

In reinforcement learning we are interested in training *agents* to *act* in an *environment* such that they maximize a *reward*. Formally, we model the problem as a Markov Decision Process which is composed of: a set of states $S$, a set of actions $A$, an initial state distribution $p(s_0)$, a stationary transition dynamics distribution with density $p(s_{t+1}\vert s_t,a_t)$ satisfying the Markov property, i.e. $p(s_{t+1}\vert s_t,a_t) = p(s_{t+1}\vert s_t, a_t, s_{t-1}, a_{t-1},...,s_1,a_1,s_0,a_0)$, and a reward function $r:S \times A \rightarrow \mathbb{R}$. The Markov property assumes that the future is independent of the past, given the present. At each step our agent will receive from the environment a state $s_t$, it will choose an action $a_t$, and it will get back from the environment the next state $s_{t+1}$ and a reward $r_t$. In this post I will only consider episodes that end after a finite number of steps. 

We will endow our agent with a parametrized stochastic policy $\pi_\theta$ that will take a state and map it to a probability distribution over actions. For example, for a discrete environment with $n$ actions the output will be an $n$ dimensional vector, with the $i$-th entry corresponding to the probability that the $i$-th action will be chosen. In deep reinforcement learning $\theta$ will correspond to the parameters of a neural network with input a state and output the probability distribution over actions. We’ll also define a trajectory $\tau$ to be the list of states and actions in an episode, $\tau = (s_0,a_0,s_1,a_1,…,s_T)$. The transition dynamics and our policy determine a probability distribution $p$ over trajectories. 
$$p(\tau) =p(s_0) \prod_{t=0}^T  p(s_{t+1}\vert s_t,a_t)\pi_\theta(a_t\vert s_t).$$
Let $R(\tau)$ be the total reward gathered on trajectory $\tau$. That is if $\tau = (s_0,a_0,s_1,a_1,…,s_T)$ then $$R(\tau) = \sum_{t=0}^T r(s_t,a_t).$$
 Then our reinforcement learning problem becomes to find $\theta^{*}$ that maximizes expected reward, i.e.:
$$\theta^{*} = \arg\max_{\theta}\mathbb{E}_{p(\tau)}[R(\tau)].$$

Let $J(\theta) = \mathbb{E}_{p(\tau)}[R(\tau)]$. Then our goal is to maximize $J(\theta)$. We will do this using gradient descent. For this we have to compute the gradient of $J$ with respect to $\theta$. Fortunately this can be done using a nice trick. We have that the gradient is equal to:

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{p(\tau)}[R(\tau)] = \nabla_\theta\int_{\tau}R(\tau)p(\tau)d\tau$$
 
We can move the gradient inside the integral and notice that $R(\tau)$ is not dependent on $\theta$. Thus we have:

$$\nabla_\theta J(\theta) = \int_{\tau}R(\tau)\nabla_\theta p(\tau)d\tau =\int_{\tau}R(\tau)\nabla_\theta p(\tau) \frac{p(\tau)}{p(\tau)}d\tau $$

Using the fact that $\frac{\nabla_\theta p(\tau)}{p(\tau)} = \nabla_\theta \ln p(\tau)$, we get:

$$\nabla_\theta J(\theta) = \int_{\tau}R(\tau)\nabla_\theta (\ln p(\tau) )p(\tau)d\tau  = \mathbb{E}_{p(\tau)}[ \nabla_\theta \ln p(\tau)R(\tau)]$$

The expected value can be approximated in an unbiased way just from samples. Moreover, the gradient does not depend on the transition dynamics. This is because 

$$\nabla_\theta \ln p(\tau)  = \nabla_\theta\sum_{t = 0}^T \ln p(s_{t+1}\vert s_t,a_t) + \ln\pi_\theta(a_t\vert s_t) =\nabla_\theta\sum_{t = 0}^T \ln\pi_\theta(a_t\vert s_t) $$

This gives us the following algorithm. 
>**REINFORCE Algorithm**
>1. Initialize $\theta$
>2. for $i$ from $0$ to number of episodes:  
>3.$\quad\quad$ sample trajectory $\tau_i$  
>4.$\quad\quad$ compute gradient $\nabla_\theta J(\theta) =R(\tau_i)\nabla_\theta\sum_{t = 0}^T \ln\pi_\theta(a_t\vert s_t)$
>5.$\quad\quad \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

An intuitive way to think about the algorithm is to see that it makes trajectories with high reward more likely and trajectories with low reward less likely. 

This algorithm is easy to implement, however it has very high variance. I implemented the algorithm above and tested it on the OpenAI gym CartPole environment. The CartPole looks like this: 


![CartPole](/assets/CartPole.png)


The goal is to move the cart left or right such that the pole does not fall. The agent receives a reward equal to $1$ for every timestep the pole has not fallen. The episode ends when the pole has fallen more than $15$ degrees, or the cart has moved more than $2.4$ units from center, or the agent has held it up for 200 episodes. It is a pretty easy environment, one can hard code an algorithm that keeps the pole up for 200 episodes in 10 lines of [code](https://github.com/alexandrumilu/rl/blob/master/imitation_learning/DAgger.py). This is how my implementation of REINFORCE performed on it:


![Results](/assets/PG_on_CartPole.png)

You can find my implementation in my [github](https://github.com/alexandrumilu/rl/blob/master/policy_gradient_algorithms/base_policy_gradient_agent.py). It is slightly different than the code above. It takes a gradient descent step on a batch of episodes, rather than just one trajectory. It also uses Adam and not vanilla stochastic gradient descent. Under inspection, the gradient computed by the algorithm has very high variance and this is why it takes a lot of episodes for the agent to solve the environment. I will present how to reduce this variance in a future blog post. 
