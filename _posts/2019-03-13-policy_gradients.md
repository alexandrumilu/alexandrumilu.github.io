---
layout: post
title: Policy Gradients
mathjax: true
---

*Draft*
In this post I will introduce reinforcement learning and one of the basic algorithms - REINFORCE. I will also show some results of my own implementation on the OpenAI provided CartPole environment. 

In reinforcement learning we are interested in training *agents* to *act* in an *environment* such that they maximize a *reward*. Formally, we model the problem as a Markov Decision Process which is composed of: a set of states $S$, a set of actions $A$, an initial state distribution $p(s_0)$, a stationary transition dynamics distribution with density $p(s_{t+1}\mid s_t,a_t)$ satisfying the Markov property, i.e. $p(s_{t+1}\mid s_t,a_t) = p(s_{t+1}\mid s_t, a_t, s_{t-1}, a_{t-1},...,s_1,a_1,s_0,a_0)$, and a reward function $r:S \times A \rightarrow \mathbb{R}$. At each step our agent will receive from the environment a state $s_t$, it will choose an action $a_t$, and it will get back from the environment the next state $s_{t+1}$ and a reward $r_t$. 
