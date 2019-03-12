---
layout: post
title: Policy Gradients
mathjax: true
---

Test...$$n=1$$
2.
$\nabla_{\theta}J(\theta)=(\sum_{t=0}^T\nabla_{\theta}\ln(\pi_\theta(a_t|s_t))R(\tau)$
>**REINFORCE**
1. Initialize $\theta$
2. **for** each episode $i$ **do**
3. $\quad\quad$ sample trajectory ${\tau_i}$ using $\pi_\theta$
4. $\quad\quad$ $\nabla_{\theta}J(\theta)=\sum_{t=0}^T\big(\nabla_{\theta}\ln(\pi_\theta(a_t|s_t)\big)R(\tau)$
5. $\quad\quad \theta \leftarrow \theta + \alpha\nabla_{\theta}J(\theta)$ 
