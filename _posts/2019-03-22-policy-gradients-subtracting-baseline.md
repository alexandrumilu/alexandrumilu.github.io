﻿We have seen in the previous [post](https://alexandrumilu.github.io/2019/03/21/Policy-Gradients-Reducing-Variance/) how to reduce variance of the policy gradient algorithm by only considering the "reward-to-go". In this post I will show we can further reduce variance by subtracting a baseline that is a function of the state. 

We saw earlier that 

$$\nabla_\theta J(\theta) =\mathbb{E}_{p(\tau)} \sum_{t=0}^T \nabla_\theta \ln \pi(a_t\vert s_t)\sum_{t'=t}^Tr(s_{t'},a_{t'}).$$

We will show in this post that the following is true for any $b:S\rightarrow \mathbb{R}$:

$$\nabla_\theta J(\theta) =\mathbb{E}_{p(\tau)} \sum_{t=0}^T \nabla_\theta \ln \pi(a_t\vert s_t)[\sum_{t'=t}^Tr(s_{t'},a_{t'})-b(s_t)]$$

To show this, we must prove that

$$\mathbb{E}_{p(\tau)} \sum_{t=0}^T \nabla_\theta \ln \pi(a_t\vert s_t)b(s_t)=0$$

We will actually prove that 
$$f(s_t) = \mathbb{E}_{p(\tau)}  \nabla_\theta \ln \pi(a_t\vert s_t)b(s_t)=0,\forall t\in\{0,1,...,T-1\}$$
We will use the same tactic we had when we proved that we can just sum up the rewards to go, that is, we will divide the trajectory into 2: $\tau_t = (s_0,a_0,...s_t)$ the trajectory until time step $t$ and $\tau_t' = (a_t,s_{t+1},...,s_T)$ - the trajectory after time $t$. 

$$f(s_t) = \int_{\tau}\nabla_\theta(\ln\pi(a_t\vert s_t))b(s_t)p(\tau).$$

By dividing the sum up into the two trajectories we get 

$$f(s_t) = \int_{\tau_t,\tau_t'}\nabla_\theta(\ln\pi(a_t\vert s_t))b(s_t)p(\tau_t)p(\tau_t'\vert \tau_t).$$

By grouping all the terms that do not appear in $\tau_t'$ we get

$$f(s_t) = \int_{\tau_t}b(s_t)p(\tau_t)\int_{\tau_t'}\nabla_\theta(\ln\pi(a_t\vert s_t))p(\tau_t'\vert \tau_t).$$

Now let's take a closer look at 
$$g(s_t) =  \int_{\tau_t'}\nabla_\theta(\ln\pi(a_t\vert s_t))p(\tau_t'\vert \tau_t).$$

Notice that because of the Markov property $p(\tau_t'\vert \tau_t) = p(\tau_t'\vert s_t)$. Now divide the sum into two parts, one that contains $a_t$ and one that does not. 

$$g(s_t) =  \int_{a_t}\nabla_\theta(\ln\pi(a_t\vert s_t))p(a_t\vert s_t) \int_{(s_{t+1},...,s_T)}p(s_{t+1},...,s_T\vert a_t,s_t)$$

One can notice that the second integral is equal to $1$ and that $p(a_t\vert s_t) = \pi(a_t|s_t)$. We are left with 

$$g(s_t) =  \int_{a_t}\nabla_\theta(\ln\pi(a_t\vert s_t))\pi(a_t\vert s_t). $$

But this is basically the inverse of the "log trick". 

$$ g(s_t) =   \int_{a_t}\frac{\nabla_\theta(\pi(a_t\vert s_t))}{\pi(a_t\vert s_t)}\pi(a_t\vert s_t) = \nabla_\theta(\int_{a_t}\pi(a_t\vert s_t)) = \nabla_\theta(1) = 0$$

Thus, $f(s_t)=0$. So we have proved that we can subtract a baseline function of the state without changing the value of the gradient. What would be the best function to subtract? Intuitively, the policy gradient algorithm was making actions that gave high reward more likely than ones that gave low reward. So, a good baseline seems to be to choose the average reward of the actions when in a particular state. For example if a state is particularly good, that is all actions from that state have high reward, it seems good to penalize that state by subtracting the average **value** of that state. 

Formally, let's define the **value** of the state to be 

$$V_\pi(s_t) = \mathbb{E}_{\tau_\pi} (\sum_{t'=t}^T r(s_t,a_t))$$

where the expected value is taken on policies generated by following policy $\pi$ with parameter $\theta$. We will try to fit a function $V_\phi:S\rightarrow \mathbb{R}$ with parameters $\phi$ to the reward to go from a state, i.e. find $\phi^*$ s.t. 

$$\phi^* = \arg\min_\phi \sum_{t=0}^T( R_t-V_\phi(s_t))^2$$

Usually $\phi$ will be parameters of a neural network. 

Using this, we get the following algorithm which we will call the Vanilla Policy Gradient. 
>**Vanilla Policy Gradient Algorithm**
1. Initialize $\theta,\phi$
2. while still learning: 
3. $\quad\quad$ sample $m$ trajectories $\tau$
4. $\quad\quad$ for each time-step in each trajectory compute reward to go 
$$R_t = \sum_{t'=t}^T r(s_t,a_t)$$
5. $\quad\quad$ fit $\phi$ to  the function below by taking some number of gradient steps
 $$\phi = \arg\min_\phi \frac{1}{m}\sum_{i=1}^m\sum_{t=0}^T( R_t-V_\phi(s_t))^2$$
6.  $\quad\quad$ update $\theta$ using the gradient
$$\nabla_\theta J(\theta) =\frac{1}{m}\sum_{i=1}^m(\sum_{t=0}^T \nabla_\theta \ln \pi(a_t\vert s_t)[\sum_{t'=t}^Tr(s_{t'},a_{t'})-V_\phi(s_t)])$$

My implementation of the algorithm above is [here](https://github.com/alexandrumilu/rl/blob/master/policy_gradient_algorithms/value_policy_gradient_agent.py). Running it on the CartPole environment we get the following results which are better than the results using total rewards and using "reward-to-go". 

![VPG results](/assets/pgvalue.png)

This algorithm still has some downsides. It is on-policy - the agent needs sample trajectories from the current policy in order to improve the policy. Also, there is no guarantee in the current form of the algorithm that the new policy - the one after the gradient update - will be better than the old one. In future blog posts I will discuss some approaches to guarantee improvement. 