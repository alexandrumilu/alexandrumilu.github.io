---
layout: post
title: Value Iteration Algorithm
mathjax: true
---

In this post I will present the Value Iteration algorithm and prove that it works using Banach's fixed point theorem.

The structure of the post will be as follows. First I will define the Value function of a policy, then introduce the algorithms, then introduce Banach's fixed point theorem, then prove that the algorithms converge using Banach's fixed point theorem. 

In reinforcement learning we are modeling the problem using a Markov Decision Process that is composed of: a set of states $S$, a set of actions $A$, a stationary transition dynamics distribution with density $p(s_{t+1}\vert s_t,a_t)$ satisfying the Markov property and a reward function $r:S \times A \rightarrow \mathbb{R}$. 

In this post we will consider only stationary deterministic policies $\pi:S\rightarrow A,$ which take in a state and choose an action according to some learned rule. For the math to work out easier we will also add a discount factor $\gamma$ to our model that discounts any reward obtained in the future (a reward $r$ obtained $t$ steps into the future will have value $\gamma^t r$. One can think about this discount factor as $1$ minus probability of dying at each time step. We will also consider MDPs that have an infinite number of steps. One can take an MDP that has a finite number of states and add a terminal state with reward $0$ to it. We will also consider only MDP's where $S$ and $A$ are finite sets, in order to simplify the exposition. The results can be proved if $S$ and $A$ are infinite too. 

We will define a function $$f:\mathbb{R}^n \rightarrow \mathbb{R}^n$$ to be a contraction under norm $$\|\cdot \|$$ if there exists $L \in (0,1)$ such that

$$  \| f(x) - f(y)\|\leq L\| x - y\|,\forall x,y \in \mathbb{R}^n.$$


We will define the value function of a policy $\pi$, $V^\pi : S \rightarrow \mathbb{R}$ as:

$$V^\pi(s) = \mathbb{E}(\sum_{t=0}^\infty \gamma^tr(s_t,\pi(s_t))\vert s_0 = s)  $$

In words the value of a state $s$ under a policy $\pi$ is the expected reward obtained from that state when using policy $\pi$ to choose actions. 

One can see that the following set of equations (called the Bellman equations) holds for every $s \in S$:

$$V^\pi(s) = r(s,\pi(s)) + \gamma \sum_{s'} p(s'\vert s,\pi(s))V^\pi(s'). $$

Define the Bellman operator $T^\pi$ by:
$$T^\pi (V(s)) = r(s,\pi(s)) + \gamma \sum_{s'} p(s'\vert s,\pi(s))V(s'). $$

Then the Bellman equations can be succinctly written as:

$$T^\pi V^\pi = V^\pi.$$ 

We can compute the value of a policy as follows. Start with a random value function $V_0$. Define the sequence $\{V_n\}$ recursively by $V_n = T^\pi V_{n-1}$. We will prove that $T^\pi$ is a maximum norm contraction and using Banach's fixed point theorem this means that the sequence $\{V_n\}$ converges to the unique solution $$V^\pi$$  - the true value of the policy. 

Let $$V^*$$ be the optimal value function - that is the value function corresponding to the optimal policy. One can see that $$V^*$$ satisfies the following set of equations (Bellman optimality equations):

$$V^*(s) = \max_{a\in A} (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)V^*(s')),\forall s \in S.$$

As above we will define the operator $T^*$ by:

$$T^*(V(s)) = \max_{a\in A} (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)V(s')),\forall s \in S.$$

So the Bellman optimality equations can be written as 

$$T^*V^* = V^*.$$

The value iteration algorithm is similar to the algorithm of computing the value of the policy presented above. Start with a value function $V_0$  and a policy determined using the formula below. Then at each step compute $V_{n} = T^*V_{n-1}$ and 

$$\pi_n(s) = \arg\max_{a\in A} (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)V_n(s')),\forall s \in S $$

We will show that $V_n$ converges to the value function of the optimal policy and thus the policy converges to the optimal policy. 

**Banach's Fixed Point Theorem**

Let $f:\mathbb{R}^n \rightarrow \mathbb{R}^n$ such that 

$$  \| f(x) - f(y)\|\leq L\| x - y\|,\forall x,y \in \mathbb{R}^n $$

where $L \in (0,1)$ and $\|\cdot \|$ is a norm on $\mathbb{R}^n$. Then there exists a unique $x^* \in \mathbb{R}^n$ such that $f(x^*) = x^*$. We will call $x^*$ a fixed point of $f$. Furthermore, for all $x_0\in \mathbb{R}^n$ define a sequence $\{x_n\}$ recursively as $x_{n+1} = f(x_n)$. Then, we have that $\{x_n\}$ converges to $x^*$. 

*Proof:* First notice that we cannot have two fixed points $x^*$ and $y^*$. Assume that we could. Then we would have 

$$ \| x^* - y^*\| =  \| f(x^*) - f(y^*)\|\leq L\| x^* - y^*\|$$

which is impossible since $L <1$ and the norm is positive if $x^* \not=y^*$. 

Let $x_0\in \mathbb{R}^n$ and $\{x_n\}$ be a sequence s.t. $x_{n+1} = f(x_n),\forall n \in \mathbb{N}.$ Let's take a look at $\|x_m - x_n\|,m>n$. 

$$\|x_m - x_n\| = \|f(x_{m-1}) - f(x_{n-1})\|\leq L\|x_{m-1} - x_{n-1}\|$$

By applying the same inequality $n$ times we get 

$$\|x_m - x_n\|\leq L^{n}\|x_{m-n}-x_0\|$$

From the triangle inequality we have:

$$\|x_{m-n}-x_0\|\leq \|x_{m-n}-x_{m-n-1}\|+\|x_{m-n-1}-x_{m-n-2}\|+...+\|x_{1}-x_0\|$$

By applying the inequality above to each of the terms on the RHS we obtain:
$$\|x_{m-n}-x_0\|\leq (\sum_{i=0}^{m-n-1} L^i)\|x_{1}-x_0\|\leq \frac{1}{1-L}\|x_{1}-x_0\|$$

Thus, 

$$\|x_m - x_n\|\leq L^{n}\frac{1}{1-L}\|x_{1}-x_0\|$$

The RHS goes to $0$ as $n$ goes to infinity and the LHS is non-negative. Thus $\{x_n\}$ is a Cauchy sequence. In the Euclidian space Cauchy sequences are convergent. Let $x^*$ be the limit. All we have to do now is show that $f(x^*) = x^*$. This follows from the following:

$$\|f(x^*)-x^*\|\leq\|f(x^*)-f(x_n)\|+\|f(x_n) - x^*\|\leq L\|x^*-x_n\|+\|x_{n+1}-x^*\|$$

Because the RHS converges to $0$ as $n$ goes to infinity and the LHS is non-negative we obtain that $f(x^*) = x^*$ and our proof is complete. 

Let $\pi$ be a policy and let $T^\pi$ be the Bellman operator associated with that policy. We are assuming that the state space is finite, thus $T^{\pi}:\mathbb{R}^{\vert S\vert}\rightarrow\mathbb{R}^{\vert S\vert}$. We will show that $T^\pi$ is a contraction under the infinity norm defined as 
$$\|V\|_{\infty} = \max_{s\in S}V(s)$$

We have 

$$\|T^\pi V-T^\pi U\|_{\infty} = \max_{s\in S}  \gamma \sum_{s'} p(s'\vert s,\pi(s))(V(s')-U(s')) $$

We can bound 

$$V(s') - U(s')\leq \max_{s'\in S} (V(s')-U(s')).$$

Therefore we get

 $$\|T^\pi V-T^\pi U\|_{\infty} \leq \max_{s\in S}  \gamma \sum_{s'} p(s'\vert s,\pi(s))\max_{s'\in S} (V(s')-U(s')).$$

The $\max$ comes in front of the sum and $$\sum_{s'} p(s'\vert s,\pi(s))=1.$$

Thus, we obtain

$$\|T^\pi V-T^\pi U\|_{\infty}\leq \gamma \max_{s'\in S} (V(s')-U(s')) = \gamma \|V-U\|_\infty$$

So we have proved $T^\pi$ is a contraction. We know that 
$$T^\pi V^\pi = V^\pi.$$

Therefore, according to Banach's fixed point theorem, by starting with a random value function $V_0$ and at each step computing $V_{n+1} = T^\pi V_{n}$ we obtain the true value function $V^\pi$.

Now we will prove that $T^*$ is a contraction under the same norm and thus prove the validity of the value iteration algorithm. To prove that $T^*$ is a contraction under the infinity norm we follow the same strategy as above. 

$$\|T^* V-T^* U\|_{\infty} = \max_{s\in S}(\max_{a\in A} (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)V(s'))-\max_{a\in A} (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)U(s'))$$

However, the difference of two maximums is less than the maximum of the difference, thus

$$\|T^* V-T^* U\|_{\infty} \leq \max_{s\in S,a\in A} (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)V(s'))- (r(s,a)+\gamma \sum_{s'} p(s'\vert s,a)U(s'))$$

Cancelling the reward term, we get:
$$\|T^* V-T^* U\|_{\infty} \leq \max_{s\in S,a\in A} \gamma \sum_{s'} p(s'\vert s,a)(V(s')-U(s'))$$

As above, we get:

$$\|T^* V-T^* U\|_{\infty} \leq \max_{s\in S,a\in A} \gamma \max_{s' \in S}(V(s')-U(s')) = \gamma \|V-U\|_\infty$$

Thus, $T^*$ is a contraction. Therefore the value iteration algorithm converges. 

In this post we saw how it is possible to provably solve a reinforcement learning problem when we know the transition dynamics. However, when the state and action spaces are large this method will not work. It will obviously also fail when we do not know the transition dynamics.   
