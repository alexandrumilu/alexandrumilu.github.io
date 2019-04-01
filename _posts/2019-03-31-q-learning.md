In this post I will introduce the Q-Learning algorithm and its extension to large state spaces - Deep Q-Learning (DQN)

The structure of this post will be as follows. First I will introduce Q-values, then introduce the Q-learning algorithm, argue it converges for small state spaces, then present how researchers extended it to large state spaces giving us DQN. 

Q-learning is an algorithm that solves the reinforcement learning problem when one does not know the transition dynamics. 

Similarly to how we introduced value functions in the previous [post](https://alexandrumilu.github.io/2019/03/30/value-iteration/) we can define the function $$Q^\pi:S \times A \rightarrow \mathbb{R}$$ for a policy $\pi$ that would take in a state and an action and return the expected return after taking that action in that state and following policy $\pi$ afterwards. 

$$Q^\pi(s,a) = r(s,a) + \sum_{s' \in S} p(s'\vert s,a)V^\pi(s')$$

Let $$Q^*$$ be the Q-value of the optimal policy. The goal of the Q-learning algorithm is to learn $$Q^*$$. The optimal policy can then be recovered by taking $$\pi^*(s) = \arg\max_{a\in A} Q^*(s,a).$$

For small environments with few states and actions we can just store every value $$Q(s,a)$$ in a $$\vert S\vert \times \vert A \vert $$ table

The Q-learning is the following:

>**Q-learning Algorithm**
1. initialize $Q_0(s,a),\forall s,a.$
2. for step number $n:0,1,2,...$ do:
3. $\quad\quad$ from state $s_n$ perform action $a_n$ observe state $s_{n+1}$, reward $r_n$.
4. $\quad\quad Q_{n+1}(s_n,a_n) \leftarrow (1-\alpha_n)Q_{n+1}(s_n,a_n) + \alpha_n(r_n+\max_aQ_n(s_{n+1},a))$



This simple algorithm can be showed to converge when $Q_n$ can be stored in a table (not too many states or actions) under reasonable conditions. 

However, when the state space is too big and $Q_n$ cannot be stored in a table and we have to use some kind of function approximation we have to make some changes. From now on we will consider Q-functions parametrized by $\theta$, which in deep learning would be the parameters of a neural network. We will also use the following notation 
$$T(s_n,a_n) = r(s_n,a_n) + \max_aQ_n(s_{n+1},a).$$

The simplest way to adapt the tabular Q-learning algorithm shown above would be to just replace step 4 by 

$$\theta_{n+1} = \theta_n - \alpha \nabla_\theta[(T(s_n,a_n) - Q_\theta(s_n,a_n))^2] $$

so we would change $$\theta$$ in the direction of the gradient of the loss function 

$$L = (T(s_n,a_n) - Q_\theta(s_n,a_n))^2.$$

This makes sense since by learning we want to make $Q_\theta$ look more like the actual reward we obtained plus the value of the next state. The gradient step I wrote above looks a lot until regression until one notices that $T$ also depends on $\theta$. However, we compute the gradient by considering $T$ to be a constant. So, when we change $\theta$ we also change the target value $T$ and more specifically $Q_\theta(s_{n+1},a)$. Let's say action $a_n$ was a good one and we got a high reward. $\theta$ will change such that $Q_\theta(s_{n},a_n)$ increases. However neural networks are smooth and since $s_{n+1}$ is the state after $s_n$ it is reasonable to assume that they are similar. So $Q_\theta(s_{n+1},a_n)$ will increase. Next time we are in the state $s_n$ and we pick the action $a_n$, again we get the same high reward, but $Q_\theta(s_{n+1},a_n)$ has also increased so the gradient update will increase $Q(s_n,a_n)$ and $Q_\theta(s_{n+1},a_n)$ even more. Thus we end up chasing our own tail and failing to compute actual q-values. To mitigate this effect we can do the following.

We will have a target network with parameters $\theta'$ that only copies the real network's parameters $\theta$ every so often. In the original paper they set $$\theta' \leftarrow \theta$$ every $$c$$ steps but one can also do Polyak averaging where 

$$\theta' \leftarrow \beta \theta + (1-\beta)\theta' $$

for some small fixed $\beta$. 

Using this target network our regression targets will be more stable and we will mitigate the chasing our tail effect explained above.  

Another change to the classic Q-learning they made in the DQN [paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) was to store every set $$b_n = (s_n,a_n,s_{n+1},r_n)$$ in a replay buffer $$B = \{b_1,...,b_K\}$$ and then take the gradient step on a batch of $m$ samples randomly chosen from the replay buffer. This was another step to making the new step 4 look more like classic regression. Without the replay buffer the samples that were fed into the regression would have high correlation given that they were collected as consecutive steps of the agent acting in the environment. Regression works under the assumption that the samples are chosen i.i.d. from the sample distribution. That assumption was violated by collecting samples online. However, by having a large replay buffer we are taking gradient steps on not as correlated samples. 

Using the two changes explained above we get the following algorithm:

>**Deep Q-Learning Algorithm**
1. initialize $\theta,\theta' = \theta,$ replay buffer $B$
2. for step number $n:0,1,2,...$ do:
3. $\quad\quad$ from state $s_n$ perform action $a_n$ observe state $s_{n+1}$, reward $r_n$ and boolean $d_n$ and add them to the replay buffer
4. $\quad\quad$ sample $m$ times from the replay buffer
5. $\quad\quad$ compute targets $T_i = r_i+(1-d_i)\max_aQ_{\theta'}(s_{i+1},a)$
6. $\quad\quad \theta \leftarrow \theta - \nabla_\theta((T_i-Q_\theta(s_i,a_i))^2)$
7. $\quad\quad$update $\theta'$ using any method presented above 

In the algorithm above $d_n$ is a boolean that is equal to True if the state was terminal and False otherwise. I introduced it to make it by default that $T_i = r_i$ when the state is terminal. 

There are some improvements that can be made to the classic DQN algorithm. The chasing our tail effect can be mitigated even more by using a technique called [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf). Also, one can use a replay buffer that makes more ["important" transitions](https://arxiv.org/pdf/1511.05952.pdf) more likely to be sampled under different definitions of "important". 
