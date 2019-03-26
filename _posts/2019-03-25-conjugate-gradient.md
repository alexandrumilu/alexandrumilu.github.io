---
layout: post
title: Conjugate Gradient Algorithm
mathjax: true
---

This post will be about one of the algorithms that can be used to solve the equation $Ax = b$ - the conjugate gradient algorithm.

Why use this algorithm and not the classic one that we learned in linear algebra? This algorithm is good because we do not have to compute the matrix $A$ which might be expensive. It also converges quicker when $A$ has few distinct eigenvalues. 

Throughout this post $A\in \mathbb{R}^{n\times n}$ will be a symmetric positive definite matrix and $b$ will be an $n$-dimensional vector. First, notice that solving $Ax = b$ is equivalent to minimizing the following function 

$$\phi(x) = \frac{1}{2} x^TAx - b^Tx.$$

This is because $\phi(x)$ is convex and its gradient is equal to 

$$\nabla_x\phi(x) = Ax-b$$

We will define a set of not null vectors $$\{p_0,p_1,...,p_m\}$$ conjugate with respect to $A$ if 
$$p_i^TAp_j = 0,\forall i\not=j.$$

It is easy to see that the vectors are linearly independent since if we let $$\{ c_0,c_1,...,c_m \}\subset \mathbb{R}$$ s.t 

$$\sum_{i=0}^m c_ip_i = 0$$ 

and we multiply on the left by $p_j^TA$ for every $j$ we get that 
$$c_jp_j^TAp_j=0,\forall j$$ 
which is only possible if $c_j=0$ since $A$ was positive definite. 

Now, let's assume that we have $n$ conjugate vectors with respect to $A$, $$\{p_0,p_1,...,p_{n-1}\}$$. Let $x_0 \in \mathbb{R}^n$ and define the sequence $\{x_k\}$ using the following relation: $x_{k+1} = x_k + \alpha_k p_k$, where 
$$\alpha_k = \arg\min_\alpha\phi(x_k+\alpha p_k).$$

Then we can show that $$x_n = x^*$$, where $$x^*$$ is the minimum of $$\phi$$.   

First, let's find a closed form expression for $\alpha_k$. 

$$\phi(x_k+\alpha p_k) = \frac{1}{2}(x_k+\alpha p_k)^TA(x_k+\alpha p_k) - b^T(x_k+\alpha p_k)$$

By taking the derivative with respect to $\alpha$ we obtain:

$$\frac{d\phi}{d\alpha} = \alpha p_k^TAp_k+x_k^TAp_k-b^Tp_k$$
By making it equal to $0$ we get that 

$$\alpha_k = \frac{-x_k^TAp_k+b^Tp_k}{ p_k^TAp_k}.$$

Let $r_k = Ax_k-b$. Then we have 

$$\alpha_k = \frac{-r_k^Tp_k}{ p_k^TAp_k}.$$

From the definition of our sequence, we have 
$$x_n = x_0+\sum_{i=0}^{n-1}\alpha_ip_i.$$

The set of conjugate vectors $$\{p_0,p_1,...,p_{n-1}\}$$ is  a set of $n$ linearly independent vectors and thus it spans $\mathbb{R}^n$. This means there exist $$\beta_0,...,\beta_{n-1}\in \mathbb{R}^n$$ such that 
$$x^* - x_0 = \sum_{i=0}^{n-1}\beta_ip_i$$

All that is left to prove is that $\alpha_i = \beta_i,\forall i$. By multiplying on the left with $p_i^TA$ and using that $$\{p_0,p_1,...,p_{n-1}\}$$ are conjugate we get that

$$p_i^TA(x^*-x_0) = \beta_i p_i^TAp_i.$$

From the definition of our sequence we have 

$$x_i = x_0+\sum_{j=0}^{i-1}\alpha_jp_j.$$

By multiplying on the left with $p_i^TA$ we get that 
$$p_i^TAx_i = p_i^TAx_0.$$

Thus,
$$p_i^TA(x^*-x_0) =p_i^TA(x^*-x_i)  = p_i^T(b-Ax_i) = -p_i^Tr_i$$

Therefore, 

$$\beta_i = \frac{-p_i^Tr_i}{ p_i^TAp_i} = \alpha_i.$$ 

Thus, if we can find $n$ conjugate vectors we can form a sequence that converges to the optimum in at most $n$ steps. All we have to do now is figure our how to construct the conjugate vectors. 

Fortunately for us there is a simple algorithm that iteratively produces a conjugate vector, given by the following recursive relation:

$$p_{k} = -r_k+\beta_{k} p_{k-1},$$

where $\beta_{k}$ is chosen such that $p_k$ and $p_{k-1}$ are conjugate. But how do we guarantee that $p_k$ and $p_i$ for $i<k-1$ are conjugate? Magically it will be true and we will prove it. 

First, let's find a closed form solution for $\beta_{k}$. By multiplying on the left with $p_{k-1}^TA$ we get:

$$p_{k-1}^TAp_k = -p_{k-1}^TAr_k +\beta_{k}p_{k-1}^TAp_{k-1}$$

So if we choose 

$$\beta_{k} = \frac{p_{k-1}^TAr_k}{p_{k-1}^TAp_{k-1}}$$

we have that $p_k$ and $p_{k-1}$ are conjugate. Now let's prove that $p_k$ and $p_i$ are conjugate $$\forall i \in\{0,1,...,k-2\}$$. To do this, we notice that 

$$p_i^TAp_k = -p_i^TAr_k + \beta_{k} p_i^TAp_{k-1}$$

We we will prove by induction that $$\{p_0,...,p_k\}$$ are conjugate. The induction hypothesis is true for $$\{p_0,p_1\}$$ from the way we choose $\beta_1$. Assume $$\{p_0,...,p_{k-1}\}$$ are conjugate. I will prove that $$\{p_0,...,p_{k-1},p_k = -r_k+\beta_kp_{k-1}\}$$ are conjugate. From the induction hypothesis 


$$\beta_{k} p_i^TAp_{k-1} = 0$$

Thus we must prove that 

$$r_k^TAp_i=0,\forall i,0\leq i\leq k-2.$$

The way we will do this is first we prove that $r_k$ is orthogonal to $p_i$ for $$i\in\{0,1,...,k-1\}$$. And then we prove that $Ap_i$ belongs to the subspace spanned by $$i\in\{p_0,p_1,...,p_{k-1}\}$$. These two results will give us the conclusion. 

We will prove by induction over $k$ that if $$\{p_0,p_1,...,p_{k-1}\}$$ are conjugate then $r_k^Tp_i = 0$, for $i\in\{0,1,...,k-1\}$. First notice that from $$x_{k+1} = x_k+\alpha_k p_k$$, by multiplying on the left with $A$ and subtracting $b$ we get $r_{k+1} = r_k + \alpha_k Ap_k$. For $k=0$, we must prove that $r_1^Tp_0 = 0$. This is equivalent to 

$$(r_0+\alpha_0Ap_0)^Tp_0 = 0$$ 

which is true since 

$$\alpha_0 = \frac{-r_0^Tp_0}{ p_0^TAp_0}.$$

Now, assume the induction hypothesis is true for $k$ and we will prove it for $k+1$. 
First, let $$i\in\{0,1,...,k-1\}$$.


$$r_{k+1}^Tp_i = (r_k+\alpha_kAp_k)^Tp_i = r_k^Tp_i+\alpha_kp_k^TAp_i = 0$$


because $r_k^Tp_i=0$ from the induction hypothesis and $p_k^TAp_i =0$ from being conjugate vectors. Now, let $i=k$.

$$r_{k+1}^Tp_k = (r_k+\alpha_kAp_k)^Tp_k = 0$$

because 

$$\alpha_k = \frac{-r_k^Tp_k}{ p_k^TAp_k}.$$

So we have proved that if $$\{p_0,p_1,...,p_{k-1}\}$$ are conjugate then $r_k^Tp_i = 0$, for $$i\in\{0,1,...,k-1\}$$. Now we must prove that $$Ap_i \in span(p_0,...,p_{i+1}),\forall i.$$ In order to do this we will show that 

$$span(p_0,...,p_i) = span(r_0,r_1,...,r_i) = span(r_0,Ar_0,...A^ir_0).$$

We will also prove this by induction over $i$. It is easy to see it is true for $i=0$ since $p_0 = -r_0$. Assume it is true for $i$. 

We have that $p_i \in span(r_0,Ar_0,...A^ir_0)$, so $Ap_i \in span(r_0,Ar_0,...A^{i+1}r_0)$. We also have that $r_i \in span(r_0,Ar_0,...A^ir_0)\subset span(r_0,Ar_0,...A^{i+1}r_0)$ so $r_{i+1} = r_{i}+\alpha_iAp_i \in span(r_0,Ar_0,...A^{i+1}r_0)$. Therefore 


$$span(r_0,r_1,...,r_{i+1}) \subset span(r_0,Ar_0,...A^{i+1}r_0)$$


Now notice that $A^{i}r_0 \in span(p_0,...,p_i)$, so $A^{i+1}r_0 \in span(Ap_0,...,Ap_i)$. Using that $Ap_k = \frac{r_{k+1}-r_k}{\alpha_k}$ we get that $A^{i+1}r_0 \in span(p_0,p_1,...,p_{i+1})$. Therefore 


$$span(p_0,p_1,...,p_{i+1}) \supset span(r_0,Ar_0,...A^{i+1}r_0)$$

Using that $p_{i+1}=-r_{i+1}+\beta_{i+1} p_{i}$ and the induction hypothesis we also get that

$$span(p_0,p_1,...,p_{i+1}) \subset span(r_0,r_1,...r_{i+1}).$$

Thus,


$$span(p_0,...,p_{i+1}) = span(r_0,r_1,...,r_{i+1}) = span(r_0,Ar_0,...A^{i+1}r_0).$$

And we are done. To summarize, we have proved that if $\{p_0,p_1,...,p_{k-1}\}$ are conjugate then $r_k^Tp_i = 0$, for $i\in\{0,1,...,k-1\}$. We have also proved that $p_i \in span(r_0,Ar_0,...A^ir_0)$, so $Ap_i \in span(r_0,Ar_0,...A^{i+1}r_0) = span(p_0,p_1,...,p_{i+1})$. Therefore, if $i\leq k-2$ as we needed above we have that $r_k^TAp_i = 0$. Thus, $\{p_0,...,p_k\}$ are conjugate. 

Before we give the algorithm we will make a few more simplifications to the calculations of $\alpha_k$ and $\beta_k$. We have that $0 = \beta_k r_k^Tp_{k-1} = r_k^T(p_k+r_k)$. So we can replace 

$$\alpha_k = \frac{-r_k^Tp_k}{ p_k^TAp_k}$$

with
 
$$\alpha_k = \frac{r_k^Tr_k}{ p_k^TAp_k}.$$ 

For $\beta_k$ first notice that $r_k = \beta_kp_{k-1}-p_k$, so because $r_{k+1}$ is orthogonal to the RHS $r_{k+1}$ is orthogonal to $r_k$. Using this we have:

$$ \beta_{k+1} = \frac{p_{k}^TAr_{k+1}}{p_{k}^TAp_{k}} = \frac{r_{k+1}^TAp_k}{p_{k}^TAp_{k}} =\frac{r_{k+1}^T(\frac{r_{k+1}-r_k}{\alpha_k})}{p_{k}^TAp_{k}} =  \frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k}$$


Thus, we get the following algorithm:
>**Conjugate Gradient Algorithm**
1. Initialize $x_0,r_0 = Ax_0-b, p_0 = -r_0$
2. repeat: 
3. $$\alpha_k \leftarrow \frac{r_k^Tr_k}{p_k^TAp_k} $$
4. $$x_{k+1} \leftarrow x_k + \alpha_k p_k$$
5. $$r_{k+1} \leftarrow r_k + \alpha_k Ap_k$$
6. $$\beta_{k+1} \leftarrow \frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k}$$
7. $$p_{k+1} \leftarrow -r_{k+1}+\beta_{k+1} p_{k}$$
8. $$k\leftarrow k+1$$
9. until $r_k = 0$


 In this post I introduced the conjugate gradient algorithm and I will write about one of its applications in a future post about one of the more popular reinforcement learning algorithms - TRPO. 
