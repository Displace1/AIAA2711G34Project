# Importance Sampling: Principles, Algorithms, and Applications

## 1. Introduction

Importance Sampling (IS) is a fundamental technique in Monte Carlo methods, widely applied in statistical physics, Bayesian inference, financial engineering, and computer graphics. When direct sampling from the target distribution is infeasible or when computing expectations presents significant challenges, importance sampling introduces a proposal distribution to transform the computational problem into a more tractable sampling problem.

This chapter first reviews foundational sampling methods, including direct sampling and rejection sampling, to establish the necessary background for understanding the motivation behind importance sampling.

## 2. Basic Sampling Methods

### 2.1 Inverse Transform Sampling

Inverse Transform Sampling is based on the probability integral transform: if $U \sim \text{Uniform}[0,1]$ and $F$ is the cumulative distribution function (CDF) of the target distribution, then $X = F^{-1}(U)$ follows the target distribution $F$.

**Theorem 2.1** (Probability Integral Transform): Let $F$ be the CDF of a continuous random variable with quantile function $F^{-1}$. Then
$$X = F^{-1}(U), \quad U \sim \text{Uniform}[0,1] \implies X \sim F$$

**Proof**: For any $x \in \mathbb{R}$,
$$P(X \leq x) = P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)$$
Thus, the distribution function of $X$ is $F$.

**Applicability**: Inverse transform sampling requires $F^{-1}$ to have an explicit form or be efficiently computable. Typical distributions include exponential, Cauchy, and logistic distributions.

**Example**: The CDF of exponential distribution $\text{Exp}(\lambda)$ is $F(x) = 1 - e^{-\lambda x}$, with inverse function
$$F^{-1}(u) = -\frac{1}{\lambda}\ln(1-u)$$

```python
import numpy as np

def sample_exponential(lam: float, n: int) -> np.ndarray:
    """
    Generate exponential distribution samples using inverse transform sampling.

    Parameters:
        lam: Rate parameter λ > 0
        n: Number of samples
    Returns:
        Array of samples following Exp(lam)
    """
    u = np.random.uniform(0, 1, n)
    return -np.log(1 - u) / lam
```

**Limitations**: Common distributions such as Gaussian lack closed-form CDF inverses, requiring numerical methods (e.g., inverse error function) or alternative sampling strategies.

### 2.2 Rejection Sampling

Rejection Sampling constructs a proposal distribution that envelopes the target distribution, applicable when the target density is known but direct sampling is difficult.

**Algorithm 2.1** (Rejection Sampling):
Given target density $p(x)$ and proposal density $q(x)$ satisfying $p(x) \leq M \cdot q(x)$ for all $x$, where $M \geq 1$ is a constant:

1. Sample candidate $x$ from $q(x)$;
2. Sample $u$ from $\text{Uniform}[0,1]$;
3. Accept $x$ if $u \leq \frac{p(x)}{M \cdot q(x)}$; otherwise reject and return to step 1.

**Theorem 2.2** (Validity of Rejection Sampling): Algorithm 2.1 outputs samples following distribution $p(x)$.

**Proof**: Let $(X, U)$ denote the joint sampling, with acceptance event $\mathcal{A} = \{U \leq p(X)/(Mq(X))\}$. For any measurable set $\mathcal{B}$,
$$P(X \in \mathcal{B} \mid \mathcal{A}) = \frac{P(X \in \mathcal{B}, U \leq p(X)/(Mq(X)))}{P(\mathcal{A})}$$

The numerator equals:
$$\int_{\mathcal{B}} \int_0^{p(x)/(Mq(x))} q(x) \, du \, dx = \int_{\mathcal{B}} \frac{p(x)}{M} \, dx$$

The denominator equals:
$$\int_{\mathbb{R}^d} \frac{p(x)}{M} \, dx = \frac{1}{M}$$

Therefore $P(X \in \mathcal{B} \mid \mathcal{A}) = \int_{\mathcal{B}} p(x) \, dx$, confirming that accepted samples follow $p(x)$.

**Acceptance Rate Analysis**: The per-iteration acceptance probability is
$$P(\text{accept}) = \frac{1}{M}$$
with expected number of iterations equal to $M$. When $M \gg 1$, algorithm efficiency degrades significantly.

**Example**: Sampling truncated normal distribution using uniform proposal

```python
from scipy.stats import norm

def truncated_normal_rejection(a: float, b: float, n: int) -> np.ndarray:
    """
    Generate truncated normal samples using rejection sampling.

    Parameters:
        a, b: Truncation interval [a, b]
        n: Number of samples
    """
    def p(x):
        if a <= x <= b:
            return norm.pdf(x) / (norm.cdf(b) - norm.cdf(a))
        return 0.0

    # Proposal: uniform distribution
    M = 1.0 / (norm.cdf(b) - norm.cdf(a)) * (b - a)

    samples = []
    while len(samples) < n:
        x = np.random.uniform(a, b)
        u = np.random.uniform(0, 1)
        if u <= p(x) * (b - a) / M:
            samples.append(x)

    return np.array(samples)
```

**Drawbacks**: When the target distribution exhibits locally high density, the constant $M$ must be large, resulting in low acceptance rates and significantly reduced sampling efficiency. This limitation motivates the development of more efficient sampling strategies—importance sampling.

## 3. Importance Sampling

### 3.1 Problem Formulation

Consider computing the expectation:
$$\mu = \mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) \, dx$$

Practical challenges include:
- Direct sampling from $p(x)$ is computationally infeasible or prohibitively expensive;
- $p(x)$ is known only up to a normalizing constant $\tilde{p}(x) = Z \cdot p(x)$, where $Z$ is unknown (e.g., posterior distributions in Bayesian inference);
- $f(x)$ takes small values in high-probability regions of $p(x)$, leading to high variance in standard Monte Carlo estimation.

Importance sampling addresses these challenges by introducing a **proposal distribution** $q(x)$, transforming the integral into an expectation with respect to $q$.

### 3.2 Fundamental Derivation

Let $q(x)$ be a proposal distribution satisfying $p(x) > 0 \implies q(x) > 0$ (support coverage condition). Rewrite the expectation:
$$\mu = \int f(x) p(x) \, dx = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$$

**Definition 3.1** (Importance Weight): The function $w(x) = \frac{p(x)}{q(x)}$ is called the importance weight or likelihood ratio.

Sampling independently $\{x_i\}_{i=1}^N$ from $q(x)$ yields the **standard importance sampling estimator**:
$$\hat{\mu}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^N f(x_i) w(x_i)$$

### 3.3 Normalized Importance Sampling

When $p(x)$ is known only up to a normalizing constant $\tilde{p}(x) = Z \cdot p(x)$, define unnormalized weights $\tilde{w}(x) = \frac{\tilde{p}(x)}{q(x)}$. Since $Z = \int \tilde{p}(x) \, dx$ is unknown, we employ the **normalized importance sampling estimator** (NIS):
$$\hat{\mu}_{\text{NIS}} = \frac{\sum_{i=1}^N f(x_i) \tilde{w}(x_i)}{\sum_{i=1}^N \tilde{w}(x_i)} = \sum_{i=1}^n f(x_i) \bar{w}_i$$
where $\bar{w}_i = \frac{\tilde{w}(x_i)}{\sum_{j=1}^N \tilde{w}(x_j)}$ are normalized weights.

### 3.4 Theoretical Properties

**Theorem 3.1** (Unbiasedness): The standard importance sampling estimator $\hat{\mu}_{\text{IS}}$ is unbiased for $\mu$.

**Proof**:
$$\mathbb{E}_q[\hat{\mu}_{\text{IS}}] = \frac{1}{N} \sum_{i=1}^N \mathbb{E}_q[f(x_i)w(x_i)] = \mathbb{E}_q[f(x)w(x)] = \int f(x)\frac{p(x)}{q(x)}q(x)\,dx = \mu$$

**Theorem 3.2** (Consistency): The normalized importance sampling estimator $\hat{\mu}_{\text{NIS}}$ converges to $\mu$ in probability, i.e., $\hat{\mu}_{\text{NIS}} \xrightarrow{P} \mu$ as $N \to \infty$.

**Proof**: By the Strong Law of Large Numbers,
$$\frac{1}{N}\sum_{i=1}^N \tilde{w}(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}_q[\tilde{w}(x)] = Z$$
$$\frac{1}{N}\sum_{i=1}^N f(x_i)\tilde{w}(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}_q[f(x)\tilde{w}(x)] = Z\mu$$
By the Continuous Mapping Theorem,
$$\hat{\mu}_{\text{NIS}} = \frac{\frac{1}{N}\sum_{i=1}^N f(x_i)\tilde{w}(x_i)}{\frac{1}{N}\sum_{i=1}^N \tilde{w}(x_i)} \xrightarrow{\text{a.s.}} \frac{Z\mu}{Z} = \mu$$
Thus $\hat{\mu}_{\text{NIS}}$ is strongly consistent, hence weakly consistent.

**Remark**: $\hat{\mu}_{\text{NIS}}$ is biased with bias of order $O(N^{-1})$, but consistency ensures reliability for large samples.

**Theorem 3.3** (Variance Analysis): The variance of the standard importance sampling estimator is
$$\text{Var}(\hat{\mu}_{\text{IS}}) = \frac{1}{N}\text{Var}_q(f(x)w(x)) = \frac{1}{N}\left(\int \frac{f^2(x)p^2(x)}{q(x)}\,dx - \mu^2\right)$$

**Optimal Proposal Distribution**: Minimizing variance is equivalent to minimizing $\int \frac{f^2(x)p^2(x)}{q(x)}\,dx$. By calculus of variations, the optimal proposal is
$$q^*(x) = \frac{|f(x)|p(x)}{\int |f(t)|p(t)\,dt}$$
yielding $\text{Var}(\hat{\mu}_{\text{IS}}) = 0$ when $f(x) \geq 0$ or $f(x) \leq 0$.

**Proof**: For any proposal $q$, consider
$$\text{Var}_q(f(x)w(x)) = \mathbb{E}_q[f^2(x)w^2(x)] - \mu^2 = \int \frac{f^2(x)p^2(x)}{q(x)}\,dx - \mu^2$$
By the Cauchy-Schwarz inequality,
$$\left(\int |f(x)|p(x)\,dx\right)^2 = \left(\int \frac{|f(x)|p(x)}{\sqrt{q(x)}} \cdot \sqrt{q(x)}\,dx\right)^2 \leq \int \frac{f^2(x)p^2(x)}{q(x)}\,dx \cdot \int q(x)\,dx = \int \frac{f^2(x)p^2(x)}{q(x)}\,dx$$
Equality holds if and only if $\frac{|f(x)|p(x)}{\sqrt{q(x)}} \propto \sqrt{q(x)}$, i.e., $q(x) \propto |f(x)|p(x)$.

### 3.5 Weight Degeneracy

In high-dimensional spaces, importance sampling often suffers from **weight degeneracy**: a small number of samples dominate the weights, causing the effective sample size (ESS) to be much smaller than the actual sample size $N$.

**Definition 3.2** (Effective Sample Size):
$$\text{ESS} = \frac{\left(\sum_{i=1}^N \bar{w}_i\right)^2}{\sum_{i=1}^N \bar{w}_i^2} = \frac{1}{\sum_{i=1}^N \bar{w}_i^2}$$

When weights are uniform, $\text{ESS} = N$; when one weight equals 1 and others equal 0, $\text{ESS} = 1$. In practice, if $\text{ESS} \ll N$, this indicates poor match between proposal and target distributions, requiring adjustment of $q(x)$ or adoption of adaptive methods.

---

## 4. Summary

This article systematically presents the mathematical foundations of importance sampling:
1. **Inverse Transform Sampling** and **Rejection Sampling** provide the foundation for understanding importance sampling;
2. **Importance Sampling** transforms expectation computation through importance weights, applicable to distributions that are difficult to sample directly;
3. **Normalized Importance Sampling** handles unnormalized target distributions with consistency but introduces small-sample bias;
4. **Optimal Proposal Distribution** $q^*(x) \propto |f(x)|p(x)$ minimizes variance;
5. **Weight Degeneracy** represents the core challenge in high-dimensional applications, requiring monitoring through effective sample size.

Importance sampling forms the basis for advanced methods such as particle filtering, sequential Monte Carlo, and variance reduction techniques, with its theoretical properties providing rigorous guarantees for practical applications.
