 我来系统介绍机器学习中常用的采样方法，从基础到高级，涵盖不同应用场景。

---

## 一、基础采样方法

### 1. 直接采样（Direct Sampling）

**原理**：利用均匀分布 $U[0,1]$ 通过变换得到目标分布。

#### 逆变换采样（Inverse Transform Sampling）

**数学基础**：若 $F$ 是目标分布的CDF，$F^{-1}$ 是其逆函数，则
$$X = F^{-1}(U), \quad U \sim \text{Uniform}[0,1] \implies X \sim F$$

**适用**：CDF易求逆的分布（指数、柯西、逻辑斯蒂等）

```python
import numpy as np

# 指数分布采样：F(x) = 1 - e^(-λx), F^(-1)(u) = -ln(1-u)/λ
def sample_exponential(lam, n):
    u = np.random.uniform(0, 1, n)
    return -np.log(1 - u) / lam

# 与numpy对比
samples = sample_exponential(1.0, 10000)
# 等价于 np.random.exponential(1.0, 10000)
```

**局限**：高斯分布没有闭式CDF逆，需要Box-Muller等专门方法。

---

### 2. 拒绝采样（Rejection Sampling）

**核心思想**：用一个易采样的提议分布 $q(x)$ 覆盖目标分布 $p(x)$。

**算法**：
1. 找常数 $M$ 使得 $p(x) \leq M \cdot q(x)$ 对所有 $x$ 成立
2. 从 $q(x)$ 采样 $x$
3. 从 $\text{Uniform}[0,1]$ 采样 $u$
4. 若 $u \leq \frac{p(x)}{M \cdot q(x)}$，接受 $x$；否则拒绝，重复

**接受率**：$\frac{1}{M}$（希望 $M \approx 1$）

```python
def rejection_sampling(p, q, q_sampler, M, n_samples):
    """
    p: 目标密度函数
    q: 提议密度函数  
    q_sampler: 从q采样的函数
    M: 上界常数
    """
    samples = []
    while len(samples) < n_samples:
        x = q_sampler()      # 从q采样
        u = np.random.uniform(0, 1)
        if u <= p(x) / (M * q(x)):  # 接受准则
            samples.append(x)
    return np.array(samples)

# 例子：用柯西分布采样截断正态
from scipy.stats import norm, cauchy

def truncated_normal_rejection(a, b, n):
    # 截断在[a,b]的正态分布
    def p(x):
        if a <= x <= b:
            return norm.pdf(x) / (norm.cdf(b) - norm.cdf(a))
        return 0
    
    M = 1 / (norm.cdf(b) - norm.cdf(a))  # 缩放因子
    # 提议分布：均匀覆盖[a,b]
    def q_sampler():
        return np.random.uniform(a, b)
    def q(x):
        return 1 / (b - a) if a <= x <= b else 0
    
    return rejection_sampling(p, q, q_sampler, M * (b-a), n)
```

**缺点**：高维空间中接受率指数下降（维度灾难）。

---

## 二、马尔可夫链蒙特卡洛（MCMC）

**核心思想**：构造马尔可夫链，使其平稳分布为目标分布 $p(x)$。

### 1. Metropolis-Hastings 算法

**最通用的MCMC方法**。

**算法步骤**：
1. 初始化 $x^{(0)}$
2. 对 $t = 1, 2, ...$：
   - 从提议分布 $q(x'|x^{(t-1)})$ 采样候选点 $x'$
   - 计算接受概率：
     $$\alpha = \min\left(1, \frac{p(x')q(x^{(t-1)}|x')}{p(x^{(t-1)})q(x'|x^{(t-1)})}\right)$$
   - 以概率 $\alpha$ 接受 $x^{(t)} = x'$，否则 $x^{(t)} = x^{(t-1)}$

**关键**：只需知道 $p$ 的未归一化形式！

```python
def metropolis_hastings(p_tilde, q_sampler, q_density, x0, n_samples):
    """
    p_tilde: 未归一化目标密度
    q_sampler: x' ~ q(·|x) 的采样函数
    q_density: q(x'|x) 的密度函数
    """
    samples = [x0]
    x_current = x0
    
    for _ in range(n_samples - 1):
        x_proposal = q_sampler(x_current)
        
        # 计算接受率
        numerator = p_tilde(x_proposal) * q_density(x_current, x_proposal)
        denominator = p_tilde(x_current) * q_density(x_proposal, x_current)
        alpha = min(1, numerator / denominator)
        
        if np.random.uniform(0, 1) < alpha:
            x_current = x_proposal
        samples.append(x_current)
    
    return np.array(samples)

# 例子：采样双峰分布
def bimodal_p_tilde(x):
    return np.exp(-(x**2 - 1)**2)  # 未归一化双峰密度

# 随机游走提议：q(x'|x) = N(x, 0.1^2)
def rw_sampler(x):
    return np.random.normal(x, 0.1)

def rw_density(x_prime, x):
    return norm.pdf(x_prime, x, 0.1)

samples = metropolis_hastings(bimodal_p_tilde, rw_sampler, rw_density, 0.0, 10000)
```

---

### 2. Gibbs 采样

**条件采样**：当联合分布难采样，但条件分布易采样时。

**算法**：对 $d$ 维变量 $x = (x_1, ..., x_d)$，逐坐标更新：
$$x_i^{(t)} \sim p(x_i | x_1^{(t)}, ..., x_{i-1}^{(t)}, x_{i+1}^{(t-1)}, ..., x_d^{(t-1)})$$

**特点**：
- 接受率恒为1（总是接受）
- 需要知道所有条件分布

```python
def gibbs_sampling(conditional_samplers, x0, n_samples):
    """
    conditional_samplers: [sampler_1, ..., sampler_d]
    每个 sampler_i 输入当前x，返回从 p(x_i | x_{-i}) 的样本
    """
    d = len(x0)
    samples = np.zeros((n_samples, d))
    x_current = x0.copy()
    
    for t in range(n_samples):
        for i in range(d):
            x_current[i] = conditional_samplers[i](x_current)
        samples[t] = x_current.copy()
    
    return samples

# 例子：二元正态分布采样
# (X,Y) ~ N(0, Σ), Σ = [[1, ρ], [ρ, 1]]
rho = 0.8

def cond_x_given_y(x_current):
    y = x_current[1]
    x_mean = rho * y
    x_var = 1 - rho**2
    return np.random.normal(x_mean, np.sqrt(x_var))

def cond_y_given_x(x_current):
    x = x_current[0]
    y_mean = rho * x
    y_var = 1 - rho**2
    return np.random.normal(y_mean, np.sqrt(y_var))

samples = gibbs_sampling([cond_x_given_y, cond_y_given_x], [0.0, 0.0], 10000)
```

**应用场景**：概率图模型、主题模型（LDA）、贝叶斯网络。

---

### 3. 高级MCMC方法

| 方法 | 核心思想 | 优势 |
|------|---------|------|
| **Hamiltonian Monte Carlo (HMC)** | 引入动量变量，利用梯度信息模拟哈密顿动力学 | 大步长、低自相关、适合高维 |
| **No-U-Turn Sampler (NUTS)** | 自动确定轨迹长度，避免手动调参 | PyMC3、Stan的默认采样器 |
| **Slice Sampling** | 引入辅助变量，均匀采样水平集 | 无需调参提议分布 |
| **Adaptive MCMC** | 在线调整提议分布协方差 | 自动适应目标形状 |

**HMC直观理解**：
- 将采样看作物理粒子在能量地形上的运动
- 梯度 $\nabla \log p(x)$ 提供"力"的方向
- 动量帮助穿越狭窄山谷，避免随机游走的低效

```python
# HMC伪代码（简化版）
def hmc_sample(p_log, grad_log_p, x0, n_samples, L, epsilon):
    """
    p_log: 对数目标密度
    grad_log_p: 对数密度的梯度
    L:  leapfrog步数
    epsilon: 步长
    """
    samples = [x0]
    x = x0
    
    for _ in range(n_samples):
        # 采样动量
        p = np.random.normal(0, 1, len(x))
        current_H = -p_log(x) + 0.5 * np.dot(p, p)
        
        # Leapfrog积分
        x_new, p_new = leapfrog(x, p, grad_log_p, L, epsilon)
        
        # Metropolis接受
        proposed_H = -p_log(x_new) + 0.5 * np.dot(p_new, p_new)
        if np.random.uniform(0, 1) < np.exp(current_H - proposed_H):
            x = x_new
        
        samples.append(x.copy())
    
    return samples
```

---

## 三、变分推断中的采样

### 重参数化技巧（Reparameterization Trick）

**问题**：从 $q_\phi(z) = \mathcal{N}(z; \mu_\phi, \sigma_\phi^2)$ 采样后，如何对 $\phi$ 求梯度？

**解决方案**：
$$z = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**关键**：随机性来自与 $\phi$ 无关的 $\epsilon$，梯度可传播！

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # 输出mu和log_var
        )
        self.decoder = nn.Sequential(...)  # 省略
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 标准正态采样
        return mu + std * eps  # 重参数化
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
```

---

## 四、近似推断采样

### 1. 重要性采样（回顾与扩展）

已在前面详细讲解，核心公式：
$$\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i)\frac{p(x_i)}{q(x_i)}, \quad x_i \sim q$$

**序列重要性采样（SIS）/粒子滤波**：

用于时序模型 $p(x_{1:T}|y_{1:T})$：

```python
def particle_filter(initial_dist, transition, likelihood, observations, n_particles):
    """
    粒子滤波（序列重要性采样）
    """
    T = len(observations)
    particles = np.zeros((T, n_particles))
    weights = np.zeros((T, n_particles))
    
    # 初始化
    particles[0] = initial_dist.sample(n_particles)
    weights[0] = likelihood(particles[0], observations[0])
    weights[0] /= np.sum(weights[0])
    
    for t in range(1, T):
        # 重采样（防止权重退化）
        indices = np.random.choice(n_particles, n_particles, p=weights[t-1])
        particles_prev = particles[t-1, indices]
        
        # 传播
        particles[t] = transition.sample(particles_prev)
        
        # 更新权重
        weights[t] = likelihood(particles[t], observations[t])
        weights[t] /= np.sum(weights[t])
    
    return particles, weights
```

---

### 2. 郎之万动力学（Langevin Dynamics）

**基于梯度的采样**，离散化SDE：
$$dx_t = \nabla \log p(x_t) dt + \sqrt{2} dW_t$$

**无MCMC接受步骤**，适合大规模数据（与SGD结合）。

```python
def langevin_dynamics(p_log, grad_log_p, x0, n_steps, epsilon):
    """
    无偏朗之万动力学（ULA: Unadjusted Langevin Algorithm）
    """
    x = x0
