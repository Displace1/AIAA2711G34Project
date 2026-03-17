  作为大学机器学习讲师，我会按照**认知递进**和**数学基础**的逻辑，为学生设计以下前置学习路径：

---

## 一、前置知识体系总览

```
学习重要性采样前必须掌握：
├── 概率论基础（第1层）
├── 基础采样方法（第2层）  
├── 马尔可夫链理论（第3层）
└── 蒙特卡洛方法（第4层）
```

---

## 二、第1层：概率论与统计基础（2-3周）

### 核心概念

| 主题 | 关键内容 | 与重要性采样的联系 |
|------|---------|------------------|
| **期望与方差** | $\mathbb{E}[X]$, $\text{Var}(X)$, 协方差 | IS估计量的无偏性、方差分析 |
| **条件概率与贝叶斯定理** | $P(A\|B)$, 后验 $p(\theta\|D) \propto p(D\|\theta)p(\theta)$ | 贝叶斯推断是IS的主要应用场景 |
| **概率密度/质量函数** | PDF/PMF的性质，支撑集 | 权重 $w(x)=p(x)/q(x)$ 的定义域 |
| **变换采样** | 逆CDF法 $X = F^{-1}(U)$ | 理解"为什么有些分布难以采样" |
| **大数定律与CLT** | $\bar{X}_n \to \mu$, 渐近正态性 | IS估计量的收敛性保证 |

### 必须掌握的数学技巧

**Jacobian变换**（理解变量变换对密度的影响）：
$$p_Y(y) = p_X(g^{-1}(y)) \left| \frac{d}{dy}g^{-1}(y) \right|$$

**为什么重要**：重要性采样本质上是**测度变换**，需要理解密度如何随变量变换而变化。

---

## 三、第2层：基础采样方法（3-4周）

### 2.1 精确采样方法（Exact Sampling）

#### 逆变换采样（Inverse Transform Sampling）
```python
import numpy as np

def inverse_transform_sampling(pdf, cdf_inv, n_samples):
    """
    通用框架：U ~ Uniform[0,1], X = F^{-1}(U)
    """
    u = np.random.uniform(0, 1, n_samples)
    return cdf_inv(u)

# 例子：指数分布
def sample_exponential(lam, n):
    u = np.random.uniform(0, 1, n)
    return -np.log(1 - u) / lam  # F^{-1}(u) = -ln(1-u)/λ
```

**教学要点**：
- 适用条件：CDF可解析求逆
- 局限性：高斯分布的CDF没有闭式逆（需数值近似）
- **关键认知**：很多分布**理论上**可采样，但**计算上**困难

#### 拒绝采样（Rejection Sampling）

**算法**：
```
目标：从 p(x) 采样
提议：选择易采样的 q(x)，找到 M 使得 p(x) ≤ Mq(x)

重复：
    1. 从 q(x) 采样 x
    2. 从 Uniform[0,1] 采样 u
    3. 如果 u < p(x)/(Mq(x))，接受 x
    否则拒绝，重试
```

**Python实现**：
```python
def rejection_sampling(p, q_sampler, q_density, M, n_samples):
    """
    p: 目标密度函数（可计算，但难采样）
    q_sampler: 提议分布采样函数（易采样）
    q_density: 提议分布密度（易计算）
    M: 常数，满足 p(x) <= M*q(x)
    """
    samples = []
    while len(samples) < n_samples:
        x = q_sampler()      # 从q采样（易）
        u = np.random.uniform(0, 1)
        if u < p(x) / (M * q_density(x)):  # 需要计算p(x)！
            samples.append(x)
    return np.array(samples)
```

**与重要性采样的对比**：

| 特性 | 拒绝采样 | 重要性采样 |
|------|---------|-----------|
| 输出 | 精确服从 $p(x)$ 的样本 | 加权样本（期望正确） |
| 需要 $p(x)$ | 需要（计算接受概率） | 需要（计算权重） |
| 效率 | 依赖接受率（高维时→0） | 依赖权重方差（高维时爆炸） |
| 用途 | 获取无偏样本 | 估计期望 |

**关键过渡**：拒绝采样也需要计算 $p(x)$，但**丢弃**了被拒绝的样本；重要性采样**保留所有样本**，用权重修正偏差。

---

### 2.2 特殊分布采样技巧

| 分布 | 方法 | 教学目的 |
|------|------|---------|
| 高斯分布 | Box-Muller变换 | 理解变换技巧 |
| 多元高斯 | Cholesky分解 $\Sigma = LL^T$ | 为卡尔曼滤波、VAE铺垫 |
| 类别分布 | Gumbel-Softmax/别名方法 | 离散采样基础 |
| Dirichlet | Gamma分布归一化 | 贝叶斯非参数基础 |

**Box-Muller示例**（理解"构造性"采样）：
```python
def box_muller(n):
    """用均匀分布构造高斯分布"""
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    z0 = r * np.cos(theta)  # ~ N(0,1)
    z1 = r * np.sin(theta)  # ~ N(0,1)
    return z0, z1
```

---

## 四、第3层：马尔可夫链与随机过程（2-3周）

### 3.1 马尔可夫链基础

**必须掌握**：
- 转移核 $K(x'|x)$ 与细致平衡条件
- 平稳分布 $\pi$：$\pi(x') = \int K(x'|x)\pi(x)dx$
- 遍历性（Ergodicity）：链收敛到唯一平稳分布

**与MCMC的关系**：MCMC构造转移核使得 $p(x)$ 成为平稳分布，从而**间接采样**。

### 3.2 Metropolis-Hastings算法

**算法**（理解"间接采样"的思想）：
```
当前状态 x_t
从提议 q(x'|x_t) 采样候选 x'
计算接受率 α = min(1, [p(x')q(x_t|x')]/[p(x_t)q(x'|x_t)])
以概率 α 接受 x_{t+1} = x'，否则 x_{t+1} = x_t
```

**关键洞察**：
- 只需要知道 $p(x)$ 的**比率**（归一化常数抵消！）
- 通过**构造马尔可夫链**而非直接采样获得样本

**Python实现**：
```python
def metropolis_hastings(p, q_sampler, q_density, x0, n_samples):
    """
    p: 未归一化目标密度（只需比率）
    """
    samples = [x0]
    x_current = x0
    
    for _ in range(n_samples):
        x_proposal = q_sampler(x_current)
        
        # 计算接受率（p只需未归一化形式！）
        numerator = p(x_proposal) * q_density(x_current, x_proposal)
        denominator = p(x_current) * q_density(x_proposal, x_current)
        alpha = min(1, numerator / denominator)
        
        if np.random.uniform(0, 1) < alpha:
            x_current = x_proposal
        samples.append(x_current)
    
    return np.array(samples[1000:])  # 丢弃burn-in
```

---

## 五、第4层：蒙特卡洛方法（2周）

### 5.1 经典蒙特卡洛积分

**问题**：估计 $I = \int_a^b f(x)dx$

**方法**：$I \approx \frac{b-a}{N} \sum_{i=1}^N f(x_i)$, $x_i \sim \text{Uniform}[a,b]$

**误差分析**：
$$\text{MSE} = \frac{(b-a)^2 \text{Var}(f(X))}{N} = O(N^{-1})$$

**对比数值积分**：
- 梯形法则：$O(N^{-2/d})$（维度灾难！）
- 蒙特卡洛：$O(N^{-1/2})$（与维度无关）

**核心认知**：蒙特卡洛在高维空间战胜确定性方法。

### 5.2 从蒙特卡洛到重要性采样

**经典MC的局限**：
- 均匀采样在 $f(x)$ 集中在小区域时效率低
- 方差 $\text{Var}(f(X))$ 可能很大

**引入重要性采样的动机**：
> "如果 $f(x)$ 在某些区域更重要，我们应该在那里多采样，并用权重修正分布偏差"

**方差对比**：
- 经典MC：$\sigma^2_{MC} = \text{Var}_p(f(X))$
- 最优IS：$\sigma^2_{IS} = 0$（若 $q \propto |f|p$，理论极限）

---

## 六、完整教学路线图

```
第1-3周：概率论基础
    └─ 作业：证明IS估计量的无偏性
    
第4-7周：精确采样方法
    ├─ 逆变换采样（可解析逆的情况）
    ├─ 拒绝采样（需要计算密度，但难采样）
    └─ 作业：实现拒绝采样，观察高维接受率崩溃
    
第8-10周：马尔可夫链与MCMC
    ├─ MH算法（只需密度比率，无需归一化）
    ├─ Gibbs采样（条件分布易采样的情况）
    └─ 作业：用MCMC采样双变量高斯混合
    
第11-12周：蒙特卡洛积分
    ├─ 经典MC方法
    ├─ 方差缩减技术（对偶变量、控制变量）
    └─ 引入问题：当p(x)难采样但密度已知时怎么办？
    
第13-14周：重要性采样（正式引入）
    ├─ 数学推导（测度变换）
    ├─ 归一化IS（处理未归一化密度）
    ├─ 最优提议分布与诊断（ESS）
    └─ 大作业：实现贝叶斯逻辑回归的IS推断
```

---

## 七、关键过渡示例

### 课堂演示：从拒绝采样到重要性采样

**场景**：估计 $\mathbb{E}_{p}[X^2]$，$p(x) = \mathcal{N}(x; 5, 0.5)$，$q(x) = \mathcal{N}(x; 0, 3)$

**拒绝采样版本**：
```python
samples = []
for _ in range(100000):
    x = np.random.normal(0, 3)
    if np.random.uniform(0,1) < norm.pdf(x,5,0.5)/(M*norm.pdf(x,0,3)):
        samples.append(x)
# 接受率 ≈ 0.01，有效样本1000个
estimate = np.mean([x**2 for x in samples])
```

**重要性采样版本**：
```python
x = np.random.normal(0, 3, 100000)  # 全部保留！
w = norm.pdf(x,5,0.5) / norm.pdf(x,0,3)
estimate = np.mean(x**2 * w)
# 使用全部10万个样本，但权重方差可能大
```

**课堂讨论**：
- 拒绝采样"浪费"了99%的计算，但给出完美样本
- 重要性采样"利用"了所有计算，但引入权重方差
- **核心权衡**：样本质量 vs 计算效率

---

## 八、学生应先掌握的Python技能

```python
# 必须熟练的编程模式

# 1. 向量化计算（避免循环）
samples = np.random.normal(mu, sigma, size=(n_samples, dim))

# 2. 对数空间计算（防止下溢）
log_weights = log_p(samples) - log_q(samples)
log_weights -= np.max(log_weights)  # 数值稳定性
weights = np.exp(log_weights)
weights /= np.sum(weights)

# 3. 有效样本量计算
ess = np.sum(weights)**2 / np.sum(weights**2)

# 4. 加权统计量
weighted_mean = np.sum(samples * weights)
weighted_var = np.sum(weights * (samples - weighted_mean)**2)
```

---

## 九、总结：前置知识检查清单

在讲授重要性采样前，学生应当能够：

| 能力 | 验证问题 |
|------|---------|
| 解释为什么需要采样 | "为什么不能直接积分计算期望？" |
| 实现拒绝采样 | 编写函数从复杂分布采样 |
| 计算提议分布的接受率 | 分析高维时接受率为何崩溃 |
| 解释MCMC的基本思想 | "MH算法为什么只需要密度比率？" |
| 计算蒙特卡洛估计的方差 | 推导误差随样本的变化规律 |
| 理解测度变换 | 解释 $dx$ 与 $dy$ 在变量替换时的关系 |

**只有当学生理解"直接采样困难但密度计算容易"这一核心矛盾时，重要性采样的引入才会显得自然且必要。**