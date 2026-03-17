# 重要性采样：原理、算法与应用

## 1. 引言

重要性采样（Importance Sampling）是蒙特卡洛方法中的核心技术之一，广泛应用于统计物理、贝叶斯推断、金融工程及计算机图形学等领域。当目标分布难以直接采样或计算期望时，重要性采样通过引入提议分布（proposal distribution），将计算问题转化为更易处理的采样问题。

本章首先回顾直接采样与拒绝采样等基础方法，为理解重要性采样的必要性奠定基础。

## 2. 基本采样方法

### 2.1 逆变换采样

逆变换采样（Inverse Transform Sampling）基于概率积分变换理论：若随机变量 $U \sim \text{Uniform}[0,1]$，$F$ 为目标分布的累积分布函数（CDF），则 $X = F^{-1}(U)$ 服从目标分布 $F$。

**定理 2.1**（概率积分变换）：设 $F$ 为连续型随机变量的累积分布函数，$F^{-1}$ 为其分位函数，则
$$X = F^{-1}(U), \quad U \sim \text{Uniform}[0,1] \implies X \sim F$$

**证明**：对任意 $x \in \mathbb{R}$，
$$P(X \leq x) = P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)$$
故 $X$ 的分布函数为 $F$。

**适用条件**：逆变换采样要求 $F^{-1}$ 具有显式表达式或可高效数值计算。典型适用分布包括指数分布、柯西分布、逻辑斯蒂分布等。

**示例**：指数分布 $\text{Exp}(\lambda)$ 的 CDF 为 $F(x) = 1 - e^{-\lambda x}$，其逆函数为
$$F^{-1}(u) = -\frac{1}{\lambda}\ln(1-u)$$

```python
import numpy as np

def sample_exponential(lam: float, n: int) -> np.ndarray:
    """
    使用逆变换采样生成指数分布样本。

    参数:
        lam: 速率参数 λ > 0
        n: 样本数量
    返回:
        服从 Exp(lam) 的样本数组
    """
    u = np.random.uniform(0, 1, n)
    return -np.log(1 - u) / lam
```

**局限性**：高斯分布等常见分布的 CDF 无闭式逆函数，需借助数值方法（如误差函数逆）或替代采样策略。

### 2.2 拒绝采样

拒绝采样（Rejection Sampling）通过构造覆盖目标分布的提议分布实现采样，适用于目标密度已知但难以直接采样的情形。

**算法 2.1**（拒绝采样）：
设目标密度 $p(x)$ 与提议密度 $q(x)$ 满足 $p(x) \leq M \cdot q(x)$ 对所有 $x$ 成立，$M \geq 1$ 为常数。

1. 从 $q(x)$ 采样候选点 $x$；
2. 从 $\text{Uniform}[0,1]$ 采样 $u$；
3. 若 $u \leq \frac{p(x)}{M \cdot q(x)}$，接受 $x$；否则拒绝并返回步骤 1。

**定理 2.2**（拒绝采样的正确性）：算法 2.1 输出的样本服从分布 $p(x)$。

**证明**：设 $(X, U)$ 为联合采样，接受事件为 $\mathcal{A} = \{U \leq p(X)/(Mq(X))\}$。对任意可测集 $\mathcal{B}$，
$$P(X \in \mathcal{B} \mid \mathcal{A}) = \frac{P(X \in \mathcal{B}, U \leq p(X)/(Mq(X)))}{P(\mathcal{A})}$$

分子为：
$$\int_{\mathcal{B}} \int_0^{p(x)/(Mq(x))} q(x) \, du \, dx = \int_{\mathcal{B}} \frac{p(x)}{M} \, dx$$

分母为：
$$\int_{\mathbb{R}^d} \frac{p(x)}{M} \, dx = \frac{1}{M}$$

因此 $P(X \in \mathcal{B} \mid \mathcal{A}) = \int_{\mathcal{B}} p(x) \, dx$，即接受样本服从 $p(x)$。

**接受率分析**：单次迭代接受概率为
$$P(\text{accept}) = \frac{1}{M}$$
期望迭代次数为 $M$。当 $M \gg 1$ 时，算法效率急剧下降。

**示例**：用柯西分布采样截断正态分布

```python
from scipy.stats import norm, cauchy

def truncated_normal_rejection(a: float, b: float, n: int) -> np.ndarray:
    """
    使用拒绝采样生成截断正态分布样本。

    参数:
        a, b: 截断区间 [a, b]
        n: 样本数量
    """
    def p(x):
        if a <= x <= b:
            return norm.pdf(x) / (norm.cdf(b) - norm.cdf(a))
        return 0.0

    # 提议分布：均匀分布
    M = 1.0 / (norm.cdf(b) - norm.cdf(a)) * (b - a)

    samples = []
    while len(samples) < n:
        x = np.random.uniform(a, b)
        u = np.random.uniform(0, 1)
        if u <= p(x) * (b - a) / M:
            samples.append(x)

    return np.array(samples)
```

**缺点**：当目标分布在局部具有极高密度时，常数 $M$ 需取较大值，导致接受率过低，采样效率显著下降。此局限促使我们寻求更高效的采样策略——重要性采样。

## 3. 重要性采样

### 3.1 问题背景

考虑计算期望值：
$$\mu = \mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) \, dx$$

实践中常面临以下困难：
- 从 $p(x)$ 直接采样计算上不可行或代价高昂；
- $p(x)$ 仅已知未归一化形式 $\tilde{p}(x) = Z \cdot p(x)$，归一化常数 $Z$ 未知（如贝叶斯推断中的后验分布）；
- $f(x)$ 在 $p(x)$ 高概率区域取值较小，导致标准蒙特卡洛估计方差过大。

重要性采样通过引入**提议分布** $q(x)$，将积分转化为关于 $q$ 的期望，从而规避上述困难。

### 3.2 基本推导

设 $q(x)$ 为提议分布，满足 $p(x) > 0 \implies q(x) > 0$（支撑覆盖条件）。重写期望：
$$\mu = \int f(x) p(x) \, dx = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$$

**定义 3.1**（重要性权重）：函数 $w(x) = \frac{p(x)}{q(x)}$ 称为重要性权重（Importance Weight）或似然比（Likelihood Ratio）。

从 $q(x)$ 独立采样 $\{x_i\}_{i=1}^N$，得到**标准重要性采样估计量**：
$$\hat{\mu}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^N f(x_i) w(x_i)$$

### 3.3 归一化重要性采样

当 $p(x)$ 仅已知未归一化形式 $\tilde{p}(x) = Z \cdot p(x)$ 时，定义未归一化权重 $\tilde{w}(x) = \frac{\tilde{p}(x)}{q(x)}$。由于 $Z = \int \tilde{p}(x) \, dx$ 未知，采用**归一化重要性采样估计量**（Normalized Importance Sampling, NIS）：
$$\hat{\mu}_{\text{NIS}} = \frac{\sum_{i=1}^N f(x_i) \tilde{w}(x_i)}{\sum_{i=1}^N \tilde{w}(x_i)} = \sum_{i=1}^n f(x_i) \bar{w}_i$$
其中 $\bar{w}_i = \frac{\tilde{w}(x_i)}{\sum_{j=1}^N \tilde{w}(x_j)}$ 为归一化权重。

### 3.4 理论性质

**定理 3.1**（无偏性）：标准重要性采样估计量 $\hat{\mu}_{\text{IS}}$ 是 $\mu$ 的无偏估计。

**证明**：
$$\mathbb{E}_q[\hat{\mu}_{\text{IS}}] = \frac{1}{N} \sum_{i=1}^N \mathbb{E}_q[f(x_i)w(x_i)] = \mathbb{E}_q[f(x)w(x)] = \int f(x)\frac{p(x)}{q(x)}q(x)\,dx = \mu$$

**定理 3.2**（一致性）：归一化重要性采样估计量 $\hat{\mu}_{\text{NIS}}$ 依概率收敛于 $\mu$，即 $\hat{\mu}_{\text{NIS}} \xrightarrow{P} \mu$（当 $N \to \infty$）。

**证明**：由强大数定律，
$$\frac{1}{N}\sum_{i=1}^N \tilde{w}(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}_q[\tilde{w}(x)] = Z$$
$$\frac{1}{N}\sum_{i=1}^N f(x_i)\tilde{w}(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}_q[f(x)\tilde{w}(x)] = Z\mu$$
由连续映射定理，
$$\hat{\mu}_{\text{NIS}} = \frac{\frac{1}{N}\sum_{i=1}^N f(x_i)\tilde{w}(x_i)}{\frac{1}{N}\sum_{i=1}^N \tilde{w}(x_i)} \xrightarrow{\text{a.s.}} \frac{Z\mu}{Z} = \mu$$
故 $\hat{\mu}_{\text{NIS}}$ 是强一致的，自然弱一致。

**注**：$\hat{\mu}_{\text{NIS}}$ 是有偏估计，偏差为 $O(N^{-1})$，但一致性保证大样本下的可靠性。

**定理 3.3**（方差分析）：标准重要性采样估计量的方差为
$$\text{Var}(\hat{\mu}_{\text{IS}}) = \frac{1}{N}\text{Var}_q(f(x)w(x)) = \frac{1}{N}\left(\int \frac{f^2(x)p^2(x)}{q(x)}\,dx - \mu^2\right)$$

**最优提议分布**：最小化方差等价于最小化 $\int \frac{f^2(x)p^2(x)}{q(x)}\,dx$。由变分法，最优提议为
$$q^*(x) = \frac{|f(x)|p(x)}{\int |f(t)|p(t)\,dt}$$
此时 $\text{Var}(\hat{\mu}_{\text{IS}}) = 0$（当 $f(x) \geq 0$ 或 $f(x) \leq 0$ 时）。

**证明**：对任意提议分布 $q$，考虑
$$\text{Var}_q(f(x)w(x)) = \mathbb{E}_q[f^2(x)w^2(x)] - \mu^2 = \int \frac{f^2(x)p^2(x)}{q(x)}\,dx - \mu^2$$
由柯西-施瓦茨不等式，
$$\left(\int |f(x)|p(x)\,dx\right)^2 = \left(\int \frac{|f(x)|p(x)}{\sqrt{q(x)}} \cdot \sqrt{q(x)}\,dx\right)^2 \leq \int \frac{f^2(x)p^2(x)}{q(x)}\,dx \cdot \int q(x)\,dx = \int \frac{f^2(x)p^2(x)}{q(x)}\,dx$$
当且仅当 $\frac{|f(x)|p(x)}{\sqrt{q(x)}} \propto \sqrt{q(x)}$，即 $q(x) \propto |f(x)|p(x)$ 时取等。

### 3.5 权重退化问题

在高维空间中，重要性采样常面临**权重退化**（Weight Degeneracy）：少数样本的权重占主导地位，导致有效样本量（Effective Sample Size, ESS）远低于实际样本量 $N$。

**定义 3.2**（有效样本量）：
$$\text{ESS} = \frac{\left(\sum_{i=1}^N \bar{w}_i\right)^2}{\sum_{i=1}^N \bar{w}_i^2} = \frac{1}{\sum_{i=1}^N \bar{w}_i^2}$$

当权重均匀时 $\text{ESS} = N$；当某一权重为 1 其余为 0 时 $\text{ESS} = 1$。实践中，若 $\text{ESS} \ll N$，表明提议分布与目标分布匹配不佳，需调整 $q(x)$ 或采用自适应方法。

---

## 4. 总结

本文系统阐述了重要性采样的数学基础：
1. **逆变换采样**与**拒绝采样**为理解重要性采样提供基础；
2. **重要性采样**通过重要性权重转换期望计算，适用于难以直接采样的分布；
3. **归一化重要性采样**处理未归一化目标分布，具有一致性但引入小样本偏差；
4. **最优提议分布** $q^*(x) \propto |f(x)|p(x)$ 可最小化方差；
5. **权重退化**是高维应用中的核心挑战，需通过有效样本量监控。

重要性采样是粒子滤波、序列蒙特卡洛、方差缩减等高级方法的基础，其理论性质为实际应用提供严格保证。
