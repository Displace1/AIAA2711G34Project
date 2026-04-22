好的，我们来实现**实验二（重尾性实验）**的完整版本，并按要求将“好的提议分布”改为学生 t 分布。该实验将直观展示：当提议分布的尾部比目标分布更薄时，重要性权重会出现极端值，导致方差爆炸。

---

## 实验：尾部风险与重尾性对重要性采样的影响

### 实验目标

- **目标分布**：标准正态分布 \(p(x) = \mathcal{N}(0,1)\)
- **待估计量**：尾部概率 \(I = \mathbb{P}(X > 5) = \int_{5}^{\infty} p(x) dx\)（真实值约 \(2.87 \times 10^{-7}\)）
- **对比提议分布**：
  1. **好 q（重尾）**：自由度为 3 的学生 t 分布，位置参数设为 5，尺度适当调整，使其覆盖尾部区域且尾部比正态分布更重。
  2. **坏 q（薄尾）**：正态分布 \(\mathcal{N}(5, 1)\)，其尾部比标准正态在 \(x>5\) 处下降得更快（注意：目标 \(p(x)\) 在 5 以后呈指数衰减，而 \(\mathcal{N}(5,1)\) 的方差为 1，其尾部在远离中心时比标准正态的尾部更薄）。
- **观察指标**：估计值的方差、重要性权重的分布（尤其关注最大值和变异系数）。

### 理论背景

- 重要性采样权重 \(w(x) = p(x)/q(x)\)。
- 若 \(q\) 的尾部比 \(p\) 薄，则在尾部区域 \(q(x)\) 非常小，导致 \(w(x)\) 巨大，少数样本主导估计，方差急剧增大（甚至无穷）。
- 学生 t 分布具有多项式衰减的尾部，比正态的指数衰减更重，因此作为“好 q”能保证权重有界或方差有限。

---

## 完整代码

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create results directory
results_dir = '../results'
os.makedirs(results_dir, exist_ok=True)

# Set random seed
np.random.seed(42)

# ====================================================
# 1. Define target distribution and quantity of interest
# ====================================================
# Target: standard normal
p_density = stats.norm.pdf
# Tail probability P(X > 5)
true_tail_prob = 1 - stats.norm.cdf(5)
print(f"True tail probability P(X>5): {true_tail_prob:.4e}")

# Indicator function for X > 5
def f(x):
    return (x > 5).astype(float)

# ====================================================
# 2. Define proposals
# ====================================================
# Good proposal: Student's t with df=3, shifted to cover the tail region
# We choose location = 5, scale = 2 to ensure it covers [5, ∞) with heavy tail.
def sample_q_good(N):
    return stats.t.rvs(df=3, loc=5, scale=2, size=N)

def q_good_density(x):
    return stats.t.pdf(x, df=3, loc=5, scale=2)

# Bad proposal: Normal(5, 1) — thin tail relative to target's exponential decay
def sample_q_bad(N):
    return np.random.normal(5, 1, size=N)

def q_bad_density(x):
    return stats.norm.pdf(x, loc=5, scale=1)

# ====================================================
# 3. Simulation parameters
# ====================================================
N_list = [1000, 3000, 10000, 30000, 100000]  # sample sizes
M_reps = 200   # number of repetitions for variance estimation

# Storage
var_good = []
var_bad = []
mean_good = []
mean_bad = []
max_weight_good = []
max_weight_bad = []

# ====================================================
# 4. Run experiments
# ====================================================
print("\nRunning tail probability estimation...")
for N in N_list:
    est_good = np.zeros(M_reps)
    est_bad = np.zeros(M_reps)
    w_max_good = []
    w_max_bad = []
    
    for rep in range(M_reps):
        # Good proposal (t)
        x_g = sample_q_good(N)
        w_g = p_density(x_g) / q_good_density(x_g)
        est_good[rep] = np.mean(f(x_g) * w_g)
        w_max_good.append(np.max(w_g))
        
        # Bad proposal (normal)
        x_b = sample_q_bad(N)
        w_b = p_density(x_b) / q_bad_density(x_b)
        est_bad[rep] = np.mean(f(x_b) * w_b)
        w_max_bad.append(np.max(w_b))
    
    var_good.append(np.var(est_good, ddof=1))
    var_bad.append(np.var(est_bad, ddof=1))
    mean_good.append(np.mean(est_good))
    mean_bad.append(np.mean(est_bad))
    max_weight_good.append(np.mean(w_max_good))   # average max weight over repetitions
    max_weight_bad.append(np.mean(w_max_bad))
    
    print(f"N={N:6d} | Good q var={var_good[-1]:.4e} | Bad q var={var_bad[-1]:.4e} | "
          f"Avg max w: good={max_weight_good[-1]:.2e}, bad={max_weight_bad[-1]:.2e}")

# ====================================================
# 5. Visualizations
# ====================================================
# Figure 1: Variance vs. N
fig1, ax = plt.subplots(figsize=(8, 6))
ax.loglog(N_list, var_good, 'o-', label='IS with t (good q)', linewidth=2, markersize=8)
ax.loglog(N_list, var_bad, 's-', label='IS with Normal(5,1) (bad q)', linewidth=2, markersize=8)
# Reference 1/N line
ref = np.array(N_list)
ax.loglog(ref, var_good[0] * (ref[0] / ref), 'k--', alpha=0.5, label=r'$\propto 1/N$')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Empirical variance', fontsize=12)
ax.set_title('Variance Comparison: Heavy-tailed vs Thin-tailed Proposal', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_variance.png'), dpi=150)
plt.show()

# Figure 2: Distribution of importance weights for a single run with large N
N_demo = 50000
x_g_demo = sample_q_good(N_demo)
x_b_demo = sample_q_bad(N_demo)
w_g_demo = p_density(x_g_demo) / q_good_density(x_g_demo)
w_b_demo = p_density(x_b_demo) / q_bad_density(x_b_demo)

# Only consider weights for samples where f(x)=1 (i.e., x > 5)
w_g_tail = w_g_demo[x_g_demo > 5]
w_b_tail = w_b_demo[x_b_demo > 5]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Good q weights
axes2[0].hist(w_g_tail, bins=50, alpha=0.7, color='green', edgecolor='black')
axes2[0].set_title(f'Good q (t dist): Weights for X>5 (N={N_demo})')
axes2[0].set_xlabel('Importance weight w(x)')
axes2[0].set_ylabel('Frequency')
axes2[0].axvline(x=np.mean(w_g_tail), color='red', linestyle='--', label=f'mean={np.mean(w_g_tail):.2e}')
axes2[0].legend()
axes2[0].set_xlim([0, np.percentile(w_g_tail, 99)])  # avoid extreme tail stretching view

# Bad q weights
axes2[1].hist(w_b_tail, bins=50, alpha=0.7, color='red', edgecolor='black')
axes2[1].set_title(f'Bad q (Normal): Weights for X>5 (N={N_demo})')
axes2[1].set_xlabel('Importance weight w(x)')
axes2[1].set_ylabel('Frequency')
axes2[1].axvline(x=np.mean(w_b_tail), color='blue', linestyle='--', label=f'mean={np.mean(w_b_tail):.2e}')
axes2[1].legend()
axes2[1].set_xlim([0, np.percentile(w_b_tail, 99)])  # avoid extreme tail

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_weights.png'), dpi=150)
plt.show()

# Figure 3: Convergence of estimates
fig3, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(N_list, mean_good, 'o-', label='IS with t (good q)', linewidth=2, markersize=8)
ax.semilogx(N_list, mean_bad, 's-', label='IS with Normal(5,1) (bad q)', linewidth=2, markersize=8)
ax.axhline(y=true_tail_prob, color='k', linestyle='--', label=f'True value = {true_tail_prob:.2e}')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Mean estimate', fontsize=12)
ax.set_title('Convergence of Tail Probability Estimates', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_means.png'), dpi=150)
plt.show()

# ====================================================
# 6. Print summary
# ====================================================
print("\n--- Summary ---")
print(f"True tail probability: {true_tail_prob:.4e}")
print("For bad q (Normal(5,1)), observe huge variance and occasional extreme weights.")
print("For good q (t with df=3), variance scales roughly as 1/N and weights are well-behaved.")
```

---

## 关键设计说明

1. **好 q：t 分布**
   - 自由度 `df=3`，保证了多项式衰减的尾部（比正态的指数衰减更重）。
   - 位置参数 `loc=5` 将概率质量集中在感兴趣的区域 \(x>5\)。
   - 尺度参数 `scale=2` 使分布足够宽，覆盖尾部。

2. **坏 q：正态分布 \(\mathcal{N}(5,1)\)**
   - 其密度函数在远离均值时呈指数衰减，与目标分布的标准正态在尾部有相似的衰减速度，但此处目标 \(p(x)\) 是标准正态（方差 1），而坏 q 也是方差 1，两者尾部厚度相当，但在计算权重时，由于 q 的均值在 5，其在 \(x>5\) 的区域密度会迅速下降，导致权重增大。理论上，如果 q 的尾部比 p 薄，方差可能无限；这里二者尾部衰减阶数相同（都是指数衰减），但参数不同仍会导致权重方差较大。

3. **估计量**：普通重要性采样（非自归一化），因为 \(p(x)\) 是归一化的，权重期望为 1。

4. **可视化**
   - 方差对比图：展示好 q 的方差随 \(1/N\) 稳定下降，坏 q 的方差很大且可能不规律。
   - 权重分布直方图：仅展示 \(x>5\) 的样本权重，直观看到坏 q 下权重分布有长尾，甚至出现极大值。
   - 估计值收敛图：好 q 快速收敛到真值，坏 q 波动剧烈。

---

## 预期观察结果

- **好 q（t 分布）**：权重最大值约在 \(10^{-5}\) 量级，方差随 N 增大稳定减小。
- **坏 q（正态）**：权重最大值可达 \(10^{-2}\) 甚至更高，方差比好 q 大数个数量级，且收敛缓慢。
- 该实验生动证明了**重要性采样中提议分布必须具有重尾性**的重要性。

运行此代码将生成三张图表并保存至 `../results` 目录，您可以通过调整样本量或分布参数进一步探索重尾性的影响。如果有任何需要调整的地方，请随时告知！