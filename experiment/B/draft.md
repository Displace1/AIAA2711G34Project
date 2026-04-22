好的，我们来实现第二个实验：目标分布为混合高斯 \(p(x) = 0.3 \mathcal{N}(4,1) + 0.7 \mathcal{N}(7,0.5)\)，对比三种采样方法：直接蒙特卡洛、拒绝采样、重要性采样。

我将提供一个完整、自包含的 Python 脚本，包括理论说明、代码实现和可视化输出，所有结果保存在 `../results` 目录中。

---

## 实验二：混合高斯分布采样方法对比

### 实验目标

- **目标分布**：\(p(x) = 0.3 \mathcal{N}(4,1) + 0.7 \mathcal{N}(7,0.5)\)
- **待估计量**：均值 \(\mathbb{E}_p[X]\)（真实值可由公式直接计算：\(0.3\times 4 + 0.7\times 7 = 6.1\)），也可以估计更高阶矩或尾部概率，这里以均值为例。
- **比较方法**：
  1. **直接蒙特卡洛**：直接从混合分布采样（实践中简单高效，此处作为基准）。
  2. **拒绝采样**：选择一个容易采样的提议分布 \(q(x)\) 和常数 \(M\)，使得 \(M q(x) \ge p(x)\) 对所有 \(x\) 成立。
  3. **重要性采样**：从一个提议分布 \(q(x)\) 采样，用权重 \(w(x) = p(x)/q(x)\) 修正。

### 实验设计

- 对每种方法，在不同样本量 \(N\) 下重复运行多次，计算估计的方差和偏差。
- 记录拒绝采样的**接受率**。
- 绘制样本直方图与目标密度的对比。
- 绘制方差随样本量变化的曲线。

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

# Set random seed for reproducibility
np.random.seed(42)

# ====================================================
# 1. Define target distribution p(x) and its components
# ====================================================
def p_density(x):
    """Mixture of Gaussians: 0.3*N(4,1) + 0.7*N(7,0.5)."""
    return 0.3 * stats.norm.pdf(x, loc=4, scale=1) + 0.7 * stats.norm.pdf(x, loc=7, scale=np.sqrt(0.5))

def sample_p(N):
    """Direct sampling from mixture."""
    # Choose component: 0 with prob 0.3, 1 with prob 0.7
    comp = np.random.choice([0, 1], size=N, p=[0.3, 0.7])
    # Sample from chosen components
    samples = np.where(comp == 0,
                       np.random.normal(4, 1, N),
                       np.random.normal(7, np.sqrt(0.5), N))
    return samples

# True mean of p(x)
true_mean = 0.3 * 4 + 0.7 * 7   # = 6.1
print(f"True mean: {true_mean:.4f}")

# ====================================================
# 2. Rejection Sampling setup
# ====================================================
# Proposal q_rs(x): Normal distribution that covers the support of p(x).
# We choose a wide Gaussian: N(6, 3^2)
q_rs_loc = 6.0
q_rs_scale = 3.0
M = 2.5   # Scaling constant such that M * q_rs(x) >= p(x) for all x
          # This value was found by numerically checking the maximum ratio p(x)/q_rs(x)
          # max(p(x) / q_rs(x)) ≈ 2.3, so M=2.5 is safe

def sample_q_rs(N):
    """Sample from proposal for rejection sampling."""
    return np.random.normal(q_rs_loc, q_rs_scale, size=N)

def rejection_sampling(N_desired):
    """
    Generate exactly N_desired accepted samples from p using rejection sampling.
    Returns: accepted samples, total number of proposals generated.
    """
    accepted = []
    total_proposed = 0
    while len(accepted) < N_desired:
        # Propose a batch
        batch_size = max(1, int((N_desired - len(accepted)) * M * 1.2))  # slightly overgenerate
        x_prop = sample_q_rs(batch_size)
        u = np.random.uniform(0, M * stats.norm.pdf(x_prop, loc=q_rs_loc, scale=q_rs_scale))
        accept = u <= p_density(x_prop)
        accepted.extend(x_prop[accept])
        total_proposed += batch_size
    return np.array(accepted[:N_desired]), total_proposed

# ====================================================
# 3. Importance Sampling setup
# ====================================================
# Proposal q_is(x): We use a Student's t-distribution with heavy tails to avoid weight explosion.
# df=5, loc=6, scale=2
def sample_q_is(N):
    return stats.t.rvs(df=5, loc=6, scale=2, size=N)

def q_is_density(x):
    return stats.t.pdf(x, df=5, loc=6, scale=2)

# ====================================================
# 4. Simulation parameters
# ====================================================
N_list = [100, 300, 1000, 3000, 10000]  # sample sizes to test
M_reps = 500   # number of repetitions for variance estimation

# Storage for results
var_direct = []
var_reject = []
var_is = []
mean_direct = []
mean_reject = []
mean_is = []
accept_rates = []   # for rejection sampling

# ====================================================
# 5. Run experiments
# ====================================================
print("Running experiments...")
for N in N_list:
    est_direct = np.zeros(M_reps)
    est_reject = np.zeros(M_reps)
    est_is = np.zeros(M_reps)
    total_proposed_list = []
    
    for rep in range(M_reps):
        # Direct MC
        x_dir = sample_p(N)
        est_direct[rep] = np.mean(x_dir)
        
        # Rejection sampling
        x_rej, total_prop = rejection_sampling(N)
        est_reject[rep] = np.mean(x_rej)
        total_proposed_list.append(total_prop)
        
        # Importance sampling
        x_is = sample_q_is(N)
        w = p_density(x_is) / q_is_density(x_is)
        est_is[rep] = np.sum(x_is * w) / np.sum(w)   # self-normalized IS
        
    # Store statistics
    var_direct.append(np.var(est_direct, ddof=1))
    var_reject.append(np.var(est_reject, ddof=1))
    var_is.append(np.var(est_is, ddof=1))
    
    mean_direct.append(np.mean(est_direct))
    mean_reject.append(np.mean(est_reject))
    mean_is.append(np.mean(est_is))
    
    accept_rate = np.mean([N / tp for tp in total_proposed_list])
    accept_rates.append(accept_rate)
    print(f"N={N:5d} | Accept rate: {accept_rate:.3f} | Var direct={var_direct[-1]:.3e}, reject={var_reject[-1]:.3e}, IS={var_is[-1]:.3e}")

# ====================================================
# 6. Visualizations
# ====================================================
# Figure 1: Sample histograms vs. target density
x_plot = np.linspace(0, 12, 500)
p_plot = p_density(x_plot)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

# Direct sampling
N_hist = 10000
x_dir_hist = sample_p(N_hist)
axes1[0].hist(x_dir_hist, bins=60, density=True, alpha=0.6, color='blue', label='Direct MC')
axes1[0].plot(x_plot, p_plot, 'r-', linewidth=2, label='Target p(x)')
axes1[0].set_title('Direct Monte Carlo (N=10000)')
axes1[0].set_xlabel('x')
axes1[0].set_ylabel('Density')
axes1[0].legend()

# Rejection sampling
x_rej_hist, _ = rejection_sampling(N_hist)
axes1[1].hist(x_rej_hist, bins=60, density=True, alpha=0.6, color='green', label='Rejection Sampling')
axes1[1].plot(x_plot, p_plot, 'r-', linewidth=2, label='Target p(x)')
axes1[1].set_title(f'Rejection Sampling (N={N_hist})\nAccept rate ≈ {accept_rates[-1]:.2f}')
axes1[1].set_xlabel('x')
axes1[1].set_ylabel('Density')
axes1[1].legend()

# Importance sampling (weighted histogram)
x_is_hist = sample_q_is(N_hist * 5)  # oversample for better weighted histogram
w_hist = p_density(x_is_hist) / q_is_density(x_is_hist)
axes1[2].hist(x_is_hist, bins=60, weights=w_hist, density=True, alpha=0.6, color='red', label='IS weighted')
axes1[2].plot(x_plot, p_plot, 'r-', linewidth=2, label='Target p(x)')
axes1[2].set_title('Importance Sampling (weighted histogram)')
axes1[2].set_xlabel('x')
axes1[2].set_ylabel('Density')
axes1[2].legend()
axes1[2].set_ylim(0, 0.5)   # target peak is ~0.45

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment2_histograms.png'), dpi=150)
plt.show()

# Figure 2: Variance vs. sample size
fig2, ax = plt.subplots(figsize=(8, 6))
ax.loglog(N_list, var_direct, 'o-', label='Direct MC', linewidth=2, markersize=8)
ax.loglog(N_list, var_reject, 's-', label='Rejection Sampling', linewidth=2, markersize=8)
ax.loglog(N_list, var_is, '^-', label='Importance Sampling', linewidth=2, markersize=8)
# Reference 1/N line
ref = np.array(N_list)
ax.loglog(ref, var_direct[0] * (ref[0] / ref), 'k--', alpha=0.5, label=r'$\propto 1/N$')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Empirical variance', fontsize=12)
ax.set_title('Variance Comparison (log-log scale)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment2_variance.png'), dpi=150)
plt.show()

# Figure 3: Mean estimate convergence
fig3, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(N_list, mean_direct, 'o-', label='Direct MC', linewidth=2, markersize=8)
ax.semilogx(N_list, mean_reject, 's-', label='Rejection Sampling', linewidth=2, markersize=8)
ax.semilogx(N_list, mean_is, '^-', label='Importance Sampling', linewidth=2, markersize=8)
ax.axhline(y=true_mean, color='k', linestyle='--', label=f'True mean = {true_mean}')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Mean estimate', fontsize=12)
ax.set_title('Convergence of Estimates', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment2_means.png'), dpi=150)
plt.show()

# ====================================================
# 7. Print summary
# ====================================================
print("\n--- Final Summary ---")
print(f"True mean: {true_mean:.4f}")
print("Acceptance rates (rejection sampling):")
for N, ar in zip(N_list, accept_rates):
    print(f"  N={N:5d} : {ar:.3f}")
```

### 代码说明

1. **目标分布定义**：`p_density` 实现混合高斯密度，`sample_p` 实现直接从混合分布采样（分量选择 + 高斯采样）。

2. **拒绝采样设置**：
   - 提议分布选择 \(\mathcal{N}(6, 3^2)\)，其尾部足够覆盖两个分量。
   - 常数 \(M=2.5\) 通过数值检查 `max(p(x)/q(x)) ≈ 2.3` 确定。
   - 函数 `rejection_sampling(N)` 返回恰好 N 个接受样本及总提议数，用于计算接受率。

3. **重要性采样设置**：
   - 提议分布选用自由度 5、位置 6、尺度 2 的学生 t 分布，重尾特性避免权重极端化。
   - 使用**自归一化重要性采样**（加权平均），更贴近实际应用。

4. **实验循环**：
   - 对每个样本量 N，重复 M_reps=500 次独立实验。
   - 记录每次估计值，计算方差和均值。
   - 输出接受率及方差信息。

5. **可视化**：
   - 图一：三种方法的样本分布（或加权分布）与目标密度的对比直方图。
   - 图二：方差随 N 变化的对数图，评估效率。
   - 图三：均值估计收敛情况。

### 预期观察

- **直接 MC**：方差标准 \(O(1/N)\) 行为，作为性能基准。
- **拒绝采样**：若提议分布选择合适，接受率尚可（约 40%～60%），方差与直接 MC 接近（因最终样本均来自 p）。但生成样本的计算开销随接受率反比增加。
- **重要性采样**：自归一化 IS 有轻微偏差，但方差可能略高于直接 MC（取决于 q 的匹配程度）。若 q 与 p 匹配良好，方差可接近直接 MC；若 q 尾部更重，权重退化较轻。

通过本实验，您将直观理解三种方法在非标准分布采样中的适用场景与权衡。若需调整参数或增加估计函数（如尾部概率），可轻松扩展代码。