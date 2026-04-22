## Experiment 1: Direct Monte Carlo vs. Importance Sampling (Basic Comparison)

**Goal**: Gain intuition for variance reduction by concentrating samples in "important" regions.

**Problem**: Compute the integral
\[
I = \int_0^1 e^x \, dx = e - 1 \approx 1.718281828459045
\]
Although it has an analytical solution, we deliberately express it as an expectation:
\[
I = \int_0^1 e^x \cdot 1 \, dx = \mathbb{E}_{p}[f(X)],
\]
where \(p(x) = \mathbf{1}_{[0,1]}\) is the uniform density on \([0,1]\) and \(f(x) = e^x\).

We will compare three estimators:

1. **Direct Monte Carlo (MC)**  
   Sample \(X_i \sim U(0,1)\), estimate \(\hat{I}_{\text{MC}} = \frac{1}{N}\sum_{i=1}^N e^{X_i}\).

2. **Optimal Importance Sampling (IS-opt)**  
   Use the proposal \(q_{\text{opt}}(x) = \frac{e^x}{e-1}\) on \([0,1]\).  
   Importance weights: \(w(x) = \frac{p(x)}{q_{\text{opt}}(x)} = \frac{e-1}{e^x}\).  
   Estimate: \(\hat{I}_{\text{IS-opt}} = \frac{1}{N}\sum_{i=1}^N f(X_i) w(X_i)\).

3. **Suboptimal Importance Sampling (IS-bad)**  
   Use a linear proposal \(q_{\text{bad}}(x) = 2x\) on \([0,1]\).  
   Importance weights: \(w(x) = \frac{1}{2x}\).  
   Estimate: \(\hat{I}_{\text{IS-bad}} = \frac{1}{N}\sum_{i=1}^N f(X_i) w(X_i)\).

We will run many independent repetitions for different sample sizes \(N\) and analyze the variance of each estimator.

---

### Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# True value of the integral
I_true = np.exp(1) - 1
print(f"True integral value: {I_true:.8f}")

# ====================================================
# 1. Define sampling functions and estimators
# ====================================================

def sample_uniform(N):
    """Sample from U(0,1)."""
    return np.random.uniform(0, 1, size=N)

def sample_optimal_q(N):
    """
    Sample from q_opt(x) = e^x / (e-1) on [0,1].
    Using inverse transform sampling:
    CDF F(x) = (e^x - 1) / (e - 1)
    Inverse: F^{-1}(u) = ln(1 + u*(e-1))
    """
    u = np.random.uniform(0, 1, size=N)
    return np.log(1 + u * (np.exp(1) - 1))

def sample_bad_q(N):
    """
    Sample from q_bad(x) = 2x on [0,1].
    CDF: F(x) = x^2
    Inverse: sqrt(u)
    """
    u = np.random.uniform(0, 1, size=N)
    return np.sqrt(u)

def direct_mc_estimate(samples):
    """f(x) = e^x."""
    return np.mean(np.exp(samples))

def importance_sampling_estimate(samples, weights):
    """Weighted average of f(x) = e^x."""
    return np.mean(np.exp(samples) * weights)

# ====================================================
# 2. Simulation parameters
# ====================================================

# Sample sizes to test
N_list = [10, 30, 100, 300, 1000, 3000, 10000]

# Number of independent repetitions for variance estimation
M = 1000

# Arrays to store results
var_mc = []
var_is_opt = []
var_is_bad = []
mean_mc = []
mean_is_opt = []
mean_is_bad = []

# ====================================================
# 3. Run experiments
# ====================================================

for N in N_list:
    estimates_mc = np.zeros(M)
    estimates_is_opt = np.zeros(M)
    estimates_is_bad = np.zeros(M)
    
    for rep in range(M):
        # Direct MC
        x_mc = sample_uniform(N)
        estimates_mc[rep] = direct_mc_estimate(x_mc)
        
        # IS with optimal q
        x_opt = sample_optimal_q(N)
        # p(x) = 1, q_opt(x) = e^x / (e-1)
        w_opt = (np.exp(1) - 1) / np.exp(x_opt)
        estimates_is_opt[rep] = importance_sampling_estimate(x_opt, w_opt)
        
        # IS with bad q
        x_bad = sample_bad_q(N)
        # p(x) = 1, q_bad(x) = 2x
        w_bad = 1.0 / (2.0 * x_bad)
        estimates_is_bad[rep] = importance_sampling_estimate(x_bad, w_bad)
    
    # Store statistics
    var_mc.append(np.var(estimates_mc, ddof=1))
    var_is_opt.append(np.var(estimates_is_opt, ddof=1))
    var_is_bad.append(np.var(estimates_is_bad, ddof=1))
    
    mean_mc.append(np.mean(estimates_mc))
    mean_is_opt.append(np.mean(estimates_is_opt))
    mean_is_bad.append(np.mean(estimates_is_bad))

# ====================================================
# 4. Visualization
# ====================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Variance vs. N (log-log scale)
ax = axes[0]
ax.loglog(N_list, var_mc, 'o-', label='Direct MC', linewidth=2, markersize=8)
ax.loglog(N_list, var_is_opt, 's-', label='IS (optimal q)', linewidth=2, markersize=8)
ax.loglog(N_list, var_is_bad, '^-', label='IS (bad q)', linewidth=2, markersize=8)
# Add reference line with slope -1 (1/N convergence)
ref_N = np.array(N_list)
ax.loglog(ref_N, var_mc[0] * (ref_N[0] / ref_N), 'k--', alpha=0.5, label=r'$\propto 1/N$')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Empirical variance', fontsize=12)
ax.set_title('Variance Comparison (log-log scale)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Mean estimate convergence
ax = axes[1]
ax.semilogx(N_list, mean_mc, 'o-', label='Direct MC', linewidth=2, markersize=8)
ax.semilogx(N_list, mean_is_opt, 's-', label='IS (optimal q)', linewidth=2, markersize=8)
ax.semilogx(N_list, mean_is_bad, '^-', label='IS (bad q)', linewidth=2, markersize=8)
ax.axhline(y=I_true, color='k', linestyle='--', label='True value', alpha=0.7)
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Mean estimate', fontsize=12)
ax.set_title('Convergence of Estimates', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment1_results.png', dpi=150)
plt.show()

# ====================================================
# 5. Print summary for a specific N
# ====================================================

# Pick a moderate N for detailed inspection
N_demo = 1000
x_mc_demo = sample_uniform(N_demo)
x_opt_demo = sample_optimal_q(N_demo)
x_bad_demo = sample_bad_q(N_demo)

w_opt_demo = (np.exp(1) - 1) / np.exp(x_opt_demo)
w_bad_demo = 1.0 / (2.0 * x_bad_demo)

est_mc = direct_mc_estimate(x_mc_demo)
est_opt = importance_sampling_estimate(x_opt_demo, w_opt_demo)
est_bad = importance_sampling_estimate(x_bad_demo, w_bad_demo)

print("\n--- Single run with N = {} ---".format(N_demo))
print(f"Direct MC estimate : {est_mc:.6f}")
print(f"IS (optimal) estimate: {est_opt:.6f}")
print(f"IS (bad) estimate    : {est_bad:.6f}")

# Show weight distribution for bad q
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

axes2[0].hist(w_opt_demo, bins=50, alpha=0.7, color='green', edgecolor='black')
axes2[0].set_title('Weights for optimal q (N={})'.format(N_demo))
axes2[0].set_xlabel('Importance weight w(x)')
axes2[0].set_ylabel('Frequency')
axes2[0].axvline(x=np.mean(w_opt_demo), color='red', linestyle='--', label=f'mean = {np.mean(w_opt_demo):.3f}')
axes2[0].legend()

axes2[1].hist(w_bad_demo, bins=50, alpha=0.7, color='red', edgecolor='black')
axes2[1].set_title('Weights for bad q (N={})'.format(N_demo))
axes2[1].set_xlabel('Importance weight w(x)')
axes2[1].set_ylabel('Frequency')
axes2[1].axvline(x=np.mean(w_bad_demo), color='blue', linestyle='--', label=f'mean = {np.mean(w_bad_demo):.3f}')
axes2[1].legend()
# Note: some weights can be very large because q_bad(x) is small near x=0
axes2[1].set_xlim([0, 20])  # Zoom in to see bulk

plt.tight_layout()
plt.savefig('experiment1_weights.png', dpi=150)
plt.show()

print("\n--- Weight statistics (N={}) ---".format(N_demo))
print(f"Optimal q: mean weight = {np.mean(w_opt_demo):.4f}, std = {np.std(w_opt_demo):.4f}, max = {np.max(w_opt_demo):.4f}")
print(f"Bad q    : mean weight = {np.mean(w_bad_demo):.4f}, std = {np.std(w_bad_demo):.4f}, max = {np.max(w_bad_demo):.4f}")
```

---

### Expected Observations and Explanation

When you run the code, you should see:

1. **Variance Plot (log‑log)**  
   - The variance of **Direct MC** decreases as \(1/N\) (a straight line with slope –1).  
   - The variance of **IS with optimal \(q\)** is **extremely small** (often orders of magnitude lower) for all \(N\). In fact, the optimal proposal achieves **zero variance** in theory because \(f(x) w(x)\) is constant for all \(x\). Numerical noise is only due to floating‑point precision.  
   - The variance of **IS with bad \(q\)** is **higher** than that of Direct MC, demonstrating that a poorly chosen proposal can *increase* variance instead of reducing it.

2. **Mean Estimate Convergence**  
   - All three estimators converge to the true value \(e-1 \approx 1.71828\).  
   - The optimal IS estimate is extremely stable even for very small \(N\), while the bad IS estimate fluctuates heavily.

3. **Weight Distribution Histograms**  
   - For the optimal \(q\), the weights are tightly clustered around a constant value (since \(f(x) w(x)\) is constant).  
   - For the bad \(q\), the weight distribution has a long tail toward large values (some weights become huge when \(x\) is close to 0), which inflates the variance. The mean weight remains 1 (unbiasedness), but the variance of the estimator suffers greatly.

This experiment vividly illustrates the power of a well‑matched proposal distribution and the danger of using a proposal with thinner tails than the target.

---

### How to Extend This Experiment

- **Try other suboptimal proposals**, e.g., \(q(x) \propto x^3\) or a Beta distribution with mismatched parameters.  
- **Plot the sample locations** to see that optimal \(q\) samples more densely where \(e^x\) is large.  
- **Compute the theoretical variance** using the formulas derived in the explanation and compare with empirical results.

If you have any questions about the code or the underlying theory, feel free to ask!