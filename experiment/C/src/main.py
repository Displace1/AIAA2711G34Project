import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create results directory
results_dir = '../results'
os.makedirs(results_dir, exist_ok=True)

# Set random seed
np.random.seed(42)

# ====================================================
# 1. Define target mixture distribution p(x)
# ====================================================
def p_density(x):
    """Mixture: 0.9 * N(0,1) + 0.1 * N(10, 0.25)"""
    return 0.9 * stats.norm.pdf(x, loc=0, scale=1) + 0.1 * stats.norm.pdf(x, loc=10, scale=0.5)  # sd=sqrt(0.25)=0.5

# Function to integrate for true value: x * p(x) * I(x>8)
def integrand(x):
    return x * p_density(x)

# Compute true integral using numerical integration
true_integral, _ = quad(integrand, 8, np.inf)
print(f"True integral E[X * I(X>8)]: {true_integral:.8f}")

# Indicator function
def f(x):
    return x * (x > 8)

# ====================================================
# 2. Define proposals
# ====================================================
# Good proposal: t-distribution centered at 10, scale=1.5, df=3 (heavy tail)
def sample_q_good(N):
    return stats.t.rvs(df=3, loc=8, scale=1.5, size=N)

def q_good_density(x):
    return stats.t.pdf(x, df=3, loc=8, scale=1.5)

# Bad proposal: Normal(10, 1) (thin tail)
def sample_q_bad(N):
    return np.random.normal(8, 1, size=N)

def q_bad_density(x):
    return stats.norm.pdf(x, loc=8, scale=1)

# ====================================================
# 3. Simulation parameters
# ====================================================
N_list = [2000, 5000, 10000, 20000, 50000, 100000, 200000]
M_reps = 200

var_good = []
var_bad = []
mean_good = []
mean_bad = []
max_weight_good = []
max_weight_bad = []

# ====================================================
# 4. Run experiments
# ====================================================
print("\nEstimating E[X * I(X>8)] under mixture p(x)...")
for N in N_list:
    est_good = np.zeros(M_reps)
    est_bad = np.zeros(M_reps)
    w_max_good = []
    w_max_bad = []
    
    for rep in range(M_reps):
        # Good q
        x_g = sample_q_good(N)
        w_g = p_density(x_g) / q_good_density(x_g)
        est_good[rep] = np.mean(f(x_g) * w_g)
        w_max_good.append(np.max(w_g))
        
        # Bad q
        x_b = sample_q_bad(N)
        w_b = p_density(x_b) / q_bad_density(x_b)
        est_bad[rep] = np.mean(f(x_b) * w_b)
        w_max_bad.append(np.max(w_b))
    
    var_good.append(np.var(est_good, ddof=1))
    var_bad.append(np.var(est_bad, ddof=1))
    mean_good.append(np.mean(est_good))
    mean_bad.append(np.mean(est_bad))
    max_weight_good.append(np.mean(w_max_good))
    max_weight_bad.append(np.mean(w_max_bad))
    
    print(f"N={N:7d} | Good q var={var_good[-1]:.4e} | Bad q var={var_bad[-1]:.4e} | "
          f"Avg max w: good={max_weight_good[-1]:.2e}, bad={max_weight_bad[-1]:.2e}")

# ====================================================
# 5. Visualizations
# ====================================================
# Figure 1: Variance vs. N
fig1, ax = plt.subplots(figsize=(8, 6))
ax.loglog(N_list, var_good, 'o-', label='IS with t (good q)', linewidth=2, markersize=8)
ax.loglog(N_list, var_bad, 's-', label='IS with Normal(10,1) (bad q)', linewidth=2, markersize=8)
ref = np.array(N_list)
ax.loglog(ref, var_good[0] * (ref[0] / ref), 'k--', alpha=0.5, label=r'$\propto 1/N$')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Empirical variance', fontsize=12)
ax.set_title('Variance Comparison: Mixture with Rare Event', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_variance.png'), dpi=150)
plt.show()

# Figure 2: Weight histograms for tail region (X>8)
N_demo = 100000
x_g_demo = sample_q_good(N_demo)
x_b_demo = sample_q_bad(N_demo)
w_g_demo = p_density(x_g_demo) / q_good_density(x_g_demo)
w_b_demo = p_density(x_b_demo) / q_bad_density(x_b_demo)

w_g_tail = w_g_demo[x_g_demo > 8]
w_b_tail = w_b_demo[x_b_demo > 8]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

axes2[0].hist(w_g_tail, bins=50, alpha=0.7, color='green', edgecolor='black')
axes2[0].set_title(f'Good q (t dist): Weights for X>8 (N={N_demo})')
axes2[0].set_xlabel('Importance weight w(x)')
axes2[0].set_ylabel('Frequency')
axes2[0].axvline(x=np.mean(w_g_tail), color='red', linestyle='--', label=f'mean={np.mean(w_g_tail):.3f}')
axes2[0].legend()
axes2[0].set_xlim([0, np.percentile(w_g_tail, 99)])

axes2[1].hist(w_b_tail, bins=50, alpha=0.7, color='red', edgecolor='black')
axes2[1].set_title(f'Bad q (Normal): Weights for X>8 (N={N_demo})')
axes2[1].set_xlabel('Importance weight w(x)')
axes2[1].set_ylabel('Frequency')
axes2[1].axvline(x=np.mean(w_b_tail), color='blue', linestyle='--', label=f'mean={np.mean(w_b_tail):.3f}')
axes2[1].legend()
axes2[1].set_xlim([0, np.percentile(w_b_tail, 99)])

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_weights.png'), dpi=150)
plt.show()

# Figure 3: Convergence of estimates
fig3, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(N_list, mean_good, 'o-', label='IS with t (good q)', linewidth=2, markersize=8)
ax.semilogx(N_list, mean_bad, 's-', label='IS with Normal(10,1) (bad q)', linewidth=2, markersize=8)
ax.axhline(y=true_integral, color='k', linestyle='--', label=f'True value = {true_integral:.5f}')
ax.set_xlabel('Number of samples N', fontsize=12)
ax.set_ylabel('Mean estimate', fontsize=12)
ax.set_title('Convergence of Integral Estimates', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_means.png'), dpi=150)
plt.show()

# ====================================================
# 5.5 Density comparison of proposals and target
# ====================================================
x_plot = np.linspace(-5, 15, 1000)
p_vals = p_density(x_plot)
q_good_vals = q_good_density(x_plot)
q_bad_vals = q_bad_density(x_plot)

fig4, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, p_vals, 'k-', linewidth=2, label='Target p(x) = 0.99 N(0,1) + 0.01 N(10,0.25)')
ax.plot(x_plot, q_good_vals, 'g-', linewidth=2, label='Good q: t(df=3, loc=10, scale=1.5)')
ax.plot(x_plot, q_bad_vals, 'r-', linewidth=2, label='Bad q: N(10,1)')

# Highlight the tail region x > 8
ax.axvspan(8, 15, alpha=0.2, color='gray', label='Region of interest (X > 8)')
ax.axvline(x=8, color='gray', linestyle='--', alpha=0.7)

ax.set_xlim(-5, 15)
# Use a broken y-axis or log scale for clarity since main peak is at 0 (density ~0.4) and secondary peak at 10 (density ~0.08)
ax.set_ylim(0, 0.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Comparison of Target and Proposal Distributions', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Inset: zoom on tail region with log scale
ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper right')
x_tail = np.linspace(8, 15, 500)
ax_inset.semilogy(x_tail, p_density(x_tail), 'k-', linewidth=2)
ax_inset.semilogy(x_tail, q_good_density(x_tail), 'g-', linewidth=2)
ax_inset.semilogy(x_tail, q_bad_density(x_tail), 'r-', linewidth=2)
ax_inset.set_xlim(8, 15)
ax_inset.set_ylim(1e-8, 1e-1)
ax_inset.set_xlabel('x (tail region)', fontsize=8)
ax_inset.set_ylabel('Density (log)', fontsize=8)
ax_inset.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'tail_exp_density_comparison.png'), dpi=150)
plt.show()

# ====================================================
# 6. Print summary
# ====================================================
print("\n--- Summary ---")
print(f"True integral E[X * I(X>8)]: {true_integral:.8f}")
print("Good q (t, heavy-tailed) should exhibit significantly lower variance than bad q (normal).")
print("This demonstrates the critical role of heavy-tailed proposals for rare-event estimation.")