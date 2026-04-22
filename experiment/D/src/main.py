import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

# Create results directory if not exists
os.makedirs('../results', exist_ok=True)

np.random.seed(42)

# ========== Parameters ==========
m = 2000                # number of samples
a = 0.5                 # mean shift (for non-ideal proposal)
dims = [1, 2, 5, 10, 50, 100, 1000]   # dimensions to test

# Store results
ess_ideal = []          # ESS ratio for ideal proposal
ess_offset = []         # ESS ratio for offset proposal

# ========== Experiment loop ==========
for d in dims:
    # Target distribution p = N(0, I_d)
    mean_p = np.zeros(d)
    cov_p = np.eye(d)
    
    # ---- Ideal proposal: q_ideal = p (exactly the same) ----
    samples_ideal = np.random.multivariate_normal(mean_p, cov_p, size=m)
    log_p_ideal = multivariate_normal.logpdf(samples_ideal, mean=mean_p, cov=cov_p)
    log_q_ideal = multivariate_normal.logpdf(samples_ideal, mean=mean_p, cov=cov_p)
    w_ideal = np.exp(log_p_ideal - log_q_ideal)   # should be all ones, slight numerical variation
    ess_ideal_val = (np.sum(w_ideal))**2 / np.sum(w_ideal**2)
    ess_ratio_ideal = ess_ideal_val / m
    ess_ideal.append(ess_ratio_ideal)
    
    # ---- Offset proposal: q_offset = N(a, I_d), all components shifted by a ----
    mean_q = np.full(d, a)
    cov_q = np.eye(d)
    samples_offset = np.random.multivariate_normal(mean_q, cov_q, size=m)
    log_p_offset = multivariate_normal.logpdf(samples_offset, mean=mean_p, cov=cov_p)
    log_q_offset = multivariate_normal.logpdf(samples_offset, mean=mean_q, cov=cov_q)
    log_w_offset = log_p_offset - log_q_offset
    w_offset = np.exp(log_w_offset)
    ess_offset_val = (np.sum(w_offset))**2 / np.sum(w_offset**2)
    ess_ratio_offset = ess_offset_val / m
    ess_offset.append(ess_ratio_offset)
    
    print(f"d={d:3d} | Ideal proposal ESS ratio={ess_ratio_ideal:.4f} | Offset proposal ESS ratio={ess_ratio_offset:.4f}")

# ========== Plot comparison ==========
plt.figure(figsize=(8, 5))
plt.semilogy(dims, ess_ideal, 'gs-', linewidth=2, markersize=8, label='Ideal proposal (q = p)')
plt.semilogy(dims, ess_offset, 'ro-', linewidth=2, markersize=8, label=f'Offset proposal (mean shift a={a})')
plt.xlabel('Dimension d', fontsize=12)
plt.ylabel('ESS ratio (ESS/m)', fontsize=12)
plt.title('Importance Sampling Efficiency: Ideal vs Offset Proposal', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
# Save to ../results directory
plt.savefig('../results/importance_sampling_ideal_vs_offset.png', dpi=150)
plt.show()