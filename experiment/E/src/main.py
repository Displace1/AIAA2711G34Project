import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz
import time
import os

os.makedirs('../results', exist_ok=True)
np.random.seed(42)

# ========== Parameters ==========
rho = 0.9                # correlation coefficient
dims = [2, 5, 10, 20, 50]
m = 5000                 # number of samples (IS) or chain length (Gibbs)
n_repeats = 30           # number of independent runs

# Storage
is_ess_ratios = []
gibbs_ess_ratios = []
is_mse = []
gibbs_mse = []
is_time = []
gibbs_time = []

# ========== Helper functions ==========
def build_ar1_cov(d, rho):
    """AR(1) covariance matrix: Sigma[i,j] = rho^{|i-j|}"""
    indices = np.arange(d)
    cov = toeplitz(rho ** indices)
    return cov

def gibbs_sampler_mvn(cov, m, burnin=1000):
    """
    Gibbs sampler for multivariate normal N(0, cov).
    Returns samples after burnin.
    """
    d = cov.shape[0]
    # Compute conditional parameters: 
    # For variable i given all others, mean = - (precision_ii)^{-1} * (precision_ij * x_j) sum over j≠i
    # Alternatively, use covariance block formula.
    # We'll precompute the precision matrix for efficiency.
    prec = np.linalg.inv(cov)
    # For each variable i, the conditional variance = 1 / prec[i,i]
    cond_var = 1.0 / np.diag(prec)
    # The conditional mean depends on others: 
    # E[x_i | x_{-i}] = - (1/prec[i,i]) * sum_{j≠i} prec[i,j] * x_j
    # We'll compute on the fly.
    
    # Initialize x at zero (or random)
    x = np.zeros(d)
    samples = np.zeros((m, d))
    for t in range(m + burnin):
        # Loop over each dimension in random order (systematic scan)
        for i in range(d):
            # Compute conditional mean
            sum_others = -np.dot(prec[i, :], x) + prec[i, i] * x[i]
            mean_i = sum_others / prec[i, i]
            # Sample from conditional normal
            x[i] = np.random.normal(mean_i, np.sqrt(cond_var[i]))
        if t >= burnin:
            samples[t - burnin] = x.copy()
    return samples

def effective_sample_size(samples):
    """Compute ESS for MCMC samples (each column is a dimension)"""
    n = samples.shape[0]
    # For simplicity, compute ESS for first dimension (or average over dimensions)
    # We'll use first dimension for comparison
    chain = samples[:, 0]
    # Compute autocorrelation up to lag where it becomes negative
    autocorr = np.correlate(chain - chain.mean(), chain - chain.mean(), mode='full')
    autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
    # Find first lag where autocorr < 0.05, or limit to 500
    max_lag = min(500, n // 2)
    gamma = autocorr[1:max_lag+1]
    # Compute ESS = n / (1 + 2 * sum_{l=1}^∞ gamma_l)
    if np.any(gamma <= 0):
        first_neg = np.argmax(gamma <= 0)
        gamma = gamma[:first_neg]
    ess = n / (1 + 2 * np.sum(gamma))
    return ess

# ========== Main experiment ==========
for d in dims:
    print(f"\n--- Dimension d = {d} ---")
    cov = build_ar1_cov(d, rho)
    mean = np.zeros(d)
    
    # ---- Importance Sampling ----
    # Proposal: independent N(0, I) (ignores correlation)
    mean_q = np.zeros(d)
    cov_q = np.eye(d)
    is_ess_list = []
    is_mse_list = []
    start = time.time()
    for rep in range(n_repeats):
        samples_q = np.random.multivariate_normal(mean_q, cov_q, size=m)
        # Compute log weights
        log_p = multivariate_normal.logpdf(samples_q, mean=mean, cov=cov)
        log_q = multivariate_normal.logpdf(samples_q, mean=mean_q, cov=cov_q)
        log_w = log_p - log_q
        w = np.exp(log_w)
        # ESS
        ess = (np.sum(w))**2 / np.sum(w**2)
        is_ess_list.append(ess / m)
        # Estimate mean of first component (true=0)
        weighted_mean = np.sum(w * samples_q[:, 0]) / np.sum(w)
        is_mse_list.append((weighted_mean - 0)**2)
    is_time.append(time.time() - start)
    is_ess_ratios.append(np.mean(is_ess_list))
    is_mse.append(np.mean(is_mse_list))
    
    # ---- Gibbs Sampling ----
    gibbs_ess_list = []
    gibbs_mse_list = []
    start = time.time()
    for rep in range(n_repeats):
        samples = gibbs_sampler_mvn(cov, m, burnin=1000)
        # ESS for first dimension
        ess = effective_sample_size(samples)
        gibbs_ess_list.append(ess / m)
        # Estimate mean of first component
        est_mean = np.mean(samples[:, 0])
        gibbs_mse_list.append((est_mean - 0)**2)
    gibbs_time.append(time.time() - start)
    gibbs_ess_ratios.append(np.mean(gibbs_ess_list))
    gibbs_mse.append(np.mean(gibbs_mse_list))
    
    print(f"IS:   ESS ratio = {is_ess_ratios[-1]:.4e}, MSE = {is_mse[-1]:.4e}")
    print(f"Gibbs: ESS ratio = {gibbs_ess_ratios[-1]:.4f}, MSE = {gibbs_mse[-1]:.4e}")

# ========== Plotting ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. ESS ratio vs dimension
ax = axes[0,0]
ax.semilogy(dims, is_ess_ratios, 'bo-', label='IS (proposal N(0,I))', linewidth=2)
ax.semilogy(dims, gibbs_ess_ratios, 'rs-', label='Gibbs sampler', linewidth=2)
ax.set_xlabel('Dimension d', fontsize=12)
ax.set_ylabel('ESS ratio (ESS / m)', fontsize=12)
ax.set_title('Effective Sample Size Ratio', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# 2. MSE of mean estimate (first component)
ax = axes[0,1]
ax.semilogy(dims, is_mse, 'bo-', label='IS', linewidth=2)
ax.semilogy(dims, gibbs_mse, 'rs-', label='Gibbs', linewidth=2)
ax.set_xlabel('Dimension d', fontsize=12)
ax.set_ylabel('MSE (estimate of μ₁=0)', fontsize=12)
ax.set_title('Mean Squared Error', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# 3. Computational time (log scale)
ax = axes[1,0]
ax.plot(dims, is_time, 'bo-', label='IS', linewidth=2)
ax.plot(dims, gibbs_time, 'rs-', label='Gibbs', linewidth=2)
ax.set_xlabel('Dimension d', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Computation Time (30 repeats)', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# 4. ESS ratio for IS on log scale to show decay
ax = axes[1,1]
ax.semilogy(dims, is_ess_ratios, 'bo-', label='IS', linewidth=2)
ax.set_xlabel('Dimension d', fontsize=12)
ax.set_ylabel('ESS ratio (log scale)', fontsize=12)
ax.set_title('IS: Exponential Decay of ESS', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig('../results/is_vs_gibbs_highdim.png', dpi=150)
plt.show()