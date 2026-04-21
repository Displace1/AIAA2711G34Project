"""
实验A：基础正确性验证（1D高斯）
目标：估计 E_{p}[X^2]，其中 p = N(0,1)，真值 = 1
比较三种提议分布：
  - q1: N(0, 1.2^2)  （好）
  - q2: N(0, 2^2)    （一般）
  - q3: N(3, 1)      （差）
样本量网格: N = [100, 300, 1000, 3000, 10000]
重复次数: 30次
输出：
  1) 估计误差曲线（均值±标准差）
  2) 估计方差曲线（随N变化）
  3) ESS/N 曲线（均值±标准差）
所有结果保存到 results/ 并显示图形
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.distributions import Normal
from src.importance_sampling import is_estimate, compute_ess
from src.utils import set_seed, setup_plot_style

# 实验参数
TRUE_VALUE = 1.0                     # E_p[X^2] 真值
N_SAMPLES_LIST = [100, 300, 1000, 3000, 10000]
N_REPEATS = 30
RANDOM_SEED = 42

# 定义目标分布和提议分布
target = Normal(mean=0.0, std=1.0)
proposals = {
    "Good (N(0,1.2^2))": Normal(0.0, 1.2),
    "Moderate (N(0,2^2))": Normal(0.0, 2.0),
    "Bad (N(3,1))": Normal(3.0, 1.0)
}

# 被积函数 f(x) = x^2
def f(x):
    return x ** 2

def run_experiment():
    set_seed(RANDOM_SEED)
    setup_plot_style()
    
    # 存储结果: dict[proposal_name][metric] = list over N_samples
    results = {name: {"estimates": [], "variances": [], "ess_ratio": []} 
               for name in proposals}
    
    for n in N_SAMPLES_LIST:
        print(f"Processing N = {n} ...")
        for name, prop in proposals.items():
            est_list = []
            ess_list = []
            for rep in range(N_REPEATS):
                # 每次重复使用不同种子（基于主种子+重复索引）
                set_seed(RANDOM_SEED + rep * 100 + n)  # 确保不同配置不同种子
                estimate, weights = is_estimate(target, prop, f, n, return_weights=True)
                est_list.append(estimate)
                ess = compute_ess(weights)
                ess_ratio = ess / n
                ess_list.append(ess_ratio)
            
            estimates = np.array(est_list)
            ess_ratios = np.array(ess_list)
            results[name]["estimates"].append(estimates)
            results[name]["variances"].append(np.var(estimates))
            results[name]["ess_ratio"].append(ess_ratios)
    
    # 绘图
    # 图1: 估计值曲线（均值 ± 标准差）
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for name in proposals:
        means = [np.mean(results[name]["estimates"][i]) for i in range(len(N_SAMPLES_LIST))]
        stds = [np.std(results[name]["estimates"][i]) for i in range(len(N_SAMPLES_LIST))]
        ax1.errorbar(N_SAMPLES_LIST, means, yerr=stds, marker='o', capsize=3, label=name)
    ax1.axhline(y=TRUE_VALUE, color='k', linestyle='--', label='True value = 1')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of samples N')
    ax1.set_ylabel('Estimate of E[X^2]')
    ax1.set_title('Importance Sampling Estimates (Mean ± Std)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig('../results/estimate_curve.png', dpi=150)
    
    # 图2: 估计方差曲线
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for name in proposals:
        vars = results[name]["variances"]
        ax2.plot(N_SAMPLES_LIST, vars, marker='s', label=name)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of samples N')
    ax2.set_ylabel('Variance of estimates')
    ax2.set_title('Estimation Variance vs Sample Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('../results/variance_curve.png', dpi=150)
    
    # 图3: ESS/N 曲线（均值 ± 标准差）
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for name in proposals:
        means_ess = [np.mean(results[name]["ess_ratio"][i]) for i in range(len(N_SAMPLES_LIST))]
        stds_ess = [np.std(results[name]["ess_ratio"][i]) for i in range(len(N_SAMPLES_LIST))]
        ax3.errorbar(N_SAMPLES_LIST, means_ess, yerr=stds_ess, marker='^', capsize=3, label=name)
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of samples N')
    ax3.set_ylabel('ESS / N')
    ax3.set_title('Effective Sample Size Ratio (ESS/N)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig('../results/ess_ratio_curve.png', dpi=150)
    
    plt.show()
    
    # 打印一些数值结果
    print("\n=== 数值结果总结 (N=10000) ===")
    for name in proposals:
        idx = N_SAMPLES_LIST.index(10000)
        est_mean = np.mean(results[name]["estimates"][idx])
        est_std = np.std(results[name]["estimates"][idx])
        ess_mean = np.mean(results[name]["ess_ratio"][idx])
        print(f"{name}: Estimate = {est_mean:.4f} ± {est_std:.4f}, ESS/N = {ess_mean:.3f}")
    
    # 可选：保存原始数据为npz
    os.makedirs('../results', exist_ok=True)
    np.savez('../results/exp_a_results.npz', 
             N_samples=N_SAMPLES_LIST, 
             proposals=list(proposals.keys()),
             estimates={name: [arr.tolist() for arr in results[name]["estimates"]] for name in proposals},
             ess_ratios={name: [arr.tolist() for arr in results[name]["ess_ratio"]] for name in proposals})

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs('../results', exist_ok=True)
    run_experiment()