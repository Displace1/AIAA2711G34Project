"""重要性采样估计器（标准IS）"""
import numpy as np
from typing import Callable, Tuple
from distributions import Distribution

def is_estimate(
    target: Distribution,
    proposal: Distribution,
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    return_weights: bool = False
) -> Tuple[float, np.ndarray]:
    """
    标准重要性采样估计 E_p[f(X)]
    
    参数:
        target: 目标分布 p(x)
        proposal: 提议分布 q(x)
        f: 被积函数，输入形状 (n_samples,) 输出形状 (n_samples,)
        n_samples: 采样数量
        return_weights: 是否返回原始权重
    
    返回:
        estimate: 估计值
        weights: (可选) 原始权重 w_i = p(x_i)/q(x_i)
    """
    # 从提议分布采样
    x = proposal.sample(n_samples)
    
    # 计算对数权重 log w = log p - log q
    log_p = target.log_pdf(x)
    log_q = proposal.log_pdf(x)
    log_w = log_p - log_q
    
    # 数值稳定：减去最大值后指数化
    max_log_w = np.max(log_w)
    # 避免指数溢出，如果max_log_w过大（>700）则给出警告，但继续计算（可能得到inf）
    if max_log_w > 700:
        print(f"Warning: max log weight = {max_log_w:.2f} > 700, may cause overflow.")
    # 稳定权重 = exp(log_w - max_log_w)
    stable_w = np.exp(log_w - max_log_w)
    # 原始权重 = stable_w * exp(max_log_w)
    # 估计量 = (1/N) * sum f(x_i) * w_i
    f_val = f(x)
    # 为了避免重复乘以exp(max_log_w)造成溢出，采用如下等价形式：
    # sum f_i * w_i = exp(max_log_w) * sum f_i * stable_w
    # 但若exp(max_log_w)溢出，则无法表示。此时改用log-sum-exp计算期望的对数
    # 为简化，这里假设权重不会极端到使exp(max_log_w)溢出（实验A中max_log_w约<10）
    raw_weights = stable_w * np.exp(max_log_w)
    estimate = np.mean(f_val * raw_weights)
    
    if return_weights:
        return estimate, raw_weights
    return estimate

def compute_ess(weights: np.ndarray) -> float:
    """
    计算有效样本量 ESS = (sum w)^2 / sum w^2
    输入原始权重（未归一化）
    """
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    if sum_w2 == 0:
        return 0.0
    return (sum_w ** 2) / sum_w2