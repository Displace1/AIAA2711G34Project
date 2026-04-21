"""诊断指标：ESS、权重熵等"""
import numpy as np

def effective_sample_size(weights: np.ndarray) -> float:
    """计算ESS，输入原始权重"""
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    if sum_w2 == 0:
        return 0.0
    return (sum_w ** 2) / sum_w2

def weight_entropy(weights: np.ndarray) -> float:
    """归一化权重的熵 H = -sum (w_norm * log(w_norm))，用于衡量权重分散程度"""
    w_norm = weights / np.sum(weights)
    # 避免log(0)
    w_norm = np.clip(w_norm, 1e-12, 1.0)
    return -np.sum(w_norm * np.log(w_norm))

def max_weight_ratio(weights: np.ndarray) -> float:
    """最大权重占比 = max(w) / sum(w)"""
    return np.max(weights) / np.sum(weights)