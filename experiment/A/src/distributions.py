"""概率分布接口及常用分布实现（高斯分布）"""
import numpy as np
from scipy.stats import norm

class Distribution:
    """分布基类，提供采样和对数密度接口"""
    def sample(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

class Normal(Distribution):
    """单变量高斯分布"""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        # 预创建scipy norm对象，便于调用logpdf
        self._dist = norm(loc=mean, scale=std)
    
    def sample(self, n_samples: int) -> np.ndarray:
        return self._dist.rvs(size=n_samples)
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.logpdf(x)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)