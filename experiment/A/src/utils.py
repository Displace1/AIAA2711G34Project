"""通用工具：随机种子、绘图样式等"""
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """固定随机种子确保可复现"""
    np.random.seed(seed)

def setup_plot_style():
    """统一绘图风格"""
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150