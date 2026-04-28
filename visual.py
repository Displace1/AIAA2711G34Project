import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import norm, uniform
from matplotlib.patches import Rectangle

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------
# 目标分布（用于拒绝采样和重要性采样）
# 这里使用一个双峰分布（混合高斯）作为示例
def target_pdf(x):
    return 0.3 * norm.pdf(x, -2, 0.8) + 0.7 * norm.pdf(x, 2, 0.8)


# 提议分布（均匀分布，范围为[-4, 6]）
def proposal_pdf(x):
    return uniform.pdf(x, -4, 10)


def proposal_rvs(size):
    return np.random.uniform(-4, 6, size)


# 常数 M（确保 M * q(x) >= p(x)）
def find_M():
    x_vals = np.linspace(-4, 6, 1000)
    ratios = target_pdf(x_vals) / proposal_pdf(x_vals)
    return np.max(ratios) * 1.05  # 稍微宽松一点


M = find_M()


# ------------------------------
# 直接采样
def direct_sampling(n_samples=1000):
    samples = np.random.normal(0, 1, n_samples)
    return samples


# ------------------------------
# 拒绝采样
def rejection_sampling(n_samples, M, proposal_rvs, target_pdf, proposal_pdf):
    samples = []
    while len(samples) < n_samples:
        x_candidate = proposal_rvs(1)[0]
        u = np.random.uniform(0, M * proposal_pdf(x_candidate))
        if u <= target_pdf(x_candidate):
            samples.append(x_candidate)
    return np.array(samples)


# ------------------------------
# 重要性采样（估计期望 E[f(X)]，其中 f(x)=x^2）
def importance_sampling(n_samples, target_pdf, proposal_rvs, proposal_pdf):
    samples = proposal_rvs(n_samples)
    weights = target_pdf(samples) / proposal_pdf(samples)
    # 归一化权重（可选，用于无偏估计）
    weights_norm = weights / np.sum(weights)
    # 估计期望 E[f(X)] = sum(weights_norm * f(samples))
    f_samples = samples ** 2
    est_mean = np.sum(weights_norm * f_samples)
    return samples, weights, est_mean


# ------------------------------
# 可视化主函数
def visualize_sampling():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("采样方法可视化：直接采样 / 拒绝采样 / 重要性采样", fontsize=16)

    # 子图布局：2行2列
    ax1 = plt.subplot(2, 2, 1)  # 直接采样
    ax2 = plt.subplot(2, 2, 2)  # 拒绝采样
    ax3 = plt.subplot(2, 2, 3)  # 目标分布与提议分布
    ax4 = plt.subplot(2, 2, 4)  # 重要性采样权重

    plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.35)

    # 初始参数
    n_samples = 1000

    # ------------------ 直接采样 ------------------
    samples_direct = direct_sampling(n_samples)
    ax1.hist(samples_direct, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    x_vals = np.linspace(-4, 4, 200)
    ax1.plot(x_vals, norm.pdf(x_vals, 0, 1), 'r-', lw=2, label='标准正态分布')
    ax1.set_title("直接采样 (标准正态)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("密度")
    ax1.legend()

    # ------------------ 目标分布与提议分布曲线 ------------------
    x_range = np.linspace(-4, 6, 500)
    ax3.plot(x_range, target_pdf(x_range), 'b-', lw=2, label='目标分布 p(x)')
    ax3.plot(x_range, M * proposal_pdf(x_range), 'g--', lw=2, label=f'M * q(x) (M={M:.2f})')
    ax3.fill_between(x_range, 0, target_pdf(x_range), alpha=0.2, color='blue')
    ax3.set_title("拒绝采样：目标分布与提议分布")
    ax3.set_xlabel("x")
    ax3.set_ylabel("概率密度")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ------------------ 拒绝采样（初始） ------------------
    samples_rej = rejection_sampling(n_samples, M, proposal_rvs, target_pdf, proposal_pdf)
    ax2.hist(samples_rej, bins=40, density=True, alpha=0.6, color='orange', edgecolor='black', label='拒绝采样样本')
    ax2.plot(x_range, target_pdf(x_range), 'b-', lw=2, label='目标分布')
    ax2.set_title(f"拒绝采样 (样本数={n_samples})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("密度")
    ax2.legend()

    # ------------------ 重要性采样 ------------------
    samples_imp, weights, est_mean = importance_sampling(n_samples, target_pdf, proposal_rvs, proposal_pdf)
    ax4.scatter(samples_imp, weights, alpha=0.6, c=weights, cmap='viridis', edgecolor='k')
    ax4.axhline(y=np.mean(weights), color='r', linestyle='--', label=f'平均权重 = {np.mean(weights):.3f}')
    ax4.set_title(f"重要性采样权重分布  (估计 E[X²] = {est_mean:.4f})")
    ax4.set_xlabel("样本 x")
    ax4.set_ylabel("重要性权重")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ------------------ 添加滑块控制样本数 ------------------
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, '样本数量', 100, 5000, valinit=n_samples, valstep=100)

    # 更新函数
    def update(val):
        n = int(slider.val)
        # 更新直接采样
        new_direct = direct_sampling(n)
        ax1.clear()
        ax1.hist(new_direct, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        ax1.plot(x_vals, norm.pdf(x_vals, 0, 1), 'r-', lw=2)
        ax1.set_title("直接采样 (标准正态)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("密度")
        ax1.legend(['标准正态分布'])

        # 更新拒绝采样
        new_rej = rejection_sampling(n, M, proposal_rvs, target_pdf, proposal_pdf)
        ax2.clear()
        ax2.hist(new_rej, bins=40, density=True, alpha=0.6, color='orange', edgecolor='black')
        ax2.plot(x_range, target_pdf(x_range), 'b-', lw=2)
        ax2.set_title(f"拒绝采样 (样本数={n})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("密度")
        ax2.legend(['目标分布', '拒绝采样样本'])

        # 更新重要性采样
        new_imp, new_weights, new_est = importance_sampling(n, target_pdf, proposal_rvs, proposal_pdf)
        ax4.clear()
        ax4.scatter(new_imp, new_weights, alpha=0.6, c=new_weights, cmap='viridis', edgecolor='k')
        ax4.axhline(y=np.mean(new_weights), color='r', linestyle='--', label=f'平均权重 = {np.mean(new_weights):.3f}')
        ax4.set_title(f"重要性采样权重分布  (估计 E[X²] = {new_est:.4f})")
        ax4.set_xlabel("样本 x")
        ax4.set_ylabel("重要性权重")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # 重置按钮（可选）
    ax_reset = plt.axes([0.45, 0.02, 0.1, 0.04])
    btn_reset = Button(ax_reset, '重置')
    def reset(event):
        slider.reset()
    btn_reset.on_clicked(reset)

    plt.show()


if __name__ == "__main__":
    visualize_sampling()