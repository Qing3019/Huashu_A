# ============================================================
# A题 统一可视化配色与样式
# 使用方法：在每个 notebook 的可视化 cell 开头粘贴以下内容
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 统一配色方案 ──
C = {
    "blue":      "#2471a3",   # 主蓝色 — 曲线、柱状图
    "red":       "#cb4335",   # 强调红色 — 目标线、最优解标记
    "green":     "#1e8449",   # 认证/通过 — 稳健方案标记
    "orange":    "#e67e22",   # 橙色 — 区间高亮、右分量
    "purple":    "#7d3c98",   # 紫色 — 收敛曲线
    "grey":      "#95a5a6",   # 灰色 — 背景元素、非活跃片段
    "dark":      "#2c3e50",   # 深色 — 文字
    "electrode_l": "#2980b9", # 左电极面
    "electrode_r": "#e74c3c", # 右电极面
}

# ── 全局 matplotlib 设置 ──
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#bdc3c7",
})
print("配色与样式已加载。")
