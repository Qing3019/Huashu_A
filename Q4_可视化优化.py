# ============================================================
# Q4 优化版：成本前沿图
# 替换 Q4/问题4_混合介质最低成本_严格GJK.ipynb 的 cell[13]（f56d2755）
# ============================================================

# ── 配色 ──
C4 = {
    "frontier_90": "#2471a3",  # 90% 前沿线
    "frontier_91": "#e67e22",  # 91% 稳健前沿线
    "point_opt":   "#cb4335",  # 点估计最优
    "certified":   "#1e8449",  # 认证稳健方案
}

fig, ax = plt.subplots(figsize=(9, 5.5))

# ── 两条前沿线 ──
for frame, style, color, label in [
    (frontier_90, "-",  C4["frontier_90"], "90% 点估计前沿"),
    (frontier_91, "--", C4["frontier_91"], "91% 稳健筛选前沿"),
]:
    plot_data = frame.sort_values("N_A")
    ax.plot(
        plot_data["N_A"], plot_data["成本_元"],
        style, color=color, lw=2.0, marker="o", ms=3.5,
        markeredgewidth=0.5, markeredgecolor="white",
        label=label,
    )

# ── 两个关键解 ──
ax.scatter(
    [point_best["N_A"]], [point_best["成本_元"]],
    color=C4["point_opt"], s=120, zorder=10, edgecolors="white",
    linewidths=1.2, label="点估计最低成本 (608, 5)",
)
ax.scatter(
    [frozen_candidate["N_A"]], [frozen_candidate["成本_元"]],
    color=C4["certified"], s=120, zorder=10, edgecolors="white",
    linewidths=1.2, marker="s",
    label="独立验证通过 (611, 25)",
)

# ── 标注文字 ──
ax.annotate(
    f"({int(point_best['N_A'])}, {int(point_best['N_B'])})\n"
    f"{float(point_best['成本_元']):.4f} 元",
    xy=(int(point_best["N_A"]), float(point_best["成本_元"])),
    xytext=(int(point_best["N_A"]) - 3, float(point_best["成本_元"]) + 0.008),
    fontsize=8.5, color=C4["point_opt"], fontweight="bold",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=C4["point_opt"], alpha=0.9),
)
ax.annotate(
    f"({int(frozen_candidate['N_A'])}, {int(frozen_candidate['N_B'])})\n"
    f"{float(frozen_candidate['成本_元']):.4f} 元",
    xy=(int(frozen_candidate["N_A"]), float(frozen_candidate["成本_元"])),
    xytext=(int(frozen_candidate["N_A"]) + 2.5, float(frozen_candidate["成本_元"]) + 0.005),
    fontsize=8.5, color=C4["certified"], fontweight="bold",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=C4["certified"], alpha=0.9),
)

# ── 坐标轴 ──
ax.set_xlabel("介质 A 数量 $N_A$", fontsize=11)
ax.set_ylabel("满足 90% 导通的最低成本（元）", fontsize=11)
ax.set_title("A/B 混合介质最低成本前沿线", fontweight="bold", fontsize=13)
ax.grid(True, alpha=0.25, ls="--")

# X 轴整数刻度
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax.set_xlim(
    int(point_best["N_A"]) - 10,
    frozen_candidate["N_A"] + 2,
)

# ── 图例 ──
ax.legend(loc="upper left", fontsize=8.5, ncol=1)

plt.tight_layout()
plt.show()
