# ============================================================
# Q3 优化版：首次导通分析四合一图
# 替换 问题3_最低填充量.ipynb 的 cell[19]（49c58a07）
# ============================================================

# ── 配色 ──
C3 = {
    "cdf":        "#2471a3",  # CDF 曲线
    "target":     "#cb4335",  # 目标线 90%
    "threshold":  "#1e8449",  # 经验阈值
    "bin":        "#f39c12",  # 舍入档高亮
    "hist":       "#5d6d7e",  # 直方图
    "converge":   "#7d3c98",  # 收敛曲线
    "final":      "#1e8449",  # 最终估计
}

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
phi_axis = phi_percent(counts)

# ========== 左上：导通概率 CDF ==========
ax = axes[0, 0]
ax.step(phi_axis, probability_curve, where="post", color=C3["cdf"], lw=2.0)
ax.axhline(TARGET_PROBABILITY, color=C3["target"], ls="--", lw=1.5,
           label=f"目标概率 {TARGET_PROBABILITY:.0%}")
ax.axvline(phi_hat, color=C3["threshold"], ls="--", lw=1.5,
           label=f"经验阈值 {phi_hat:.4f}%")
ax.set_xlim(phi_percent(450), phi_percent(N_MAX))
ax.set_ylim(0, 1.02)
ax.set_xlabel("名义体积分数 (%)")
ax.set_ylabel("经验导通概率")
ax.set_title("首次导通数量的经验累积分布函数 (CDF)", fontweight="bold")
ax.grid(True, alpha=0.25, ls="--")
ax.legend(loc="lower right", fontsize=8.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# ========== 右上：90% 阈值附近局部放大 ==========
ax = axes[0, 1]
zoom = (counts >= n_hat - 20) & (counts <= n_hat + 20)
ax.step(counts[zoom], probability_curve[zoom], where="post",
        color=C3["cdf"], lw=2.0)
ax.axhline(TARGET_PROBABILITY, color=C3["target"], ls="--", lw=1.5)
ax.axvspan(candidate_low_n, candidate_high_n,
           color=C3["bin"], alpha=0.22, lw=0,
           label=f"{candidate_report:.2f}% 舍入档 [{candidate_low_n}, {candidate_high_n}]")
ax.axvline(n_hat, color=C3["threshold"], ls="--", lw=1.5,
           label=f"$\\widehat{{N}}_{{90}}$ = {n_hat}")
ax.set_xlabel("完整原圆柱数量 N")
ax.set_ylabel("经验导通概率")
ax.set_title(f"目标概率 {TARGET_PROBABILITY:.0%} 附近局部放大", fontweight="bold")
ax.grid(True, alpha=0.25, ls="--")
ax.legend(loc="lower right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# ========== 左下：首次导通时间 T 的分布 ==========
ax = axes[1, 0]
finite_t = first_passage[first_passage <= N_MAX]
n_censored = np.count_nonzero(first_passage == NO_HIT_SENTINEL)
bins = np.arange(max(250, int(finite_t.min()) - 1), N_MAX + 2, 5)
ax.hist(finite_t, bins=bins, color=C3["hist"], alpha=0.82,
        edgecolor="white", linewidth=0.3)
ax.axvline(n_hat, color=C3["target"], ls="--", lw=1.8,
           label=f"$\\widehat{{N}}_{{90}}$ = {n_hat}")
ax.set_xlabel("首次导通圆柱数 T")
ax.set_ylabel("试验频数")
ax.set_title(
    f"T 的分布（T > {N_MAX} 的右删失数 = {n_censored}，"
    f"{100 * n_censored / N_TRIALS:.2f}%）",
    fontweight="bold",
)
ax.grid(axis="y", alpha=0.25, ls="--")
ax.legend(fontsize=8.5)

# ========== 右下：N90 估计的收敛过程 ==========
ax = axes[1, 1]
checkpoints = np.unique(
    np.r_[np.arange(500, N_TRIALS + 1, 500), N_TRIALS]
).astype(int)
running = []
for m in checkpoints:
    rank = int(math.ceil(TARGET_PROBABILITY * m))
    running.append(int(np.partition(first_passage[:m], rank - 1)[rank - 1]))
ax.plot(checkpoints, phi_percent(running), color=C3["converge"], lw=2.0)
ax.axhline(phi_hat, color=C3["final"], ls="--", lw=1.5,
           label=f"最终估计 {phi_hat:.4f}%")
ax.set_xlabel("累计试验次数")
ax.set_ylabel("经验 90% 分位数体积分数 (%)")
ax.set_title("最低填充量估计的收敛过程", fontweight="bold")
ax.grid(True, alpha=0.25, ls="--")
ax.legend(fontsize=8.5)

# 统一调整
plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
plt.show()
