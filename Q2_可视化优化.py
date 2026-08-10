# ============================================================
# Q2 优化版：导通概率收敛图 + 柱状估计图
# 替换 问题2_导通概率.ipynb 的 cell[16]（c21788cd）
# ============================================================

# ── 配色 ──
C2 = ["#2471a3", "#1e8449", "#e67e22", "#cb4335"]  # 四档体积分数
BAND_ALPHA = 0.12

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# ========== 左图：累计导通概率收敛 ==========
ax = axes[0]
x = np.arange(1, N_TRIALS + 1)
labels = ["0.50%", "0.60%", "0.70%", "1.00%"]

for col, (label, color) in enumerate(zip(labels, C2)):
    ax.plot(x, running_probability[:, col], color=color, lw=1.6, label=label)
    ax.fill_between(
        x,
        running_low[:, col],
        running_high[:, col],
        color=color, alpha=BAND_ALPHA, linewidth=0,
    )

ax.set_xlabel("累计试验次数")
ax.set_ylabel("累计导通概率")
ax.set_ylim(0, 1.03)
ax.set_xlim(-20, N_TRIALS + 20)
ax.set_title("蒙特卡洛累计估计与逐点 95% Wilson 区间", fontweight="bold")
ax.grid(True, alpha=0.25, ls="--")
ax.legend(title="目标体积分数", loc="lower right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# ========== 右图：最终概率柱状图 ==========
ax = axes[1]
positions = np.arange(len(labels))
errors = np.vstack((probability - ci_low, ci_high - probability))

bars = ax.bar(positions, probability, color=C2, alpha=0.88, width=0.55,
              edgecolor="white", linewidth=0.8)
ax.errorbar(positions, probability, yerr=errors, fmt="none",
            ecolor="#2c3e50", capsize=6, lw=1.5, capthick=1.5)

# 在柱上方标注概率值
for pos, p, lo, hi in zip(positions, probability, ci_low, ci_high):
    ax.text(pos, hi + 0.02, f"{p:.3f}", ha="center", fontsize=9,
            fontweight="bold", color="#2c3e50")

ax.set_xticks(positions)
ax.set_xticklabels([f"{l}\n({n}根)" for l, n in zip(labels, n_cylinders)],
                   fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("导通概率")
ax.set_title(f"导通概率估计（M = {N_TRIALS}）", fontweight="bold")
ax.grid(axis="y", alpha=0.25, ls="--")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

# 打印数值
for ratio, count, p, lo, hi in zip(
    100 * VOLUME_RATIO, n_cylinders, probability, ci_low, ci_high
):
    print(f"体积分数 {ratio:.2f}%（{count} 根）："
          f"P̂ = {p:.4f}，95% Wilson CI = [{lo:.4f}, {hi:.4f}]")
