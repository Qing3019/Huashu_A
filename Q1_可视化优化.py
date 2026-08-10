# ============================================================
# Q1 优化版：三维导电路径图
# 替换 问题1_并查集_GJK_BFS.ipynb 的 cell[25]（plot_sheet 函数及调用）
# ============================================================

# ── 配色 ──
C1 = {
    "path":       "#cb4335",  # 导通路径 — 红色
    "left_comp":  "#2980b9",  # 左电极连通分量 — 蓝色
    "right_comp": "#e67e22",  # 右电极连通分量 — 橙色
    "other":      "#bdc3c7",  # 非活跃片段 — 浅灰
    "face_left":  "#2980b9",  # 左带电面
    "face_right": "#e74c3c",  # 右带电面
    "text":       "#2c3e50",
}

def plot_sheet_optimized(sheet_name: str, result: dict):
    """绘制单个微构体的三维导通结构图（优化版）"""
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 调整 3D 视角
    ax.view_init(elev=22, azim=-40)

    path_indices = {pid - 1 for pid in (result["path"] or [])}
    left_comp = reachable_nodes(result["adjacency"], result["left"])
    right_comp = reachable_nodes(result["adjacency"], result["right"])

    # ── 逐片段绘制 ──
    for idx, seg in enumerate(result["segments"]):
        if idx in path_indices:
            color, lw, alpha, zorder = C1["path"], 3.5, 1.0, 10
        elif result["path"] is None and idx in left_comp:
            color, lw, alpha, zorder = C1["left_comp"], 2.0, 0.85, 5
        elif result["path"] is None and idx in right_comp:
            color, lw, alpha, zorder = C1["right_comp"], 2.0, 0.85, 5
        else:
            color, lw, alpha, zorder = C1["other"], 0.6, 0.22, 1

        ax.plot(
            [seg.p0[0], seg.p1[0]],
            [seg.p0[1], seg.p1[1]],
            [seg.p0[2], seg.p1[2]],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round",
        )

    # ── 路径节点标注 ──
    for pid in (result["path"] or []):
        idx = pid - 1
        seg = result["segments"][idx]
        mid = 0.5 * (seg.p0 + seg.p1)
        ax.text(
            mid[0], mid[1], mid[2],
            f"A{pid}", fontsize=8, fontweight="bold",
            color=C1["path"], ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor=C1["path"], alpha=0.85),
        )

    # ── 带电面 ──
    half = result["half_extents"]
    yy, zz = np.meshgrid(
        np.linspace(-half[1], half[1], 5),
        np.linspace(-half[2], half[2], 5),
    )
    for x_val, face_color, label in [
        (-half[0], C1["face_left"],  "左带电面 (X=−5000)"),
        ( half[0], C1["face_right"], "右带电面 (X=+5000)"),
    ]:
        ax.plot_surface(
            np.full_like(yy, x_val), yy, zz,
            color=face_color, alpha=0.10, shade=False,
            edgecolor=face_color, linewidth=0.3, zorder=0,
        )

    # ── 坐标轴 ──
    pad = max(30, 0.03 * np.max(half))
    ax.set_xlim(-half[0] - pad, half[0] + pad)
    ax.set_ylim(-half[1] - pad, half[1] + pad)
    ax.set_zlim(-half[2] - pad, half[2] + pad)
    ax.set_xlabel("X (nm)", labelpad=8)
    ax.set_ylabel("Y (nm)", labelpad=8)
    ax.set_zlabel("Z (nm)", labelpad=8)

    # ── 标题 ──
    status = "导通" if result["conductive"] else "不导通"
    ax.set_title(
        f"{sheet_name}：{status}",
        fontsize=13, fontweight="bold", pad=18,
    )

    # ── 手工图例（3D 中 ax.legend 经常错位） ──
    from matplotlib.lines import Line2D
    legend_items = []
    if result["path"] is not None:
        legend_items.append(
            Line2D([0],[0], color=C1["path"], lw=3.5,
                   label=f"导通路径 ({len(result['path'])} 个片段)"))
    if result["path"] is None:
        legend_items.append(
            Line2D([0],[0], color=C1["left_comp"], lw=2.5,
                   label="左电极连通分量"))
        legend_items.append(
            Line2D([0],[0], color=C1["right_comp"], lw=2.5,
                   label="右电极连通分量"))
    legend_items.append(
        Line2D([0],[0], color=C1["other"], lw=0.8,
               label=f"其余片段 ({len(result['segments'])} 个总计)"))
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=8, ncol=1)

    plt.tight_layout()
    return fig, ax


# 逐个绘制三组
for sheet_name, result in results.items():
    fig, ax = plot_sheet_optimized(sheet_name, result)
    plt.show()
