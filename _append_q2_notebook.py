from pathlib import Path

import nbformat


notebook_path = Path("问题二_导通概率.ipynb")
engine_path = Path("_q2_engine_scratch.py")
notebook = nbformat.read(notebook_path, 4)
notebook.cells = [
    cell for cell in notebook.cells
    if not cell.get("metadata", {}).get("q2_mc_generated")
]
if len(notebook.cells) != 1:
    raise RuntimeError("为避免覆盖用户已有内容，本追加脚本要求Notebook当前恰好只有原始的1个单元。")

engine = engine_path.read_text(encoding="utf-8")
start_geometry = engine.index("class DynamicUnionFind:")
start_gjk = engine.index("@dataclass\nclass SupportPoint:")
start_simulation = engine.index("def sample_cylinders(")
end_simulation = engine.index("\n\nif __name__=='__main__':")

prelude = engine[:start_geometry].replace("from __future__ import annotations\n\n", "")
geometry = engine[start_geometry:start_gjk].rstrip()
gjk_code = engine[start_gjk:start_simulation].rstrip()
simulation = engine[start_simulation:end_simulation].rstrip()

cells = [
    nbformat.v4.new_markdown_cell(r"""## 第二问：蒙特卡洛随机模型

题面没有给出“任意位置、任意方向”的概率分布。这里作如下可复现假设：

1. 每根原始圆柱的轴中点在 $[-5000,5000)^3$ 内独立均匀分布；
2. 轴方向在单位球面 $S^2$ 上各向同性，用三维标准正态向量归一化生成；
3. 不同圆柱、不同试验相互独立；题面允许重叠，因此不做碰撞拒绝；
4. 第二问严格使用题面三轴均为 $\pm5000$ nm 的立方体边界；
5. 完整圆柱先按边界重新裁剪并搬运，每个非空碎片
   $$F_{i,k}=(C_i-k\odot(10000,10000,10000))\cap[-5000,5000]^3$$
   都作为独立导体；`parent_id` 只追踪来源，不自动连接；
6. 裁剪搬运后只计算盒内直接欧氏距离，不再使用周期镜像。

体积分数按原始完整圆柱的名义体积计算。一根圆柱即使产生多个碎片，体积仍只计一次。"""),
    nbformat.v4.new_code_cell(prelude.rstrip()),
    nbformat.v4.new_markdown_cell(r"""## 真实裁剪碎片与支持函数

以下代码不把斜切碎片封成普通小圆柱。先用轴段—盒相交快速取得可行点；只有轴线不进入盒而圆柱侧面可能进入时，才调用三变量 SLSQP 可行性问题。碎片 GJK 的支持点同样直接在“有限圆柱与盒的交集”上求解，不使用截断多面体、胶囊或对偶证书。"""),
    nbformat.v4.new_code_cell(geometry),
    nbformat.v4.new_markdown_cell(r"""## GJK、AABB 和独立碎片距离

碎片对先经过安全外包 AABB 和轴段距离筛选。最常见的侧面—侧面连接可直接构造盒内见证点；其余候选先计算未裁剪移位圆柱的解析 GJK，最后才调用真实斜切碎片的 SLSQP 支持 GJK。所有判断均为 `direct`，同源碎片也必须经过同样的几何判断。"""),
    nbformat.v4.new_code_cell(gjk_code),
    nbformat.v4.new_markdown_cell(r"""## 单次嵌套前缀试验

每次试验一次性生成最多707根独立同分布圆柱，并依次观察前354、424、495、707根。每个前缀本身仍具有正确的随机分布，而且逐试验必有

$$I_{354}\le I_{424}\le I_{495}\le I_{707}.$$

代码使用动态并查集；一旦较低档已经导通，继续添加圆柱不可能使其断开，因此可以立即把更高档记为导通并结束本次试验。普通蒙特卡洛样本不保存 BFS 图，以节省内存和时间。"""),
    nbformat.v4.new_code_cell(simulation),
    nbformat.v4.new_markdown_cell("## 轻量回归检查\n\n这些断言不参与概率估计，只检查随机方向、GJK基本距离、Wilson区间和‘同源碎片不自动连接’等关键逻辑。这里的回归不是统计回归。"),
    nbformat.v4.new_code_cell(r'''# 数量与传统四舍五入核验
traditional_round = np.floor(BOUNDARY_VOLUME * VOLUME_RATIO / CYLINDER_VOLUME + 0.5).astype(int)
assert np.array_equal(n_cylinders, np.array([354, 424, 495, 707]))
assert np.array_equal(n_cylinders, traditional_round)

# 各向同性方向的均值应接近0，二阶矩应接近1/3
test_rng = np.random.default_rng(20260808)
test_directions = test_rng.normal(size=(100_000, 3))
test_directions /= np.linalg.norm(test_directions, axis=1, keepdims=True)
assert np.max(np.abs(test_directions.mean(axis=0))) < 0.01
assert np.max(np.abs((test_directions**2).mean(axis=0) - 1/3)) < 0.01

# 普通有限圆柱的已知距离
test_a = Cylinder(np.array([-50., 0., 0.]), np.array([50., 0., 0.]))
test_b = Cylinder(np.array([-50., 100., 0.]), np.array([50., 100., 0.]))
test_c = Cylinder(np.array([60., 0., 0.]), np.array([160., 0., 0.]))
assert abs(gjk(test_a, test_b, threshold=None).upper - 40.0) < 1e-5
assert abs(gjk(test_a, test_c, threshold=None).upper - 10.0) < 1e-5

# 一根圆柱越过x右边界会生成左右两个同源碎片，但身份相同不会自动union
crossing = Cylinder(np.array([4000., 0., 0.]), np.array([6000., 0., 0.]), radius=30.)
crossing_fragments = generate_parent_fragments(0, crossing)
assert len(crossing_fragments) == 2
assert len({fragment.parent_id for fragment in crossing_fragments}) == 1
test_dsu = DynamicUnionFind()
left_test, right_test = test_dsu.add(), test_dsu.add()
fragment_nodes = [test_dsu.add() for _ in crossing_fragments]
test_dsu.union(left_test, fragment_nodes[1])
test_dsu.union(fragment_nodes[0], right_test)
assert not test_dsu.connected(left_test, right_test)

# Wilson区间的标准数值
lo0, hi0 = wilson_interval(0, 100)
lo50, hi50 = wilson_interval(50, 100)
assert abs(float(hi0) - 0.0369935) < 1e-5
assert np.allclose([lo50, hi50], [0.4038315, 0.5961685], atol=1e-5)

print("回归检查通过。")'''),
    nbformat.v4.new_markdown_cell(r"""## 蒙特卡洛配置与运行

默认 `QUICK_MODE=True` 只运行20次，用于验证流程、估算速度并给出粗略概率；它不能作为高精度论文结果。正式计算建议设置 `QUICK_MODE=False`，使用2500次试验，此时二项概率在最坏 $p=0.5$ 情形下的95%误差半宽约为2个百分点。

固定 `BASE_SEED` 后结果可复现。Windows/Jupyter 中默认串行最稳定；如果本机允许创建子进程，可把 `N_JOBS` 改成物理核心数附近的正整数。"""),
    nbformat.v4.new_code_cell('''BASE_SEED = 20260808
QUICK_MODE = True
QUICK_TRIALS = 20
PRODUCTION_TRIALS = 2500
N_JOBS = 1

N_TRIALS = QUICK_TRIALS if QUICK_MODE else PRODUCTION_TRIALS
if QUICK_MODE:
    print("当前为快速演示模式：概率区间会较宽；论文正式结果请运行2500次或更多。")

trial_indicators, trial_diagnostics, elapsed_seconds = run_monte_carlo(
    trial_count=N_TRIALS,
    n_levels=n_cylinders,
    base_seed=BASE_SEED,
    n_jobs=N_JOBS,
)
print(f"总耗时：{elapsed_seconds:.2f} s，平均每次：{elapsed_seconds/N_TRIALS:.3f} s")'''),
    nbformat.v4.new_markdown_cell(r"""## 概率估计与95% Wilson置信区间

对每个体积分数，若 $M$ 次试验中有 $K$ 次导通，则 $\hat p=K/M$。使用 Wilson 区间而不是普通 Wald 区间，因而在 $K=0$ 或 $K=M$ 时仍能给出合理区间。"""),
    nbformat.v4.new_code_cell('''successes = trial_indicators.sum(axis=0)
probability = successes / N_TRIALS
ci_low, ci_high = wilson_interval(successes, N_TRIALS)
mc_standard_error = np.sqrt(probability * (1 - probability) / N_TRIALS)
actual_ratio_percent = 100 * n_cylinders * CYLINDER_VOLUME / BOUNDARY_VOLUME

probability_table = pd.DataFrame({
    "目标体积分数/%": 100 * VOLUME_RATIO,
    "圆柱数": n_cylinders,
    "舍入后实际体积分数/%": actual_ratio_percent,
    "试验次数": N_TRIALS,
    "导通次数": successes,
    "导通概率估计": probability,
    "MC标准误差": mc_standard_error,
    "Wilson 95%下限": ci_low,
    "Wilson 95%上限": ci_high,
    "区间半宽": 0.5 * (ci_high - ci_low),
})
display(probability_table)

diagnostic_table = trial_diagnostics.agg(["mean", "max"]).T.rename(
    columns={"mean": "每次试验平均", "max": "单次最大"}
)
display(diagnostic_table)

planning_trials = np.array([1000, 2500, 5000])
planning_half_width = 1.959963984540054 * np.sqrt(0.25 / planning_trials)
display(pd.DataFrame({
    "正式试验次数": planning_trials,
    "95%最坏近似半宽": planning_half_width,
    "按当前串行速度估计耗时/h": elapsed_seconds / N_TRIALS * planning_trials / 3600,
}))'''),
    nbformat.v4.new_markdown_cell("## 收敛过程与概率图"),
    nbformat.v4.new_code_cell(r'''import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

running_successes = np.cumsum(trial_indicators, axis=0)
running_probability = running_successes / np.arange(1, N_TRIALS + 1)[:, None]
running_low = np.empty_like(running_probability, dtype=float)
running_high = np.empty_like(running_probability, dtype=float)
for row in range(N_TRIALS):
    running_low[row], running_high[row] = wilson_interval(running_successes[row], row + 1)

labels = [f"{100*x:.2f}%" for x in VOLUME_RATIO]
colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(labels)))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(1, N_TRIALS + 1)
for column, (label, color) in enumerate(zip(labels, colors)):
    axes[0].plot(x, running_probability[:, column], color=color, label=label)
    axes[0].fill_between(x, running_low[:, column], running_high[:, column], color=color, alpha=0.10)
axes[0].set_xlabel("累计试验次数")
axes[0].set_ylabel("累计导通概率")
axes[0].set_ylim(0, 1.02)
axes[0].set_title("蒙特卡洛累计估计与Wilson区间")
axes[0].grid(alpha=0.25)
axes[0].legend(title="目标体积分数")

positions = np.arange(len(labels))
errors = np.vstack((probability - ci_low, ci_high - probability))
axes[1].bar(positions, probability, color=colors, alpha=0.85)
axes[1].errorbar(positions, probability, yerr=errors, fmt="none", ecolor="black", capsize=5)
axes[1].set_xticks(positions, labels)
axes[1].set_ylim(0, 1.02)
axes[1].set_xlabel("目标体积分数")
axes[1].set_ylabel("导通概率")
axes[1].set_title(f"导通概率估计（M={N_TRIALS}）")
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()'''),
    nbformat.v4.new_markdown_cell(r"""## 使用说明

- 快速模式的输出只用于确认代码可运行和观察大致趋势；正式论文应把 `QUICK_MODE=False` 后重新运行，并报告固定试验次数、随机种子和 Wilson 区间。
- 四档使用共同随机数，单次试验中的四个指示变量相关，但每一列仍是正确的二项蒙特卡洛样本；共同前缀还能降低档位比较的随机波动。
- 若启用并行后系统不允许创建子进程，把 `N_JOBS` 恢复为1即可；随机流由 `(BASE_SEED, trial_index)` 唯一确定，所以串行与并行结果应一致。
- 当前模型贯彻“截断后碎片独立”的解释：同源碎片不自动导通，且最终不再进行周期镜像距离。"""),
]

for cell in cells:
    cell.metadata["q2_mc_generated"] = True
notebook.cells.extend(cells)
nbformat.write(notebook, notebook_path)
print(f"已在原始单元后追加 {len(cells)} 个单元；总单元数 {len(notebook.cells)}。")
