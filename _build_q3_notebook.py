from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import nbformat as nbf
import numpy as np
from scipy.stats import beta, binom


WORKSPACE = Path(".")
OUTPUT = WORKSPACE / "问题3_最低填充量.ipynb"
CACHE_PATH = WORKSPACE / "q3_first_passage_results.npz"
FORMAL_TRIALS = 22500


def find_q2_notebook():
    candidates = []
    for path in WORKSPACE.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if len(notebook.get("cells", [])) == 18:
            candidates.append((path, notebook))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Q2 notebook, found {len(candidates)}")
    return candidates[0]


q2_path, q2 = find_q2_notebook()
q2_parts = ["".join(q2["cells"][i]["source"]).rstrip() for i in (2, 4, 6, 8)]
addon = Path("_q3_first_passage_addon.py").read_text(encoding="utf-8").rstrip()
model_text = "\n\n".join((*q2_parts, addon)) + "\n"
engine_path = Path("_q3_engine_full.py")
if not engine_path.exists():
    raise FileNotFoundError("请先运行 _build_q3_engine.py 生成正式引擎")
disk_engine_text = engine_path.read_text(encoding="utf-8")
if disk_engine_text.replace("\r\n", "\n") != model_text.replace("\r\n", "\n"):
    raise RuntimeError("Notebook内嵌模型与正式生产引擎不一致")
# 缓存签名使用正式生产引擎文件的实际字节，和后台生成器完全一致。
model_sha256 = hashlib.sha256(engine_path.read_bytes()).hexdigest()

volume_code = r'''import math
import numpy as np

BOUNDARY_VOLUME = 10000.0**3
CYLINDER_VOLUME = math.pi * 30.0**2 * 5000.0
VOLUME_STEP_PERCENT = 100.0 * CYLINDER_VOLUME / BOUNDARY_VOLUME
TARGET_PROBABILITY = 0.90
N_MAX = 707
NO_HIT_SENTINEL = N_MAX + 1

print(f"单根介质A体积：{CYLINDER_VOLUME:.6f} nm³")
print(f"每增加一根圆柱，名义体积分数增加：{VOLUME_STEP_PERCENT:.10f}%")
print(f"第三问初始搜索上界：N_max={N_MAX}（哨兵{NO_HIT_SENTINEL}表示超过上界仍未导通）")'''

regression_code = r'''# 体积换算与传统四舍五入
assert abs(VOLUME_STEP_PERCENT - 0.001413716694115407) < 1e-15
q2_levels = np.array([354, 424, 495, 707], dtype=int)
assert np.array_equal(
    np.floor(BOUNDARY_VOLUME*np.array([0.005, 0.006, 0.007, 0.01])/CYLINDER_VOLUME + 0.5).astype(int),
    q2_levels,
)

# 各向同性和位置均匀性的矩检验
test_rng = np.random.default_rng(20260808)
test_centers = test_rng.uniform(-BOX_HALF, BOX_HALF, size=(100_000, 3))
test_directions = test_rng.normal(size=(100_000, 3))
test_directions /= np.linalg.norm(test_directions, axis=1, keepdims=True)
assert np.all((test_centers >= BOX_MIN) & (test_centers < BOX_MAX))
assert np.max(np.abs(test_centers.mean(axis=0))) < 30.0
assert np.max(np.abs(test_directions.mean(axis=0))) < 0.01
assert np.max(np.abs((test_directions**2).mean(axis=0) - 1/3)) < 0.01

# 已知有限圆柱表面距离和1.8 nm阈值
test_a = Cylinder(np.array([-50., 0., 0.]), np.array([50., 0., 0.]))
test_b = Cylinder(np.array([-50., 100., 0.]), np.array([50., 100., 0.]))
test_c = Cylinder(np.array([60., 0., 0.]), np.array([160., 0., 0.]))
assert abs(gjk(test_a, test_b, threshold=None).upper - 40.0) < 1e-5
assert abs(gjk(test_a, test_c, threshold=None).upper - 10.0) < 1e-5
at_threshold = Cylinder(
    np.array([-50., 2*RADIUS+THRESHOLD, 0.]),
    np.array([50., 2*RADIUS+THRESHOLD, 0.]),
)
above_threshold = Cylinder(
    np.array([-50., 2*RADIUS+THRESHOLD+5e-4, 0.]),
    np.array([50., 2*RADIUS+THRESHOLD+5e-4, 0.]),
)
assert gjk(test_a, at_threshold, threshold=None).upper <= CONTACT_LIMIT
assert gjk(test_a, above_threshold, threshold=None).upper > CONTACT_LIMIT

# 跨界同源碎片不能仅凭身份自动连接
crossing = Cylinder(np.array([4000., 0., 0.]), np.array([6000., 0., 0.]), radius=30.)
crossing_fragments = generate_parent_fragments(0, crossing)
assert len(crossing_fragments) == 2
test_dsu = DynamicUnionFind()
left_test, right_test = test_dsu.add(), test_dsu.add()
fragment_nodes = [test_dsu.add() for _ in crossing_fragments]
test_dsu.union(left_test, fragment_nodes[0])
test_dsu.union(fragment_nodes[1], right_test)
assert not test_dsu.connected(left_test, right_test)

# 支持点必须属于真实圆柱—盒交集
for fragment in crossing_fragments:
    for direction in np.vstack((np.eye(3), -np.eye(3))):
        point = fragment.support(direction)
        y = np.linalg.solve(fragment.matrix, point-fragment.cylinder.center)
        assert y_violation(fragment.cylinder, fragment.matrix, y, BOX_MIN, BOX_MAX) <= FEAS_TOL

# 首次导通包装器必须精确重构所有前缀状态
test_t, test_stats = simulate_first_passage(0, N_MAX, 20260808)
assert test_t == 314
assert np.array_equal(test_t <= q2_levels, np.array([True, True, True, True]))
assert test_stats["near_threshold"] == 0

# Wilson区间标准值
lo50, hi50 = wilson_interval(50, 100)
assert np.allclose([lo50, hi50], [0.4038315, 0.5961685], atol=1e-5)
print("几何、随机分布、边界碎片和首次导通回归检查通过。")'''

cache_code = r'''from pathlib import Path
import hashlib
import os
import platform
import scipy

BASE_SEED = 20260808
PRODUCTION_TRIALS = __FORMAL_TRIALS__
MIN_FORMAL_TRIALS = __FORMAL_TRIALS__
QUICK_TRIALS = 20
N_JOBS = 10
BATCH_SIZE = 100
RUN_MODE = "cache"       # "cache"、"quick"、"production" 或 "resume"
CACHE_PATH = Path("q3_first_passage_results.npz")
Q2_CACHE_PATH = Path("q2_mc_results_2500.npz")
EXPECTED_MODEL_CODE_SHA256 = "__MODEL_SHA256__"
EXPECTED_Q2_MODEL_SHA256 = "a7cdb15115b2f9757e10d45cc3692d674c3751f82fb39be5fdc91afe653df00e"
EXPECTED_Q2_RESULT_SHA256 = "902fe5bc0a119e1dbeac40f756e5aba149dc0f1ee55de897b50d8d8f434bf7d3"
assert RUN_MODE in {"cache", "quick", "production", "resume"}


def generator_environment_now():
    return {
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "RNG": "PCG64",
    }


def save_first_passage_cache(path, first_passage, diagnostics, elapsed_seconds, requested_trials):
    frame = pd.DataFrame(diagnostics)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(
        temporary,
        first_passage=np.asarray(first_passage, dtype=np.int16),
        trial_indices=np.arange(len(first_passage), dtype=np.int32),
        diagnostics=frame.to_numpy(dtype=float),
        diagnostic_columns=np.asarray(frame.columns, dtype=str),
        completed_trials=int(len(first_passage)),
        requested_trials=int(requested_trials),
        elapsed_seconds=float(elapsed_seconds),
        base_seed=BASE_SEED,
        n_max=N_MAX,
        no_hit_sentinel=NO_HIT_SENTINEL,
        target_probability=TARGET_PROBABILITY,
        model_code_sha256=EXPECTED_MODEL_CODE_SHA256,
        q2_model_code_sha256=EXPECTED_Q2_MODEL_SHA256,
        q2_indicator_sha256=EXPECTED_Q2_RESULT_SHA256,
        generator_python=platform.python_version(),
        generator_numpy=np.__version__,
        generator_scipy=scipy.__version__,
        bit_generator="PCG64",
        random_stream_schema="SeedSequence([base_seed, trial_index])",
        box_half=BOX_HALF,
        cylinder_length=LENGTH,
        cylinder_radius=RADIUS,
        threshold=THRESHOLD,
        numeric_tolerance=NUM_TOL,
        feasibility_tolerance=FEAS_TOL,
        support_feasibility_tolerance=SUPPORT_FEAS_TOL,
        support_stability_nm=SUPPORT_STABILITY_NM,
        near_threshold_tolerance=NEAR_THRESHOLD_TOL,
        gjk_gap_tolerance=GJK_GAP_TOL,
        gjk_far_guard=GJK_FAR_GUARD,
        contact_limit=CONTACT_LIMIT,
        result_sha256=hashlib.sha256(
            np.asarray(first_passage, dtype=np.int16).tobytes()
        ).hexdigest(),
    )
    os.replace(temporary, path)


def load_first_passage_cache(path, require_same_environment=False, require_complete=False):
    data = np.load(path, allow_pickle=False)
    assert int(data["base_seed"]) == BASE_SEED
    assert int(data["n_max"]) == N_MAX
    assert int(data["no_hit_sentinel"]) == NO_HIT_SENTINEL
    assert float(data["target_probability"]) == TARGET_PROBABILITY
    assert str(data["model_code_sha256"]) == EXPECTED_MODEL_CODE_SHA256
    assert str(data["q2_model_code_sha256"]) == EXPECTED_Q2_MODEL_SHA256
    assert str(data["q2_indicator_sha256"]) == EXPECTED_Q2_RESULT_SHA256
    assert str(data["bit_generator"]) == "PCG64"
    assert str(data["random_stream_schema"]) == "SeedSequence([base_seed, trial_index])"
    assert float(data["threshold"]) == THRESHOLD
    assert float(data["numeric_tolerance"]) == NUM_TOL
    assert float(data["feasibility_tolerance"]) == FEAS_TOL
    assert float(data["support_feasibility_tolerance"]) == SUPPORT_FEAS_TOL
    assert float(data["support_stability_nm"]) == SUPPORT_STABILITY_NM
    assert float(data["near_threshold_tolerance"]) == NEAR_THRESHOLD_TOL
    assert float(data["gjk_gap_tolerance"]) == GJK_GAP_TOL
    assert float(data["gjk_far_guard"]) == GJK_FAR_GUARD
    assert float(data["contact_limit"]) == CONTACT_LIMIT
    first_passage = data["first_passage"].astype(np.int16)
    assert int(data["completed_trials"]) == len(first_passage)
    if require_complete:
        assert len(first_passage) == PRODUCTION_TRIALS
        assert int(data["requested_trials"]) == PRODUCTION_TRIALS
        assert str(data["result_sha256"]) == hashlib.sha256(first_passage.tobytes()).hexdigest()
    assert np.array_equal(data["trial_indices"], np.arange(len(first_passage)))
    columns = [str(value) for value in data["diagnostic_columns"]]
    expected_columns = [
        "activated_cylinders", "fragments", "aabb_candidates", "axis_candidates",
        "side_witness", "full_gjk", "clipped_gjk", "support_calls", "fallbacks",
        "dykstra_retries", "projection_recoveries", "stability_accepts", "near_threshold",
    ]
    assert columns == expected_columns
    diagnostics = pd.DataFrame(data["diagnostics"], columns=columns)
    assert len(diagnostics) == len(first_passage)
    environment = {
        "Python": str(data["generator_python"]),
        "NumPy": str(data["generator_numpy"]),
        "SciPy": str(data["generator_scipy"]),
        "RNG": str(data["bit_generator"]),
    }
    if require_same_environment:
        assert environment == generator_environment_now(), "resume不允许跨数值环境混合续算"
    return first_passage, diagnostics, float(data["elapsed_seconds"]), environment


if RUN_MODE == "cache":
    if not CACHE_PATH.exists():
        raise FileNotFoundError("第三问正式缓存不存在；可改用quick或production模式。")
    first_passage, trial_diagnostics, elapsed_seconds, generator_environment = load_first_passage_cache(
        CACHE_PATH, require_complete=True
    )
elif RUN_MODE == "quick":
    first_passage, trial_diagnostics, elapsed_seconds = run_first_passage_monte_carlo(
        QUICK_TRIALS, N_MAX, BASE_SEED, n_jobs=1
    )
    generator_environment = generator_environment_now()
else:
    if RUN_MODE == "resume" and CACHE_PATH.exists():
        old_t, old_diagnostics, elapsed_seconds, generator_environment = load_first_passage_cache(
            CACHE_PATH, require_same_environment=True
        )
        t_batches = [old_t]
        diagnostic_batches = [old_diagnostics]
        completed = len(old_t)
    else:
        t_batches, diagnostic_batches = [], []
        elapsed_seconds = 0.0
        completed = 0
        generator_environment = generator_environment_now()
    for start in range(completed, PRODUCTION_TRIALS, BATCH_SIZE):
        batch_trials = min(BATCH_SIZE, PRODUCTION_TRIALS-start)
        batch_t, batch_diagnostics, batch_elapsed = run_first_passage_monte_carlo(
            batch_trials, N_MAX, BASE_SEED, N_JOBS, trial_start=start
        )
        t_batches.append(batch_t)
        diagnostic_batches.append(batch_diagnostics)
        elapsed_seconds += batch_elapsed
        first_passage = np.concatenate(t_batches)
        trial_diagnostics = pd.concat(diagnostic_batches, ignore_index=True)
        save_first_passage_cache(
            CACHE_PATH, first_passage, trial_diagnostics, elapsed_seconds, PRODUCTION_TRIALS
        )
        print(f"完成 {len(first_passage)}/{PRODUCTION_TRIALS} 次正式试验")
    first_passage = np.concatenate(t_batches)
    trial_diagnostics = pd.concat(diagnostic_batches, ignore_index=True)

N_TRIALS = len(first_passage)
assert np.all((first_passage >= 1) & (first_passage <= NO_HIT_SENTINEL))
assert np.isfinite(trial_diagnostics.to_numpy(dtype=float)).all()
assert (trial_diagnostics.to_numpy(dtype=float) >= 0).all()
assert int(trial_diagnostics["near_threshold"].sum()) == 0

# 和第二问2500个随机构型进行逐样本及稳定几何计数回归。
# SLSQP的投影/重试次数可能受进程级数值路径影响，只作有限非负诊断，
# 不要求跨批次逐项相等；导通布尔值必须逐比特相等。
if N_TRIALS >= 2500 and Q2_CACHE_PATH.exists():
    q2_cache = np.load(Q2_CACHE_PATH, allow_pickle=False)
    q2_levels = q2_cache["levels"].astype(int)
    reconstructed_q2 = first_passage[:2500, None] <= q2_levels[None, :]
    assert np.array_equal(reconstructed_q2, q2_cache["indicators"].astype(bool))
    q2_result_hash = hashlib.sha256(np.packbits(reconstructed_q2).tobytes()).hexdigest()
    assert q2_result_hash == EXPECTED_Q2_RESULT_SHA256
    q2_columns = [str(value) for value in q2_cache["diagnostic_columns"]]
    stable_columns = [
        "fragments", "aabb_candidates", "axis_candidates", "side_witness",
        "full_gjk", "clipped_gjk", "support_calls", "fallbacks", "near_threshold",
    ]
    stable_indices = [q2_columns.index(name) for name in stable_columns]
    assert np.array_equal(
        trial_diagnostics.loc[:2499, stable_columns].to_numpy(dtype=float),
        q2_cache["diagnostics"][:, stable_indices],
    )
    q2_regression = "导通标志逐比特一致，稳定几何计数一致"
else:
    q2_regression = "quick模式未执行完整Q2缓存回归"

result_hash = hashlib.sha256(first_passage.tobytes()).hexdigest()
print(f"载入/完成 {N_TRIALS} 次首次导通试验；记录耗时 {elapsed_seconds:.2f} s")
print(f"结果SHA-256：{result_hash}")
print("与第二问回归：", q2_regression)
print("生成缓存环境：", generator_environment)
print(f"当前读取环境：Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__}")'''
cache_code = cache_code.replace("__FORMAL_TRIALS__", str(FORMAL_TRIALS)).replace(
    "__MODEL_SHA256__", model_sha256
)

analysis_code = r'''from scipy.stats import beta, binom


def phi_percent(count):
    return np.asarray(count, dtype=float) * VOLUME_STEP_PERCENT


def round_half_up_2(value):
    return math.floor(float(value)*100.0 + 0.5 + 1e-12) / 100.0


def clopper_pearson_lower(successes, trials, tail_probability=0.025):
    if successes == 0:
        return 0.0
    return float(beta.ppf(tail_probability, successes, trials-successes+1))


def clopper_pearson_upper(successes, trials, tail_probability=0.025):
    if successes == trials:
        return 1.0
    return float(beta.ppf(1-tail_probability, successes+1, trials-successes))


counts = np.arange(1, N_MAX+1, dtype=int)
successes_curve = np.count_nonzero(first_passage[:, None] <= counts[None, :], axis=0)
probability_curve = successes_curve / N_TRIALS
assert np.all(np.diff(probability_curve) >= 0)
assert probability_curve[-1] >= TARGET_PROBABILITY

quantile_rank = int(math.ceil(TARGET_PROBABILITY*N_TRIALS))
sorted_t = np.sort(first_passage)
n_hat = int(sorted_t[quantile_rank-1])
if n_hat == NO_HIT_SENTINEL:
    raise RuntimeError("90%分位数被右删失；必须扩大N_MAX")
phi_hat = float(phi_percent(n_hat))
reported_hat = round_half_up_2(phi_hat)

# 固定样本下，非参数次序统计量95%分位数区间
order_low = max(1, int(binom.ppf(0.025, N_TRIALS, TARGET_PROBABILITY)))
order_high = min(N_TRIALS, int(binom.ppf(0.975, N_TRIALS, TARGET_PROBABILITY)) + 1)
n_quantile_low = int(sorted_t[order_low-1])
n_quantile_high = int(sorted_t[order_high-1])
if n_quantile_high == NO_HIT_SENTINEL:
    raise RuntimeError("总体N90置信区间上端被右删失；必须扩大N_MAX")
phi_quantile_low = float(phi_percent(n_quantile_low))
phi_quantile_high = float(phi_percent(n_quantile_high))
rounded_quantile_interval = (
    round_half_up_2(phi_quantile_low),
    round_half_up_2(phi_quantile_high),
)
order_rounding_certified = math.isclose(
    rounded_quantile_interval[0], rounded_quantile_interval[1], abs_tol=1e-12
)
order_certified_report = (
    rounded_quantile_interval[0] if order_rounding_certified else math.nan
)

# 2500次发现样本冻结两位小数候选，其余样本独立验证整个舍入档
DISCOVERY_TRIALS = 2500
if N_TRIALS <= DISCOVERY_TRIALS:
    raise RuntimeError("正式验证样本必须独立于2500次发现样本")
discovery_t = first_passage[:DISCOVERY_TRIALS]
validation_t = first_passage[DISCOVERY_TRIALS:]
discovery_rank = int(math.ceil(TARGET_PROBABILITY*DISCOVERY_TRIALS))
n_discovery = int(np.partition(discovery_t, discovery_rank-1)[discovery_rank-1])
candidate_report = round_half_up_2(float(phi_percent(n_discovery)))

rounded_all_counts = np.array([round_half_up_2(phi_percent(n)) for n in counts])
candidate_counts = counts[np.isclose(rounded_all_counts, candidate_report, atol=1e-12)]
if not len(candidate_counts):
    raise RuntimeError("候选两位小数档没有对应整数圆柱数")
candidate_low_n = int(candidate_counts.min())
candidate_high_n = int(candidate_counts.max())
previous_n = candidate_low_n - 1

validation_trials = len(validation_t)
previous_successes = int(np.count_nonzero(validation_t <= previous_n))
candidate_successes = int(np.count_nonzero(validation_t <= candidate_high_n))
previous_probability = previous_successes / validation_trials
candidate_probability = candidate_successes / validation_trials
previous_upper = clopper_pearson_upper(previous_successes, validation_trials, 0.025)
candidate_lower = clopper_pearson_lower(candidate_successes, validation_trials, 0.025)
rounding_bin_certified = (
    previous_upper < TARGET_PROBABILITY
    and candidate_lower >= TARGET_PROBABILITY
)

local_counts = np.unique(np.clip(
    np.r_[n_hat-5:n_hat+6, previous_n, candidate_low_n, candidate_high_n], 1, N_MAX
)).astype(int)
local_successes = successes_curve[local_counts-1]
local_probability = probability_curve[local_counts-1]
local_low, local_high = wilson_interval(local_successes, N_TRIALS)
local_table = pd.DataFrame({
    "圆柱数N": local_counts,
    "实际体积分数/%": phi_percent(local_counts),
    "两位小数显示/%": [round_half_up_2(phi_percent(n)) for n in local_counts],
    "导通次数": local_successes,
    "试验次数": N_TRIALS,
    "经验导通概率": local_probability,
    "边际Wilson 95%下限": local_low,
    "边际Wilson 95%上限": local_high,
})
display(local_table.style.format({
    "实际体积分数/%": "{:.6f}",
    "两位小数显示/%": "{:.2f}",
    "经验导通概率": "{:.4f}",
    "边际Wilson 95%下限": "{:.4f}",
    "边际Wilson 95%上限": "{:.4f}",
}))

validation_table = pd.DataFrame([
    {
        "验证目的": "排除候选舍入档下边界外侧",
        "N": previous_n,
        "实际体积分数/%": phi_percent(previous_n),
        "成功数/样本数": f"{previous_successes}/{validation_trials}",
        "经验概率": previous_probability,
        "97.5%单侧CP界": previous_upper,
        "判据": "上界 < 0.90",
        "是否通过": previous_upper < TARGET_PROBABILITY,
    },
    {
        "验证目的": "确认候选舍入档上边界已达标",
        "N": candidate_high_n,
        "实际体积分数/%": phi_percent(candidate_high_n),
        "成功数/样本数": f"{candidate_successes}/{validation_trials}",
        "经验概率": candidate_probability,
        "97.5%单侧CP界": candidate_lower,
        "判据": "下界 ≥ 0.90",
        "是否通过": candidate_lower >= TARGET_PROBABILITY,
    },
])
display(validation_table.style.format({
    "实际体积分数/%": "{:.6f}",
    "经验概率": "{:.4f}",
    "97.5%单侧CP界": "{:.4f}",
}))

summary_table = pd.DataFrame([{
    "总试验数": N_TRIALS,
    "经验N90": n_hat,
    "经验实际体积分数/%": phi_hat,
    "全样本点估计舍入/%": reported_hat,
    "总体N90分布无关95%区间": f"[{n_quantile_low}, {n_quantile_high}]",
    "区间对应体积分数/%": f"[{phi_quantile_low:.6f}, {phi_quantile_high:.6f}]",
    "区间两端舍入/%": f"[{rounded_quantile_interval[0]:.2f}, {rounded_quantile_interval[1]:.2f}]",
    "次序统计区间认证": order_rounding_certified,
    "独立验证候选档/%": candidate_report,
    "候选整数舍入档": f"[{candidate_low_n}, {candidate_high_n}]",
    "联合95%认证": rounding_bin_certified,
    "右删失样本数": int(np.count_nonzero(first_passage == NO_HIT_SENTINEL)),
    "GJK临界加密数": int(trial_diagnostics["near_threshold"].sum()),
}])
display(summary_table.style.format({
    "经验实际体积分数/%": "{:.6f}",
    "全样本点估计舍入/%": "{:.2f}",
    "独立验证候选档/%": "{:.2f}",
}))

diagnostic_summary = pd.DataFrame({
    "每次试验平均": trial_diagnostics.mean(),
    "全部试验合计": trial_diagnostics.sum(),
    "单次最大": trial_diagnostics.max(),
})
display(diagnostic_summary)

print(f"经验90%分位数：N̂90={n_hat}")
print(f"对应实际体积分数：{phi_hat:.6f}%")
print(f"全样本点估计保留两位：{reported_hat:.2f}%")
print(f"总体N90的分布无关95%区间：[{n_quantile_low}, {n_quantile_high}]")
print(f"次序统计区间的两位小数认证：{order_rounding_certified}")
print(f"两位小数档联合95%认证：{rounding_bin_certified}")
if order_rounding_certified:
    final_report = order_certified_report
    print(f"主认证（分布无关95%次序统计区间）的第三问两位小数答案：{final_report:.2f}%")
    if rounding_bin_certified:
        assert math.isclose(final_report, candidate_report, abs_tol=1e-12)
        print("发现—验证CP分支给出相同结论，作为主认证的佐证。")
else:
    final_report = math.nan
    print("主认证尚未把总体阈值限制在唯一两位小数档；不得把点估计写成已认证的最低值。")
    if rounding_bin_certified:
        print("发现—验证CP分支虽已通过，但不通过两条共享样本路线的择一规则替代预先指定的主认证。")'''

plot_code = r'''import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
phi_axis = phi_percent(counts)

axes[0, 0].step(phi_axis, probability_curve, where="post", color="#1764ab", lw=2)
axes[0, 0].axhline(TARGET_PROBABILITY, color="#d1495b", ls="--", label="目标概率90%")
axes[0, 0].axvline(phi_hat, color="#2a9d8f", ls="--", label=f"经验阈值 {phi_hat:.4f}%")
axes[0, 0].set_xlim(phi_percent(450), phi_percent(N_MAX))
axes[0, 0].set_ylim(0, 1.01)
axes[0, 0].set_xlabel("名义体积分数/%")
axes[0, 0].set_ylabel("经验导通概率")
axes[0, 0].set_title("首次导通数量的经验CDF")
axes[0, 0].grid(alpha=0.25)
axes[0, 0].legend()

zoom = (counts >= n_hat-20) & (counts <= n_hat+20)
axes[0, 1].step(counts[zoom], probability_curve[zoom], where="post", color="#1764ab", lw=2)
axes[0, 1].axhline(TARGET_PROBABILITY, color="#d1495b", ls="--")
axes[0, 1].axvspan(candidate_low_n, candidate_high_n, color="#f4a261", alpha=0.25,
                   label=f"{candidate_report:.2f}%舍入档")
axes[0, 1].axvline(n_hat, color="#2a9d8f", ls="--", label=f"N̂90={n_hat}")
axes[0, 1].set_xlabel("完整原圆柱数量N")
axes[0, 1].set_ylabel("经验导通概率")
axes[0, 1].set_title("90%附近局部放大")
axes[0, 1].grid(alpha=0.25)
axes[0, 1].legend()

finite_t = first_passage[first_passage <= N_MAX]
bins = np.arange(max(250, int(finite_t.min())-1), N_MAX+2, 5)
axes[1, 0].hist(finite_t, bins=bins, color="#457b9d", alpha=0.85)
axes[1, 0].axvline(n_hat, color="#d1495b", ls="--", label=f"N̂90={n_hat}")
axes[1, 0].set_xlabel("首次导通圆柱数T")
axes[1, 0].set_ylabel("试验频数")
axes[1, 0].set_title(f"T的分布（T>707删失数={np.count_nonzero(first_passage==NO_HIT_SENTINEL)}）")
axes[1, 0].grid(axis="y", alpha=0.25)
axes[1, 0].legend()

checkpoints = np.unique(np.r_[np.arange(250, N_TRIALS+1, 250), N_TRIALS]).astype(int)
running_n90 = []
for m in checkpoints:
    rank = int(math.ceil(TARGET_PROBABILITY*m))
    running_n90.append(int(np.partition(first_passage[:m], rank-1)[rank-1]))
axes[1, 1].plot(checkpoints, phi_percent(running_n90), color="#6a4c93", lw=1.8)
axes[1, 1].axhline(phi_hat, color="#2a9d8f", ls="--", label="最终经验估计")
axes[1, 1].set_xlabel("累计试验次数")
axes[1, 1].set_ylabel("经验90%分位体积分数/%")
axes[1, 1].set_title("最低填充量估计的收敛过程")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend()

plt.tight_layout()
plt.show()'''


def conclusion_markdown():
    if not CACHE_PATH.exists():
        return """## 第三问结论\n\n正式缓存尚未生成。运行正式蒙特卡洛后，本节将报告经验最小圆柱数、未舍入体积分数、两位小数答案和独立验证结论。"""
    data = np.load(CACHE_PATH, allow_pickle=False)
    t = data["first_passage"].astype(int)
    if len(t) < FORMAL_TRIALS:
        return f"""## 第三问结论\n\n正式计算正在断点续算，目前完成 {len(t)}/{FORMAL_TRIALS} 次；最终结论以固定样本量完成后的结果为准。"""
    sample_size = len(t)
    step = 100.0 * math.pi * 30.0**2 * 5000.0 / 10000.0**3

    def round2(value):
        return math.floor(float(value) * 100.0 + 0.5 + 1e-12) / 100.0

    rank = int(math.ceil(0.9 * sample_size))
    sorted_t = np.sort(t)
    n_hat = int(sorted_t[rank - 1])
    phi_hat = n_hat * step
    reported = round2(phi_hat)

    order_low = max(1, int(binom.ppf(0.025, sample_size, 0.9)))
    order_high = min(sample_size, int(binom.ppf(0.975, sample_size, 0.9)) + 1)
    n_low = int(sorted_t[order_low - 1])
    n_high = int(sorted_t[order_high - 1])
    if n_high == 708:
        raise RuntimeError("最终N90区间仍被右删失")
    rounded_low, rounded_high = round2(n_low * step), round2(n_high * step)
    order_certified = math.isclose(rounded_low, rounded_high, abs_tol=1e-12)

    discovery = t[:2500]
    validation = t[2500:]
    discovery_rank = int(math.ceil(0.9 * len(discovery)))
    n_discovery = int(np.partition(discovery, discovery_rank - 1)[discovery_rank - 1])
    candidate = round2(n_discovery * step)
    rounded_counts = np.array([round2(n * step) for n in range(1, 708)])
    candidate_counts = np.flatnonzero(np.isclose(rounded_counts, candidate, rtol=0, atol=1e-12)) + 1
    candidate_low, candidate_high = int(candidate_counts.min()), int(candidate_counts.max())
    previous_n = candidate_low - 1
    previous_k = int(np.count_nonzero(validation <= previous_n))
    candidate_k = int(np.count_nonzero(validation <= candidate_high))
    validation_size = len(validation)
    previous_upper = (
        1.0 if previous_k == validation_size
        else float(beta.ppf(0.975, previous_k + 1, validation_size - previous_k))
    )
    candidate_lower = (
        0.0 if candidate_k == 0
        else float(beta.ppf(0.025, candidate_k, validation_size - candidate_k + 1))
    )
    split_certified = previous_upper < 0.9 and candidate_lower >= 0.9

    if order_certified:
        final_report = rounded_low
        certification_text = "预先指定的主认证——总体分位数的分布无关95%次序统计区间——已落在同一个两位小数档"
        if split_certified:
            if not math.isclose(final_report, candidate, abs_tol=1e-12):
                raise RuntimeError("主认证与发现—验证佐证分支给出冲突结论")
            certification_text += "，且发现—验证CP分支给出相同结论作为佐证"
    else:
        final_report = math.nan
        certification_text = "主认证尚未把总体阈值限制在唯一两位小数档"
        if split_certified:
            certification_text += "；发现—验证CP分支虽通过，但不以两条共享样本路线择一的方式替代主认证"

    answer_line = (
        f"因此第三问由预先指定的分布无关95%主认证得到的两位小数答案为 **{final_report:.2f}%**。"
        if math.isfinite(final_report)
        else "因此不能把全样本点估计当作已经认证的最低两位小数填充量。"
    )
    return rf"""## 第三问结论与模型边界

- 在 {sample_size} 次独立均匀蒙特卡洛试验中，首次导通圆柱数的经验 90% 分位数为 **{n_hat} 根**；对应未舍入名义体积分数为 **{phi_hat:.6f}%**，点估计四舍五入为 **{reported:.2f}%**。
- 总体 (N_{{90}}) 的分布无关95%区间为 **[{n_low}, {n_high}]**，对应体积分数 **[{n_low*step:.6f}%, {n_high*step:.6f}%]**，两端舍入为 **[{rounded_low:.2f}%, {rounded_high:.2f}%]**。
- 2500次发现样本冻结候选档 **{candidate:.2f}%**（整数舍入档 (N={candidate_low}\sim{candidate_high})）；其余 {validation_size} 次独立验证中，(N={previous_n}) 的97.5%单侧CP上界为 **{previous_upper:.4f}**，(N={candidate_high}) 的97.5%单侧CP下界为 **{candidate_lower:.4f}**。
- 认证结论：{certification_text}。{answer_line}
- 有限蒙特卡洛给出的是带置信水平的统计结论，不是解析概率的绝对证明。
- 结论依赖第二问声明的 iid 均匀中心、球面各向同性方向，以及“周期裁剪后的碎片为独立节点、盒内只算直接欧氏距离”的模型解释。
- 体积分数按完整原圆柱的名义总体积计算；一根圆柱裁成多片仍只计一次，介质重叠体积不扣除。
"""


nb = nbf.v4.new_notebook()
nb.metadata = q2.get("metadata", {})
nb.cells = [
    nbf.v4.new_markdown_cell(r'''# A题问题3：首次导通分位数—最低填充量

第三问要求在微构体导通概率不低于 90% 的前提下，求介质 A 的最低填充量，并将体积分数精确到百分号后两位。

本 Notebook 承接前两问：第一问提供真实裁剪几何、GJK 和并查集导通判定；第二问确定 iid 均匀随机模型。第三问利用导通事件对圆柱数量的单调性，将概率约束反问题转化为“首次导通圆柱数”的 90% 分位数估计。'''),
    nbf.v4.new_markdown_cell(r'''## 1. 整数优化模型与体积换算

设放入前 (N) 根完整介质 A 后的导通概率为 (P_N)，则

\[
N_{90}=\min\{N\in\mathbb N:P_N\ge0.90\},\qquad
\phi_N=100\%\frac{N\pi(30)^2(5000)}{10000^3}.
\]

计算阶段先搜索整数 (N_{90})，最后才对 (phi_{N_{90}}) 作十进制四舍五入。周期裁剪只改变几何位置，不重复计算一根原圆柱的填充体积。'''),
    nbf.v4.new_code_cell(volume_code),
    nbf.v4.new_markdown_cell(r'''## 2. 与第二问完全一致的随机及边界模型

圆柱轴中点在 ([-5000,5000)^3) 内 iid 均匀，轴方向在单位球面上 iid 各向同性。完整圆柱先按周期规则搬运，再与基本盒求交；所有非空真实裁剪碎片分别作为图节点。同源碎片不自动合并，裁剪后不再使用周期镜像距离。'''),
    nbf.v4.new_code_cell(q2_parts[0]),
    nbf.v4.new_markdown_cell(r'''## 3. 真实裁剪碎片的纯原问题支持函数

碎片严格定义为有限圆柱和立方体的交集。支持函数直接求三变量 SLSQP 原问题，并把候选投影恢复到严格可行域；不使用短圆柱替代、胶囊、多面体或对偶证书。'''),
    nbf.v4.new_code_cell(q2_parts[1]),
    nbf.v4.new_markdown_cell(r'''## 4. AABB、GJK 与并查集导通判定

AABB和轴段距离只负责安全粗筛，最终候选使用真实裁剪碎片 GJK。最短距离不超过1.8 nm时合并并查集；数值未决会明确中止，不能静默记为不导通。'''),
    nbf.v4.new_code_cell(q2_parts[2]),
    nbf.v4.new_markdown_cell(r'''## 5. 首次导通数量 (T)

固定一串 iid 圆柱，逐根完整激活。必须加入第 (N) 根原圆柱的全部裁剪碎片后，才检查左右电极是否同根。定义

\[
T=\min\{N:\text{加入前 }N\text{ 根后首次导通}\}.
\]

由于导通事件随 (N) 单调，(P_N=P(T\le N))，故 (N_{90}) 就是 (T) 的总体90%分位数。707根仍不导通用哨兵708表示右删失。'''),
    nbf.v4.new_code_cell(q2_parts[3]),
    nbf.v4.new_code_cell(addon),
    nbf.v4.new_markdown_cell("## 6. 回归检查"),
    nbf.v4.new_code_cell(regression_code),
    nbf.v4.new_markdown_cell(r'''## 7. 正式蒙特卡洛与可复现缓存

每个试验使用 `SeedSequence([20260808, trial_id])` 建立独立 PCG64 随机流，因此结果与串并行调度无关。默认读取正式缓存；从头复算可切换为 `production`，同一数值环境断点续算用 `resume`。

正式缓存同时保存模型哈希、随机流、数值环境、全部几何容差与诊断。前2500个试验使用和第二问相同的随机流，并要求由 (T) 重构的四档布尔值逐比特一致、稳定几何计数一致。SLSQP的投影和多起点重试次数只作数值审计，不要求跨进程批次逐项相等。'''),
    nbf.v4.new_code_cell(cache_code),
    nbf.v4.new_markdown_cell(r'''## 8. 经验90%分位数与独立统计验证

点估计为 (\widehat N_{90}=T_{(\lceil0.9M\rceil)})。另外给出固定样本下的非参数次序统计量95%区间。

预先指定的主认证使用全部固定样本构造总体 \(N_{90}\) 的分布无关95%次序统计区间；只有该区间两端落入同一两位小数舍入档，才给出经过认证的最终两位答案。

另设一个发现—验证佐证分支：候选舍入档只由前2500次发现样本冻结，其余样本只用于验证候选档下边界外侧和上边界。例如0.82%对应所有四舍五入后显示0.82%的整数数量，而不是只验证名义0.82%换算出的单个数量。两个端点分别使用97.5%单侧Clopper–Pearson界，经Bonferroni后联合置信水平至少95%。该分支与全样本主认证共享数据，因此不采用“两条95%路线任一通过即可”的择一规则；它只用于佐证主认证。'''),
    nbf.v4.new_code_cell(analysis_code),
    nbf.v4.new_markdown_cell("## 9. 概率曲线、首次导通分布与收敛性"),
    nbf.v4.new_code_cell(plot_code),
    nbf.v4.new_markdown_cell(conclusion_markdown()),
]

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

nbf.write(nb, OUTPUT)
print(f"Q2 source: {q2_path}")
print(f"Q3 notebook: {OUTPUT} ({len(nb.cells)} cells)")
print(f"model SHA-256: {model_sha256}")
