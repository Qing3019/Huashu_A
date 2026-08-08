from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom


MAIN_PATH = Path("q3_first_passage_results.npz")
SEGMENT_PATH = Path("q3_first_passage_segment_10000_22500.npz")
TEMP_PATH = Path("q3_first_passage_results.merged.tmp.npz")
Q2_PATH = Path("q2_mc_results_2500.npz")
EXPECTED_MODEL_SHA256 = (
    "ed95e324f7c4622d878e2944b209165780dcd0b02e885b65e6ae5b543d106a35"
)
EXPECTED_Q2_RESULT_SHA256 = (
    "902fe5bc0a119e1dbeac40f756e5aba149dc0f1ee55de897b50d8d8f434bf7d3"
)
TOTAL_TRIALS = 22500
DISCOVERY_TRIALS = 2500
N_MAX = 707
SENTINEL = 708
STEP_PERCENT = 100.0 * math.pi * 30.0**2 * 5000.0 / 10000.0**3


def round_half_up_2(value):
    return math.floor(float(value) * 100.0 + 0.5 + 1e-12) / 100.0


def cp_lower(k, n, tail=0.025):
    return 0.0 if k == 0 else float(beta.ppf(tail, k, n-k+1))


def cp_upper(k, n, tail=0.025):
    return 1.0 if k == n else float(beta.ppf(1-tail, k+1, n-k))


main = np.load(MAIN_PATH, allow_pickle=False)
segment = np.load(SEGMENT_PATH, allow_pickle=False)

main_t = main["first_passage"].astype(np.int16)
segment_t = segment["first_passage"].astype(np.int16)
assert len(main_t) == 10000
assert len(segment_t) == 12500
assert int(main["completed_trials"]) == int(main["requested_trials"]) == 10000
assert int(segment["completed_trials"]) == int(segment["requested_trials"]) == 12500
assert int(segment["trial_start"]) == 10000
assert int(segment["trial_end"]) == 22500
assert int(segment["no_hit_sentinel"]) == SENTINEL
assert float(segment["target_probability"]) == 0.90
assert np.array_equal(main["trial_indices"], np.arange(10000))
assert np.array_equal(segment["trial_indices"], np.arange(10000, 22500))
assert str(main["model_code_sha256"]) == EXPECTED_MODEL_SHA256
assert str(segment["model_code_sha256"]) == EXPECTED_MODEL_SHA256
assert int(main["base_seed"]) == int(segment["base_seed"]) == 20260808
assert int(main["n_max"]) == int(segment["n_max"]) == N_MAX
for key in ("generator_python", "generator_numpy", "generator_scipy", "bit_generator"):
    assert str(main[key]) == str(segment[key])

main_columns = [str(value) for value in main["diagnostic_columns"]]
segment_columns = [str(value) for value in segment["diagnostic_columns"]]
assert main_columns == segment_columns
first_passage = np.concatenate((main_t, segment_t))
diagnostics = np.vstack((main["diagnostics"], segment["diagnostics"]))
trial_indices = np.arange(TOTAL_TRIALS, dtype=np.int32)
assert len(first_passage) == TOTAL_TRIALS
assert np.all((first_passage >= 1) & (first_passage <= SENTINEL))
assert diagnostics.shape[0] == TOTAL_TRIALS
assert np.isfinite(diagnostics).all()
assert (diagnostics >= 0).all()
near_index = main_columns.index("near_threshold")
assert int(diagnostics[:, near_index].sum()) == 0

# Exact regression to Q2 on the shared first 2500 random streams.
q2 = np.load(Q2_PATH, allow_pickle=False)
levels = q2["levels"].astype(int)
reconstructed = first_passage[:2500, None] <= levels[None, :]
assert np.array_equal(reconstructed, q2["indicators"].astype(bool))
q2_hash = hashlib.sha256(np.packbits(reconstructed).tobytes()).hexdigest()
assert q2_hash == EXPECTED_Q2_RESULT_SHA256
q2_columns = [str(value) for value in q2["diagnostic_columns"]]
stable_columns = [
    "fragments", "aabb_candidates", "axis_candidates", "side_witness",
    "full_gjk", "clipped_gjk", "support_calls", "fallbacks", "near_threshold",
]
assert np.array_equal(
    diagnostics[:2500, [main_columns.index(name) for name in stable_columns]],
    q2["diagnostics"][:, [q2_columns.index(name) for name in stable_columns]],
)

# Point estimate and distribution-free order-statistic interval.
sorted_t = np.sort(first_passage)
rank = int(math.ceil(0.9 * TOTAL_TRIALS))
n_hat = int(sorted_t[rank-1])
order_low = max(1, int(binom.ppf(0.025, TOTAL_TRIALS, 0.9)))
order_high = min(TOTAL_TRIALS, int(binom.ppf(0.975, TOTAL_TRIALS, 0.9)) + 1)
n_low = int(sorted_t[order_low-1])
n_high = int(sorted_t[order_high-1])
assert n_high != SENTINEL
rounded_interval = (
    round_half_up_2(n_low * STEP_PERCENT),
    round_half_up_2(n_high * STEP_PERCENT),
)
order_certified = math.isclose(*rounded_interval, abs_tol=1e-12)

# Frozen discovery candidate and independent 20,000-trial validation.
discovery = first_passage[:DISCOVERY_TRIALS]
validation = first_passage[DISCOVERY_TRIALS:]
discovery_rank = int(math.ceil(0.9 * DISCOVERY_TRIALS))
n_discovery = int(np.partition(discovery, discovery_rank-1)[discovery_rank-1])
candidate = round_half_up_2(n_discovery * STEP_PERCENT)
labels = np.array([round_half_up_2(n * STEP_PERCENT) for n in range(1, N_MAX+1)])
candidate_counts = np.flatnonzero(np.isclose(labels, candidate, rtol=0, atol=1e-12)) + 1
candidate_low = int(candidate_counts.min())
candidate_high = int(candidate_counts.max())
previous_n = candidate_low - 1
previous_k = int(np.count_nonzero(validation <= previous_n))
candidate_k = int(np.count_nonzero(validation <= candidate_high))
previous_upper = cp_upper(previous_k, len(validation))
candidate_lower = cp_lower(candidate_k, len(validation))
split_certified = previous_upper < 0.9 and candidate_lower >= 0.9
if order_certified and split_certified:
    assert math.isclose(rounded_interval[0], candidate, abs_tol=1e-12)

payload = {key: main[key] for key in main.files}
payload.update(
    first_passage=first_passage,
    trial_indices=trial_indices,
    diagnostics=diagnostics,
    completed_trials=TOTAL_TRIALS,
    requested_trials=TOTAL_TRIALS,
    elapsed_seconds=float(main["elapsed_seconds"]) + float(segment["elapsed_seconds"]),
    parallel_segments=2,
    main_segment_elapsed_seconds=float(main["elapsed_seconds"]),
    extension_segment_elapsed_seconds=float(segment["elapsed_seconds"]),
    result_sha256=hashlib.sha256(first_passage.tobytes()).hexdigest(),
)
np.savez_compressed(TEMP_PATH, **payload)
check = np.load(TEMP_PATH, allow_pickle=False)
assert np.array_equal(check["first_passage"], first_passage)
assert np.array_equal(check["trial_indices"], trial_indices)
assert int(check["completed_trials"]) == TOTAL_TRIALS
check.close()
main.close()
segment.close()
q2.close()
os.replace(TEMP_PATH, MAIN_PATH)

print(f"merged trials={TOTAL_TRIALS}")
print(f"result SHA-256={payload['result_sha256']}")
print(f"N90 point estimate={n_hat}; phi={n_hat*STEP_PERCENT:.12f}%")
print(f"order-stat 95% interval=[{n_low}, {n_high}], rounded={rounded_interval}")
print(
    f"discovery candidate={candidate:.2f}% bin=[{candidate_low},{candidate_high}]; "
    f"validation U(N={previous_n})={previous_upper:.8f}, "
    f"L(N={candidate_high})={candidate_lower:.8f}, certified={split_certified}"
)
print(f"order-certified={order_certified}; censored={int(np.count_nonzero(first_passage==SENTINEL))}")
