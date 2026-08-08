from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy

import _q3_engine_full as engine


BASE_SEED = 20260808
N_MAX = 707
NO_HIT_SENTINEL = N_MAX + 1
TARGET_PROBABILITY = 0.90
N_JOBS = 10
BATCH_SIZE = 100
CACHE_PATH = Path("q3_first_passage_results.npz")
MODEL_CODE_SHA256 = hashlib.sha256(
    Path("_q3_engine_full.py").read_bytes()
).hexdigest()
Q2_MODEL_CODE_SHA256 = (
    "a7cdb15115b2f9757e10d45cc3692d674c3751f82fb39be5fdc91afe653df00e"
)
Q2_INDICATOR_SHA256 = (
    "902fe5bc0a119e1dbeac40f756e5aba149dc0f1ee55de897b50d8d8f434bf7d3"
)


def current_environment():
    return {
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "RNG": "PCG64",
    }


def atomic_save(first_passage, diagnostics, elapsed_seconds, target_trials):
    frame = pd.DataFrame(diagnostics)
    temporary = Path(str(CACHE_PATH) + ".tmp.npz")
    np.savez_compressed(
        temporary,
        first_passage=np.asarray(first_passage, dtype=np.int16),
        trial_indices=np.arange(len(first_passage), dtype=np.int32),
        diagnostics=frame.to_numpy(dtype=float),
        diagnostic_columns=np.asarray(frame.columns, dtype=str),
        completed_trials=int(len(first_passage)),
        requested_trials=int(target_trials),
        elapsed_seconds=float(elapsed_seconds),
        base_seed=BASE_SEED,
        n_max=N_MAX,
        no_hit_sentinel=NO_HIT_SENTINEL,
        target_probability=TARGET_PROBABILITY,
        model_code_sha256=MODEL_CODE_SHA256,
        q2_model_code_sha256=Q2_MODEL_CODE_SHA256,
        q2_indicator_sha256=Q2_INDICATOR_SHA256,
        generator_python=platform.python_version(),
        generator_numpy=np.__version__,
        generator_scipy=scipy.__version__,
        bit_generator="PCG64",
        random_stream_schema="SeedSequence([base_seed, trial_index])",
        box_half=engine.BOX_HALF,
        cylinder_length=engine.LENGTH,
        cylinder_radius=engine.RADIUS,
        threshold=engine.THRESHOLD,
        numeric_tolerance=engine.NUM_TOL,
        feasibility_tolerance=engine.FEAS_TOL,
        support_feasibility_tolerance=engine.SUPPORT_FEAS_TOL,
        support_stability_nm=engine.SUPPORT_STABILITY_NM,
        near_threshold_tolerance=engine.NEAR_THRESHOLD_TOL,
        gjk_gap_tolerance=engine.GJK_GAP_TOL,
        gjk_far_guard=engine.GJK_FAR_GUARD,
        contact_limit=engine.CONTACT_LIMIT,
    )
    os.replace(temporary, CACHE_PATH)


def load_partial():
    data = np.load(CACHE_PATH, allow_pickle=False)
    assert int(data["base_seed"]) == BASE_SEED
    assert int(data["n_max"]) == N_MAX
    assert int(data["no_hit_sentinel"]) == NO_HIT_SENTINEL
    assert str(data["model_code_sha256"]) == MODEL_CODE_SHA256
    assert str(data["generator_python"]) == platform.python_version()
    assert str(data["generator_numpy"]) == np.__version__
    assert str(data["generator_scipy"]) == scipy.__version__
    assert str(data["bit_generator"]) == "PCG64"
    first_passage = data["first_passage"].astype(np.int16)
    assert int(data["completed_trials"]) == len(first_passage)
    assert np.array_equal(
        data["trial_indices"], np.arange(len(first_passage), dtype=np.int32)
    )
    columns = [str(value) for value in data["diagnostic_columns"]]
    diagnostics = pd.DataFrame(data["diagnostics"], columns=columns)
    assert len(diagnostics) == len(first_passage)
    return first_passage, diagnostics, float(data["elapsed_seconds"])


def validate_against_q2(first_passage, diagnostics):
    if len(first_passage) < 2500:
        return
    q2 = np.load("q2_mc_results_2500.npz", allow_pickle=False)
    levels = q2["levels"].astype(int)
    reconstructed = first_passage[:2500, None] <= levels[None, :]
    assert np.array_equal(reconstructed, q2["indicators"].astype(bool))
    packed_hash = hashlib.sha256(
        np.packbits(reconstructed).tobytes()
    ).hexdigest()
    assert packed_hash == Q2_INDICATOR_SHA256
    q2_columns = [str(value) for value in q2["diagnostic_columns"]]
    q2_diagnostics = q2["diagnostics"]
    stable_columns = [
        "fragments",
        "aabb_candidates",
        "axis_candidates",
        "side_witness",
        "full_gjk",
        "clipped_gjk",
        "support_calls",
        "fallbacks",
        "near_threshold",
    ]
    stable_indices = [q2_columns.index(name) for name in stable_columns]
    assert np.array_equal(
        diagnostics.loc[:2499, stable_columns].to_numpy(dtype=float),
        q2_diagnostics[:, stable_indices],
    )


def main():
    target_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 22500
    if target_trials < 2500:
        raise ValueError("formal target must be at least 2500 trials")

    if CACHE_PATH.exists():
        old_t, old_diagnostics, elapsed_seconds = load_partial()
        first_passage_batches = [old_t]
        diagnostic_batches = [old_diagnostics]
        completed = len(old_t)
        print(f"resuming {completed}/{target_trials}", flush=True)
    else:
        first_passage_batches = []
        diagnostic_batches = []
        elapsed_seconds = 0.0
        completed = 0
        print(f"starting 0/{target_trials}", flush=True)

    if completed > target_trials:
        raise RuntimeError(
            f"cache already has {completed} trials, exceeding requested {target_trials}"
        )

    for start in range(completed, target_trials, BATCH_SIZE):
        batch_trials = min(BATCH_SIZE, target_trials - start)
        batch_t, batch_diagnostics, batch_elapsed = (
            engine.run_first_passage_monte_carlo(
                batch_trials,
                max_cylinders=N_MAX,
                base_seed=BASE_SEED,
                n_jobs=N_JOBS,
                trial_start=start,
            )
        )
        first_passage_batches.append(batch_t)
        diagnostic_batches.append(batch_diagnostics)
        elapsed_seconds += batch_elapsed
        first_passage = np.concatenate(first_passage_batches)
        diagnostics = pd.concat(diagnostic_batches, ignore_index=True)
        atomic_save(first_passage, diagnostics, elapsed_seconds, target_trials)
        print(
            f"completed {len(first_passage)}/{target_trials}; "
            f"batch={batch_elapsed:.2f}s total={elapsed_seconds:.2f}s",
            flush=True,
        )

    first_passage = np.concatenate(first_passage_batches)
    diagnostics = pd.concat(diagnostic_batches, ignore_index=True)
    validate_against_q2(first_passage, diagnostics)
    assert np.isfinite(diagnostics.to_numpy(dtype=float)).all()
    assert (diagnostics.to_numpy(dtype=float) >= 0).all()
    assert int(diagnostics["near_threshold"].sum()) == 0
    result_hash = hashlib.sha256(first_passage.tobytes()).hexdigest()
    point_index = int(np.ceil(TARGET_PROBABILITY * len(first_passage))) - 1
    n_hat = int(np.partition(first_passage, point_index)[point_index])
    print("Q2 bitwise and stable-geometry regression: PASS", flush=True)
    print(
        f"N90 point estimate={n_hat}; censored="
        f"{int(np.count_nonzero(first_passage == NO_HIT_SENTINEL))}",
        flush=True,
    )
    print(f"result SHA-256={result_hash}", flush=True)
    print(f"model SHA-256={MODEL_CODE_SHA256}", flush=True)
    print(f"environment={current_environment()}", flush=True)


if __name__ == "__main__":
    main()
