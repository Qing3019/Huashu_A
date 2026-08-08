from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import _q3_engine_full as engine


TRIAL_START = 10000
TRIAL_END = 22500
N_JOBS = 12
BATCH_SIZE = 100
BASE_SEED = 20260808
N_MAX = 707
CACHE_PATH = Path("q3_first_passage_segment_10000_22500.npz")
MODEL_SHA256 = hashlib.sha256(Path("_q3_engine_full.py").read_bytes()).hexdigest()


def atomic_save(first_passage, diagnostics, elapsed_seconds):
    frame = pd.DataFrame(diagnostics)
    temporary = Path(str(CACHE_PATH) + ".tmp.npz")
    trial_indices = np.arange(
        TRIAL_START, TRIAL_START + len(first_passage), dtype=np.int32
    )
    np.savez_compressed(
        temporary,
        first_passage=np.asarray(first_passage, dtype=np.int16),
        trial_indices=trial_indices,
        diagnostics=frame.to_numpy(dtype=float),
        diagnostic_columns=np.asarray(frame.columns, dtype=str),
        completed_trials=int(len(first_passage)),
        requested_trials=int(TRIAL_END - TRIAL_START),
        trial_start=TRIAL_START,
        trial_end=TRIAL_END,
        elapsed_seconds=float(elapsed_seconds),
        base_seed=BASE_SEED,
        n_max=N_MAX,
        no_hit_sentinel=N_MAX + 1,
        target_probability=0.90,
        model_code_sha256=MODEL_SHA256,
        generator_python=platform.python_version(),
        generator_numpy=np.__version__,
        generator_scipy=scipy.__version__,
        bit_generator="PCG64",
    )
    os.replace(temporary, CACHE_PATH)


def load_partial():
    data = np.load(CACHE_PATH, allow_pickle=False)
    assert int(data["trial_start"]) == TRIAL_START
    assert int(data["trial_end"]) == TRIAL_END
    assert int(data["base_seed"]) == BASE_SEED
    assert int(data["n_max"]) == N_MAX
    assert str(data["model_code_sha256"]) == MODEL_SHA256
    assert str(data["generator_python"]) == platform.python_version()
    assert str(data["generator_numpy"]) == np.__version__
    assert str(data["generator_scipy"]) == scipy.__version__
    first_passage = data["first_passage"].astype(np.int16)
    assert int(data["completed_trials"]) == len(first_passage)
    assert np.array_equal(
        data["trial_indices"],
        np.arange(TRIAL_START, TRIAL_START + len(first_passage)),
    )
    columns = [str(value) for value in data["diagnostic_columns"]]
    diagnostics = pd.DataFrame(data["diagnostics"], columns=columns)
    assert len(diagnostics) == len(first_passage)
    return first_passage, diagnostics, float(data["elapsed_seconds"])


def main():
    if CACHE_PATH.exists():
        old_t, old_diagnostics, elapsed = load_partial()
        t_batches = [old_t]
        diagnostic_batches = [old_diagnostics]
        completed = len(old_t)
        print(f"resuming segment {completed}/{TRIAL_END-TRIAL_START}", flush=True)
    else:
        t_batches, diagnostic_batches = [], []
        elapsed = 0.0
        completed = 0
        print(f"starting segment 0/{TRIAL_END-TRIAL_START}", flush=True)

    total = TRIAL_END - TRIAL_START
    for local_start in range(completed, total, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, total - local_start)
        batch_t, batch_diagnostics, batch_elapsed = engine.run_first_passage_monte_carlo(
            batch_size,
            max_cylinders=N_MAX,
            base_seed=BASE_SEED,
            n_jobs=N_JOBS,
            trial_start=TRIAL_START + local_start,
        )
        t_batches.append(batch_t)
        diagnostic_batches.append(batch_diagnostics)
        elapsed += batch_elapsed
        first_passage = np.concatenate(t_batches)
        diagnostics = pd.concat(diagnostic_batches, ignore_index=True)
        atomic_save(first_passage, diagnostics, elapsed)
        print(
            f"completed segment {len(first_passage)}/{total}; "
            f"batch={batch_elapsed:.2f}s total={elapsed:.2f}s",
            flush=True,
        )

    first_passage = np.concatenate(t_batches)
    diagnostics = pd.concat(diagnostic_batches, ignore_index=True)
    assert len(first_passage) == total
    assert np.isfinite(diagnostics.to_numpy(dtype=float)).all()
    assert (diagnostics.to_numpy(dtype=float) >= 0).all()
    assert int(diagnostics["near_threshold"].sum()) == 0
    digest = hashlib.sha256(first_passage.tobytes()).hexdigest()
    print(f"segment complete; SHA-256={digest}", flush=True)


if __name__ == "__main__":
    main()
