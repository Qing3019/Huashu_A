from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
import types
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed


BASE_SEED = 20260808
Q3_STREAM_SIZE = 707
N_JOBS = min(10, os.cpu_count() or 1)
SCRIPT_DIR = Path(__file__).resolve().parent
Q4_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tools" else SCRIPT_DIR
WORKSPACE_DIR = Q4_DIR.parent if Q4_DIR.name == "Q4" else Q4_DIR
Q3_CACHE = WORKSPACE_DIR / "q3_first_passage_10000.npz"
NOTEBOOK_PATH = Q4_DIR / "问题4_混合介质最低成本_严格GJK.ipynb"
SEARCH_DIR = Q4_DIR / "search"
RESULT_DIR = Q4_DIR / "results"

STAGES = {
    "coarse": {
        "levels": np.unique(
            np.concatenate(
                [
                    np.arange(0, 501, 25),
                    np.arange(525, 576, 10),
                    np.arange(580, 609, 4),
                ]
            )
        ).astype(np.int16),
        "trial_indices": np.arange(200, dtype=np.int64),
        "b_max": 5395,
        "cache": SEARCH_DIR / "q4_q3A_coarse_200.npz",
    },
    "fine": {
        "levels": np.arange(500, 609, dtype=np.int16),
        "trial_indices": np.arange(1000, dtype=np.int64),
        "b_max": 965,
        "cache": SEARCH_DIR / "q4_q3A_fine_1000.npz",
    },
    "refine": {
        "levels": np.arange(575, 609, dtype=np.int16),
        "trial_indices": np.arange(10000, dtype=np.int64),
        "b_max": 301,
        "cache": SEARCH_DIR / "q4_q3A_refine_10000.npz",
    },
    "refine_low": {
        "levels": np.arange(550, 575, dtype=np.int16),
        "trial_indices": np.arange(10000, dtype=np.int64),
        "b_max": 518,
        "cache": SEARCH_DIR / "q4_q3A_refine_low_10000.npz",
    },
}


_ENGINE = None


def load_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    notebook_path = NOTEBOOK_PATH
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    module_name = "q4_same_q3_a_stream_engine"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for cell_index in (2, 4, 5, 6):
        exec(compile(notebook.cells[cell_index].source, f"{notebook_path}:cell{cell_index}", "exec"), namespace)

    q3 = namespace["q3"]
    original_sample_cylinders = q3.sample_cylinders

    def sample_q3_locked_prefix(n, rng):
        # Q3 draws all 707 centers first and all 707 directions second. Drawing
        # only n cylinders would change the direction stream, even with the same seed.
        return original_sample_cylinders(Q3_STREAM_SIZE, rng)[: int(n)]

    q3.sample_cylinders = sample_q3_locked_prefix
    _ENGINE = namespace
    return namespace


def load_q3_critical_counts():
    with np.load(Q3_CACHE, allow_pickle=False) as data:
        assert int(data["base_seed"]) == BASE_SEED
        assert int(data["max_cylinders"]) == Q3_STREAM_SIZE
        critical = data["critical_counts"].astype(np.int32)
    assert critical.shape == (10000,)
    return critical


def simulate_one(trial_index, levels, b_max):
    engine = load_engine()
    values, diagnostics = engine["simulate_frontier"](
        int(trial_index),
        np.asarray(levels, dtype=np.int16),
        a_max=int(np.max(levels)),
        b_max=int(b_max),
        base_seed=BASE_SEED,
    )
    return np.asarray(values, dtype=np.int16), diagnostics


def atomic_save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def initialize_or_load(stage_name, levels, trial_indices, b_max, critical):
    cache = STAGES[stage_name]["cache"]
    shape = (len(trial_indices), len(levels))
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            values = data["first_b"].astype(np.int16)
            done = data["done"].astype(bool)
            assert values.shape == shape
            assert done.shape == (len(trial_indices),)
            assert np.array_equal(data["a_levels"], levels)
            assert np.array_equal(data["trial_indices"], trial_indices)
            assert int(data["base_seed"]) == BASE_SEED
            assert int(data["q3_stream_size"]) == Q3_STREAM_SIZE
            assert int(data["b_max"]) == b_max
            elapsed = float(data["elapsed_seconds"])
        return values, done, elapsed

    values = np.full(shape, -1, dtype=np.int16)
    done = np.zeros(len(trial_indices), dtype=bool)
    for row, trial_index in enumerate(trial_indices):
        known = critical[int(trial_index)] <= levels
        values[row, known] = 0
        if bool(known[0]):
            done[row] = True
    return values, done, 0.0


def save_stage(stage_name, levels, trial_indices, b_max, values, done, elapsed):
    atomic_save(
        STAGES[stage_name]["cache"],
        first_b=values,
        done=done,
        a_levels=levels,
        trial_indices=trial_indices,
        completed_trials=int(np.count_nonzero(done)),
        requested_trials=len(trial_indices),
        elapsed_seconds=float(elapsed),
        base_seed=BASE_SEED,
        q3_stream_size=Q3_STREAM_SIZE,
        b_max=int(b_max),
        coupling=np.asarray(
            "SeedSequence([20260808, trial]); q3.sample_cylinders(707, rng) prefix"
        ),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def run_stage(stage_name, batch_size=100):
    config = STAGES[stage_name]
    levels = config["levels"]
    trial_indices = config["trial_indices"]
    b_max = int(config["b_max"])
    critical = load_q3_critical_counts()
    values, done, elapsed = initialize_or_load(
        stage_name, levels, trial_indices, b_max, critical
    )
    pending_rows = np.flatnonzero(~done)
    print(
        f"stage={stage_name} rows={len(trial_indices)} pending={len(pending_rows)} "
        f"levels={int(levels[0])}..{int(levels[-1])} b_max={b_max}",
        flush=True,
    )

    for start in range(0, len(pending_rows), int(batch_size)):
        rows = pending_rows[start : start + int(batch_size)]
        started = time.perf_counter()
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(simulate_one)(int(trial_indices[row]), levels, b_max)
            for row in rows
        )
        elapsed += time.perf_counter() - started
        for row, (result, _) in zip(rows, outputs):
            trial_index = int(trial_indices[row])
            known = critical[trial_index] <= levels
            result[known] = 0
            values[row] = result
            done[row] = True
        save_stage(
            stage_name, levels, trial_indices, b_max, values, done, elapsed
        )
        print(
            f"stage={stage_name} completed={int(np.count_nonzero(done))}/"
            f"{len(done)} elapsed={elapsed:.1f}s",
            flush=True,
        )

    summarize_stage(stage_name)


def summarize_stage(stage_name):
    config = STAGES[stage_name]
    with np.load(config["cache"], allow_pickle=False) as data:
        values = data["first_b"].astype(np.int64)
        done = data["done"].astype(bool)
        levels = data["a_levels"].astype(np.int64)
        b_max = int(data["b_max"])
    if not np.all(done):
        print(f"stage={stage_name} incomplete", flush=True)
        return

    rank = math.ceil(0.90 * len(values)) - 1
    b90 = np.sort(values, axis=0)[rank]
    frame = pd.DataFrame({"N_A": levels, "N_B_90": b90})
    frame["K"] = 567 * frame["N_A"] + 64 * frame["N_B_90"]
    frame["cost_yuan"] = np.pi * frame["K"] / 120000.0
    frame["censored"] = frame["N_B_90"] > b_max
    print(frame.sort_values("K").head(15).to_string(index=False), flush=True)


def validate_a_stream(trial_indices=(0, 1, 2, 17, 99)):
    critical = load_q3_critical_counts()
    engine = load_engine()
    for trial_index in trial_indices:
        threshold = int(critical[trial_index])
        levels = sorted(
            set(value for value in (threshold - 1, threshold) if 0 <= value <= 608)
        )
        if not levels:
            continue
        graph, _ = engine["build_contact_graph"](
            int(trial_index), a_max=608, b_max=0, base_seed=BASE_SEED
        )
        observed = {
            level: engine["first_b_for_a"](graph, level, 0) == 0
            for level in levels
        }
        expected = {level: threshold <= level for level in levels}
        if observed != expected:
            raise AssertionError(
                f"A stream mismatch at trial {trial_index}: "
                f"critical={threshold}, observed={observed}, expected={expected}"
            )
        print(
            f"A-stream trial={trial_index} critical={threshold} verified",
            flush=True,
        )


def final_summary():
    critical = load_q3_critical_counts()
    baseline_successes = int(np.count_nonzero(critical <= 609))
    rows = [(609, 0, 567 * 609, baseline_successes, baseline_successes / 10000)]

    formal_blocks = []
    for stage_name in ("refine_low", "refine"):
        with np.load(STAGES[stage_name]["cache"], allow_pickle=False) as data:
            values = data["first_b"].astype(np.int64)
            levels = data["a_levels"].astype(np.int64)
            b_max = int(data["b_max"])
            assert np.all(data["done"])
        formal_blocks.append((values, levels, b_max))

        rank = math.ceil(0.90 * len(values)) - 1
        b90 = np.sort(values, axis=0)[rank]
        for column, (a_count, b_count) in enumerate(zip(levels, b90)):
            if b_count > b_max:
                continue
            successes = int(np.count_nonzero(values[:, column] <= b_count))
            rows.append(
                (
                    int(a_count),
                    int(b_count),
                    int(567 * a_count + 64 * b_count),
                    successes,
                    successes / len(values),
                )
            )

    frame = pd.DataFrame(
        rows, columns=["N_A", "N_B_90", "K", "successes", "p_hat"]
    ).sort_values("K", ignore_index=True)
    frame["cost_yuan"] = np.pi * frame["K"] / 120000.0
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / "q4_q3A_final_frontier.csv", index=False, encoding="utf-8-sig")
    print(frame.head(20).to_string(index=False), flush=True)

    best = frame.iloc[0]
    cheaper_checks = []
    for values, levels, _ in formal_blocks:
        for column, a_count in enumerate(levels):
            b_limit = (int(best.K) - 1 - 567 * int(a_count)) // 64
            if b_limit >= 0:
                successes = int(np.count_nonzero(values[:, column] <= b_limit))
                cheaper_checks.append(
                    (int(a_count), int(b_limit), successes, successes / len(values))
                )
    checks = pd.DataFrame(
        cheaper_checks,
        columns=["N_A", "max_cheaper_N_B", "successes", "p_hat"],
    ).sort_values("p_hat", ascending=False)
    checks.to_csv(RESULT_DIR / "q4_q3A_cheaper_checks.csv", index=False, encoding="utf-8-sig")
    print("\nBest:", best.to_dict(), flush=True)
    print("\nBest cheaper checks:\n", checks.head(15).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["validate", "coarse", "fine", "refine_low", "refine", "summary"],
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.stage == "validate":
        validate_a_stream()
    elif args.stage == "summary":
        final_summary()
    else:
        run_stage(args.stage, args.batch_size)


if __name__ == "__main__":
    main()
