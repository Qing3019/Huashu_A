from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed

try:
    from q4_same_q3_a_stream_runner import load_engine
except ModuleNotFoundError:
    from Q4.tools.q4_same_q3_a_stream_runner import load_engine


BASE_SEED = 20260808
N_JOBS = min(10, os.cpu_count() or 1)
BATCH_SIZE = 100
ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_Q3 = ROOT / "Q3" / "q3_first_passage_results.npz"
SEGMENT_Q3 = ROOT / "Q3" / "q3_first_passage_segment_10000_22500.npz"
ORIGINAL_FRONTIER = ROOT / "Q4" / "results" / "q4_q3A_refine_10000_A550_609.npz"

SEARCH_DIR = ROOT / "Q4" / "search"
RESULT_DIR = ROOT / "Q4" / "results"
BLOCKS = {
    "validation_low": {
        "levels": np.arange(550, 575, dtype=np.int16),
        "indices": np.arange(10_000, 12_500, dtype=np.int32),
        "b_max": 518,
        "cache": SEARCH_DIR / "q4_frontier_validation_2500_A550_574.npz",
    },
    "validation_high": {
        "levels": np.arange(575, 612, dtype=np.int16),
        "indices": np.arange(10_000, 12_500, dtype=np.int32),
        "b_max": 301,
        "cache": SEARCH_DIR / "q4_frontier_validation_2500_A575_611.npz",
    },
    "original_extra": {
        "levels": np.arange(609, 612, dtype=np.int16),
        "indices": np.arange(0, 10_000, dtype=np.int32),
        "b_max": 301,
        "cache": SEARCH_DIR / "q4_frontier_original_10000_A609_611.npz",
    },
}

OUTPUT = RESULT_DIR / "q4_q3A_refine_12500_A550_611.npz"
FRONTIER_CSV = RESULT_DIR / "q4_q3A_final_frontier_12500_A550_611.csv"
CHEAPER_CSV = RESULT_DIR / "q4_q3A_cheaper_checks_12500_A550_611.csv"


def load_critical_counts() -> dict[int, int]:
    mapping: dict[int, int] = {}
    for path in (ORIGINAL_Q3, SEGMENT_Q3):
        with np.load(path, allow_pickle=False) as data:
            assert int(data["base_seed"]) == BASE_SEED
            indices = data["trial_indices"].astype(np.int32)
            values = data["first_passage"].astype(np.int16)
        mapping.update({int(i): int(v) for i, v in zip(indices, values)})
    required = set(range(12_500))
    if not required.issubset(mapping):
        raise RuntimeError("Q3 first-passage cache does not cover trials 0..12499")
    return mapping


def simulate_one(trial_index: int, levels: np.ndarray, b_max: int):
    engine = load_engine()
    values, _ = engine["simulate_frontier"](
        int(trial_index),
        np.asarray(levels, dtype=np.int16),
        a_max=int(np.max(levels)),
        b_max=int(b_max),
        base_seed=BASE_SEED,
    )
    return np.asarray(values, dtype=np.int16)


def atomic_save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_block(name, values, done, elapsed) -> None:
    block = BLOCKS[name]
    atomic_save(
        block["cache"],
        first_b=values,
        done=done,
        a_levels=block["levels"],
        trial_indices=block["indices"],
        completed_trials=int(np.count_nonzero(done)),
        requested_trials=len(done),
        elapsed_seconds=float(elapsed),
        base_seed=BASE_SEED,
        q3_stream_size=707,
        b_max=int(block["b_max"]),
        coupling=np.asarray(
            "SeedSequence([20260808, trial]); generate 707 A then use prefix"
        ),
        python=np.asarray(platform.python_version()),
        numpy=np.asarray(np.__version__),
        scipy=np.asarray(scipy.__version__),
    )


def run_block(name: str, critical: dict[int, int]):
    block = BLOCKS[name]
    levels = block["levels"]
    indices = block["indices"]
    b_max = int(block["b_max"])
    shape = (len(indices), len(levels))

    if block["cache"].exists():
        with np.load(block["cache"], allow_pickle=False) as data:
            values = data["first_b"].astype(np.int16)
            done = data["done"].astype(bool)
            elapsed = float(data["elapsed_seconds"])
            assert values.shape == shape
            assert np.array_equal(data["a_levels"], levels)
            assert np.array_equal(data["trial_indices"], indices)
            assert int(data["base_seed"]) == BASE_SEED
            assert int(data["b_max"]) == b_max
    else:
        values = np.full(shape, -1, dtype=np.int16)
        done = np.zeros(len(indices), dtype=bool)
        elapsed = 0.0
        for row, trial_index in enumerate(indices):
            known = critical[int(trial_index)] <= levels
            values[row, known] = 0
            if bool(known[0]):
                done[row] = True
        save_block(name, values, done, elapsed)

    pending = np.flatnonzero(~done)
    print(
        f"block={name}; trials={len(indices)}; pending={len(pending)}; "
        f"A={int(levels[0])}..{int(levels[-1])}; Bmax={b_max}",
        flush=True,
    )
    for start in range(0, len(pending), BATCH_SIZE):
        rows = pending[start : start + BATCH_SIZE]
        started = time.perf_counter()
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(simulate_one)(int(indices[row]), levels, b_max) for row in rows
        )
        elapsed += time.perf_counter() - started
        for row, result in zip(rows, outputs):
            known = critical[int(indices[row])] <= levels
            result[known] = 0
            values[row] = result
            done[row] = True
        save_block(name, values, done, elapsed)
        print(
            f"block={name}; completed={int(np.count_nonzero(done))}/{len(done)}; "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    return values, levels, b_max, elapsed


def build_result(block_results) -> None:
    with np.load(ORIGINAL_FRONTIER, allow_pickle=False) as data:
        original_values = data["first_b"].astype(np.int16)
        original_levels = data["a_levels"].astype(np.int16)
        original_b_max = data["b_max_by_level"].astype(np.int16)
        assert int(data["completed_trials"]) == 10_000
        assert np.array_equal(original_levels, np.arange(550, 610))

    keep = original_levels <= 608
    original_extra, extra_levels, extra_b_max, _ = block_results["original_extra"]
    first_10k = np.column_stack([original_values[:, keep], original_extra])
    first_10k_levels = np.concatenate([original_levels[keep], extra_levels])
    first_10k_bmax = np.concatenate(
        [original_b_max[keep], np.full(len(extra_levels), extra_b_max, dtype=np.int16)]
    )

    validation_low, low_levels, low_b_max, _ = block_results["validation_low"]
    validation_high, high_levels, high_b_max, _ = block_results["validation_high"]
    validation = np.column_stack([validation_low, validation_high])
    validation_levels = np.concatenate([low_levels, high_levels])
    validation_bmax = np.concatenate(
        [
            np.full(len(low_levels), low_b_max, dtype=np.int16),
            np.full(len(high_levels), high_b_max, dtype=np.int16),
        ]
    )

    assert np.array_equal(first_10k_levels, np.arange(550, 612))
    assert np.array_equal(validation_levels, first_10k_levels)
    assert np.array_equal(validation_bmax, first_10k_bmax)
    first_b = np.vstack([first_10k, validation])
    levels = first_10k_levels
    bmax = first_10k_bmax

    atomic_save(
        OUTPUT,
        first_b=first_b,
        a_levels=levels,
        b_max_by_level=bmax,
        completed_trials=12_500,
        trial_indices=np.arange(12_500, dtype=np.int32),
        base_seed=BASE_SEED,
        q3_stream_size=707,
        target_probability=0.9,
        model_id=np.asarray("q4-full-frontier-12500-A550-611-exact-gjk-v1"),
        coupling=np.asarray(
            "SeedSequence([20260808, trial]); generate 707 A then use prefix"
        ),
        python=np.asarray(platform.python_version()),
        numpy=np.asarray(np.__version__),
        scipy=np.asarray(scipy.__version__),
    )

    rank = math.ceil(0.9 * len(first_b)) - 1
    b90 = np.sort(first_b.astype(np.int64), axis=0)[rank]
    successes = np.asarray(
        [np.count_nonzero(first_b[:, col] <= b) for col, b in enumerate(b90)],
        dtype=np.int32,
    )
    frame = pd.DataFrame(
        {
            "N_A": levels.astype(int),
            "N_B_90": b90,
            "successes": successes,
            "p_hat": successes / len(first_b),
        }
    )
    frame["K"] = 567 * frame["N_A"] + 64 * frame["N_B_90"]
    frame["cost_yuan"] = np.pi * frame["K"] / 120000.0
    frame["censored"] = frame["N_B_90"].to_numpy() > bmax
    feasible = frame.loc[~frame["censored"]].sort_values("K", ignore_index=True)
    feasible.to_csv(FRONTIER_CSV, index=False, encoding="utf-8-sig")

    best = feasible.iloc[0]
    checks = []
    for col, level in enumerate(levels.astype(int)):
        b_limit = (int(best["K"]) - 1 - 567 * level) // 64
        if b_limit < 0:
            continue
        successes_below = int(np.count_nonzero(first_b[:, col] <= b_limit))
        checks.append((level, b_limit, successes_below, successes_below / len(first_b)))
    cheaper = pd.DataFrame(
        checks,
        columns=["N_A", "max_cheaper_N_B", "successes", "p_hat"],
    ).sort_values("p_hat", ascending=False, ignore_index=True)
    cheaper.to_csv(CHEAPER_CSV, index=False, encoding="utf-8-sig")

    print("\n12500-trial frontier, lowest-cost feasible points", flush=True)
    print(feasible.head(15).to_string(index=False), flush=True)
    print("\nBest", best.to_dict(), flush=True)
    print("\nHighest cheaper checks", flush=True)
    print(cheaper.head(15).to_string(index=False), flush=True)
    print(f"saved={OUTPUT}", flush=True)


def main() -> None:
    critical = load_critical_counts()
    results = {}
    for name in ("validation_low", "validation_high", "original_extra"):
        results[name] = run_block(name, critical)
    build_result(results)


if __name__ == "__main__":
    main()
