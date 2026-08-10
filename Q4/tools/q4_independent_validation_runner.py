from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy.stats import beta

try:
    from q4_same_q3_a_stream_runner import load_engine
except ModuleNotFoundError:  # 支持从工作区根目录作为模块导入
    from Q4.tools.q4_same_q3_a_stream_runner import load_engine


VALIDATION_SEED = 20260808
TRIAL_START = 10_000
N_A = 608
N_B = 5
TRIALS = 10_000
BATCH_SIZE = 100
N_JOBS = min(10, os.cpu_count() or 1)

SCRIPT_DIR = Path(__file__).resolve().parent
Q4_DIR = SCRIPT_DIR.parent
VALIDATION_DIR = Q4_DIR / "validation"
CACHE_PATH = VALIDATION_DIR / "q4_candidate_608A_5B_seed20260808_trials10000_19999_q3screen.npz"
Q3_SEGMENT_CACHE = Q4_DIR.parent / "Q3" / "q3_first_passage_segment_10000_22500.npz"

MODEL_ID = "q4-independent-fixed-candidate-608A-5B-q3-safe-screen-exact-gjk-v3"

_Q3_CACHE_MAP = None
_Q3_ENGINE = None


def load_q3_cache_map() -> dict[int, int]:
    global _Q3_CACHE_MAP
    if _Q3_CACHE_MAP is None:
        mapping: dict[int, int] = {}
        if Q3_SEGMENT_CACHE.exists():
            with np.load(Q3_SEGMENT_CACHE, allow_pickle=False) as data:
                indices = data["trial_indices"].astype(np.int64)
                first_passage = data["first_passage"].astype(np.int16)
                assert int(data["base_seed"]) == VALIDATION_SEED
                assert len(indices) == len(first_passage)
                mapping = {
                    int(trial_index): int(first)
                    for trial_index, first in zip(indices, first_passage)
                }
        _Q3_CACHE_MAP = mapping
    return _Q3_CACHE_MAP


def load_q3_engine():
    global _Q3_ENGINE
    if _Q3_ENGINE is None:
        engine_path = Q4_DIR.parent / "Q3" / "_q3_engine_full.py"
        module_name = "q3_independent_validation_engine"
        spec = importlib.util.spec_from_file_location(module_name, engine_path)
        if spec is None or spec.loader is None:
            raise ImportError(engine_path)
        q3_engine = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = q3_engine
        spec.loader.exec_module(q3_engine)

        original_sampler = q3_engine.sample_cylinders

        def locked_707_prefix(n, rng):
            return original_sampler(707, rng)[: int(n)]

        q3_engine.sample_cylinders = locked_707_prefix
        _Q3_ENGINE = q3_engine
    return _Q3_ENGINE


def a_only_conducts(trial_index: int) -> tuple[bool, bool]:
    """Return A608 connectivity and whether it came from the existing Q3 cache."""
    cached = load_q3_cache_map().get(int(trial_index))
    if cached is not None:
        return cached <= N_A, True
    q3_engine = load_q3_engine()
    indicators, _ = q3_engine.simulate_nested_checked(
        int(trial_index),
        n_levels=np.asarray([N_A], dtype=int),
        base_seed=VALIDATION_SEED,
    )
    return bool(indicators[0]), False


def simulate_chunk(trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the exact Q3 A-only result as a safe screen, then run Q4 if needed."""
    engine = None
    first_b_values = []
    diagnostic_values = []
    columns = np.asarray(["q3_cache_hit", "q3_simulated", "q4_full_trial"], dtype=str)
    for trial_index in np.asarray(trial_indices, dtype=np.int64):
        a_conducts, cache_hit = a_only_conducts(int(trial_index))
        if a_conducts:
            first_b_value = 0
            q4_full_trial = 0
        else:
            if engine is None:
                engine = load_engine()
            first_b, _ = engine["simulate_frontier"](
                int(trial_index),
                np.asarray([N_A], dtype=np.int16),
                a_max=N_A,
                b_max=N_B,
                base_seed=VALIDATION_SEED,
            )
            first_b_value = int(first_b[0])
            q4_full_trial = 1
        first_b_values.append(first_b_value)
        diagnostic_values.append(
            np.asarray(
                [int(cache_hit), int(not cache_hit), q4_full_trial],
                dtype=np.int64,
            )
        )
    return (
        np.asarray(first_b_values, dtype=np.int16),
        np.asarray(columns, dtype=str),
        np.vstack(diagnostic_values).astype(np.int64),
    )


def q3_screen_chunk(trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Screen uncached trials with the locked 707-A prefix."""
    q3_engine = load_q3_engine()
    indices = np.asarray(trial_indices, dtype=np.int64)
    conducts = np.zeros(len(indices), dtype=bool)
    for position, trial_index in enumerate(indices):
        indicators, _ = q3_engine.simulate_nested_checked(
            int(trial_index),
            n_levels=np.asarray([N_A], dtype=int),
            base_seed=VALIDATION_SEED,
        )
        conducts[position] = bool(indicators[0])
    return indices, conducts


def q4_only_chunk(trial_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the full mixed A/B graph only for trials not conducting with A alone."""
    engine = load_engine()
    indices = np.asarray(trial_indices, dtype=np.int64)
    values = np.empty(len(indices), dtype=np.int16)
    for position, trial_index in enumerate(indices):
        first_b, _ = engine["simulate_frontier"](
            int(trial_index),
            np.asarray([N_A], dtype=np.int16),
            a_max=N_A,
            b_max=N_B,
            base_seed=VALIDATION_SEED,
        )
        values[position] = int(first_b[0])
    return indices, values


def simulate_balanced_batch(
    trial_indices: np.ndarray, jobs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance cached Q3 screens, new Q3 screens, and the expensive Q4 trials."""
    indices = np.asarray(trial_indices, dtype=np.int64)
    positions = {int(trial_index): pos for pos, trial_index in enumerate(indices)}
    cache_map = load_q3_cache_map()
    a_conducts = np.zeros(len(indices), dtype=bool)
    cache_hit = np.zeros(len(indices), dtype=bool)

    missing = []
    for position, trial_index in enumerate(indices):
        cached = cache_map.get(int(trial_index))
        if cached is None:
            missing.append(int(trial_index))
        else:
            cache_hit[position] = True
            a_conducts[position] = cached <= N_A

    if missing:
        q3_jobs = min(os.cpu_count() or jobs, 2 * jobs)
        chunks = [
            chunk
            for chunk in np.array_split(np.asarray(missing, dtype=np.int64), q3_jobs)
            if len(chunk)
        ]
        screened = Parallel(n_jobs=len(chunks), backend="loky")(
            delayed(q3_screen_chunk)(chunk) for chunk in chunks
        )
        for screened_indices, screened_values in screened:
            for trial_index, conducts in zip(screened_indices, screened_values):
                a_conducts[positions[int(trial_index)]] = bool(conducts)

    first_b = np.zeros(len(indices), dtype=np.int16)
    unresolved = indices[~a_conducts]
    if len(unresolved):
        chunks = [
            chunk
            for chunk in np.array_split(unresolved, jobs)
            if len(chunk)
        ]
        q4_outputs = Parallel(n_jobs=len(chunks), backend="loky")(
            delayed(q4_only_chunk)(chunk) for chunk in chunks
        )
        for q4_indices, q4_values in q4_outputs:
            for trial_index, value in zip(q4_indices, q4_values):
                first_b[positions[int(trial_index)]] = int(value)

    columns = np.asarray(["q3_cache_hit", "q3_simulated", "q4_full_trial"], dtype=str)
    diagnostics = np.column_stack(
        [cache_hit.astype(np.int64), (~cache_hit).astype(np.int64), (~a_conducts).astype(np.int64)]
    )
    return first_b, columns, diagnostics


def save_cache(
    first_b: np.ndarray,
    diagnostics: np.ndarray,
    diagnostic_columns: np.ndarray,
    elapsed_seconds: float,
) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_name(CACHE_PATH.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        first_b=np.asarray(first_b, dtype=np.int16),
        diagnostics=np.asarray(diagnostics, dtype=np.int64),
        diagnostic_columns=np.asarray(diagnostic_columns, dtype=str),
        completed_trials=len(first_b),
        requested_trials=TRIALS,
        elapsed_seconds=float(elapsed_seconds),
        validation_seed=VALIDATION_SEED,
        trial_start=TRIAL_START,
        n_a=N_A,
        n_b=N_B,
        model_id=np.asarray(MODEL_ID),
        random_stream_schema=np.asarray(
            "SeedSequence([20260808, trial_index]); trial_index=10000..19999; "
            "fixed candidate before validation"
        ),
        python=np.asarray(platform.python_version()),
        numpy=np.asarray(np.__version__),
        scipy=np.asarray(scipy.__version__),
    )
    temporary.replace(CACHE_PATH)


def load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(CACHE_PATH, allow_pickle=False) as data:
        assert int(data["requested_trials"]) == TRIALS
        assert int(data["validation_seed"]) == VALIDATION_SEED
        assert int(data["trial_start"]) == TRIAL_START
        assert int(data["n_a"]) == N_A
        assert int(data["n_b"]) == N_B
        assert str(data["model_id"]) == MODEL_ID
        assert str(data["python"]) == platform.python_version()
        assert str(data["numpy"]) == np.__version__
        assert str(data["scipy"]) == scipy.__version__
        first_b = data["first_b"].astype(np.int16)
        diagnostics = data["diagnostics"].astype(np.int64)
        columns = data["diagnostic_columns"].astype(str)
        elapsed = float(data["elapsed_seconds"])
        assert int(data["completed_trials"]) == len(first_b)
    return first_b, diagnostics, columns, elapsed


def summarize(first_b: np.ndarray) -> dict[str, float | int | bool | str]:
    successes = int(np.count_nonzero(first_b <= N_B))
    failures = int(len(first_b) - successes)
    p_hat = successes / len(first_b)
    cp_lower_one_sided = (
        0.0 if successes == 0 else float(beta.ppf(0.05, successes, failures + 1))
    )
    cp_lower_two_sided = (
        0.0 if successes == 0 else float(beta.ppf(0.025, successes, failures + 1))
    )
    cp_upper_two_sided = (
        1.0 if failures == 0 else float(beta.ppf(0.975, successes + 1, failures))
    )
    return {
        "N_A": N_A,
        "N_B": N_B,
        "trials": len(first_b),
        "successes": successes,
        "failures": failures,
        "p_hat": p_hat,
        "cp_lower_one_sided_95": cp_lower_one_sided,
        "cp_lower_two_sided_95": cp_lower_two_sided,
        "cp_upper_two_sided_95": cp_upper_two_sided,
        "certified_p_ge_0_9": bool(cp_lower_one_sided >= 0.9),
        "sha256": hashlib.sha256(first_b.tobytes()).hexdigest(),
    }


def run(mode: str, jobs: int, batch_size: int) -> dict[str, float | int | bool | str]:
    if CACHE_PATH.exists() and mode in {"auto", "resume", "cache"}:
        first_b, diagnostics, columns, elapsed_seconds = load_cache()
    elif mode == "cache":
        raise FileNotFoundError(CACHE_PATH)
    else:
        first_b = np.empty(0, dtype=np.int16)
        diagnostics = np.empty((0, 0), dtype=np.int64)
        columns = np.empty(0, dtype=str)
        elapsed_seconds = 0.0

    if mode == "production" and len(first_b):
        first_b = np.empty(0, dtype=np.int16)
        diagnostics = np.empty((0, 0), dtype=np.int64)
        columns = np.empty(0, dtype=str)
        elapsed_seconds = 0.0

    if mode != "cache":
        for completed_start in range(len(first_b), TRIALS, batch_size):
            completed_stop = min(TRIALS, completed_start + batch_size)
            start = TRIAL_START + completed_start
            stop = TRIAL_START + completed_stop
            started = time.perf_counter()
            batch_first_b, batch_columns, batch_diagnostics = simulate_balanced_batch(
                np.arange(start, stop, dtype=np.int64), jobs
            )
            elapsed_seconds += time.perf_counter() - started
            if len(columns):
                assert np.array_equal(columns, batch_columns)
            else:
                columns = batch_columns
                diagnostics = np.empty((0, len(columns)), dtype=np.int64)

            first_b = np.concatenate([first_b, batch_first_b])
            diagnostics = np.vstack([diagnostics, batch_diagnostics])
            save_cache(first_b, diagnostics, columns, elapsed_seconds)
            partial = summarize(first_b)
            print(
                f"completed {len(first_b)}/{TRIALS}; "
                f"successes={partial['successes']}; "
                f"p_hat={partial['p_hat']:.6f}; "
                f"elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )

    assert len(first_b) == TRIALS
    result = summarize(first_b)
    print("\nIndependent validation result")
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["auto", "resume", "production", "cache"], default="auto"
    )
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run(args.mode, args.jobs, args.batch_size)


if __name__ == "__main__":
    main()
