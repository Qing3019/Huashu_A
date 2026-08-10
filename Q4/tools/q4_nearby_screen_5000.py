from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

try:
    from q4_same_q3_a_stream_runner import load_engine
except ModuleNotFoundError:
    from Q4.tools.q4_same_q3_a_stream_runner import load_engine


BASE_SEED = 20260808
TRIAL_START = 10_000
TRIALS = 5_000
N_JOBS = min(10, os.cpu_count() or 1)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
Q3_CACHE = ROOT / "Q3" / "q3_first_passage_segment_10000_22500.npz"
Q4_608_5_CACHE = (
    ROOT
    / "Q4"
    / "validation"
    / "q4_candidate_608A_5B_seed20260808_trials10000_19999_q3screen.npz"
)
OUTPUT = (
    ROOT
    / "Q4"
    / "validation"
    / "q4_nearby_candidates_seed20260808_trials10000_14999.npz"
)
PRIOR_OUTPUT = (
    ROOT
    / "Q4"
    / "validation"
    / "q4_nearby_candidates_seed20260808_trials10000_12499.npz"
)


def simulate_607_14_chunk(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    engine = load_engine()
    values = np.empty(len(indices), dtype=np.int16)
    for position, trial_index in enumerate(indices):
        first_b, _ = engine["simulate_frontier"](
            int(trial_index),
            np.asarray([607], dtype=np.int16),
            a_max=607,
            b_max=14,
            base_seed=BASE_SEED,
        )
        values[position] = int(first_b[0])
    return np.asarray(indices, dtype=np.int32), values


def main() -> None:
    with np.load(Q3_CACHE, allow_pickle=False) as data:
        assert int(data["base_seed"]) == BASE_SEED
        q3_indices = data["trial_indices"].astype(np.int32)
        q3_first = data["first_passage"].astype(np.int16)
    lookup = {int(i): int(t) for i, t in zip(q3_indices, q3_first)}
    indices = np.arange(TRIAL_START, TRIAL_START + TRIALS, dtype=np.int32)
    a_first = np.asarray([lookup[int(i)] for i in indices], dtype=np.int16)

    with np.load(Q4_608_5_CACHE, allow_pickle=False) as data:
        assert int(data["validation_seed"]) == BASE_SEED
        assert int(data["trial_start"]) == TRIAL_START
        assert int(data["completed_trials"]) >= TRIALS
        first_b_608 = data["first_b"][:TRIALS].astype(np.int16)

    # A607 alone already conducting is an exact safe shortcut.  Only the
    # unresolved trials require the mixed A/B geometry calculation.
    first_b_607 = np.zeros(TRIALS, dtype=np.int16)
    known_607 = a_first <= 607
    reused_trials = 0
    if PRIOR_OUTPUT.exists():
        with np.load(PRIOR_OUTPUT, allow_pickle=False) as data:
            prior_indices = data["trial_indices"].astype(np.int32)
            prior_values = data["first_b_607_14"].astype(np.int16)
            assert int(data["base_seed"]) == BASE_SEED
        prior_positions = prior_indices.astype(np.int64) - TRIAL_START
        valid = (prior_positions >= 0) & (prior_positions < TRIALS)
        prior_positions = prior_positions[valid].astype(int)
        first_b_607[prior_positions] = prior_values[valid]
        known_607[prior_positions] = True
        reused_trials = len(prior_positions)

    unresolved_positions = np.flatnonzero((a_first > 607) & (~known_607))
    unresolved_indices = indices[unresolved_positions]
    started = time.perf_counter()
    if len(unresolved_indices):
        chunks = [
            chunk
            for chunk in np.array_split(unresolved_indices, N_JOBS)
            if len(chunk)
        ]
        outputs = Parallel(n_jobs=len(chunks), backend="loky")(
            delayed(simulate_607_14_chunk)(chunk) for chunk in chunks
        )
        value_lookup = {
            int(i): int(v)
            for output_indices, output_values in outputs
            for i, v in zip(output_indices, output_values)
        }
        first_b_607[unresolved_positions] = np.asarray(
            [value_lookup[int(i)] for i in unresolved_indices], dtype=np.int16
        )
    elapsed = time.perf_counter() - started

    candidates = np.asarray([[608, 5], [607, 14], [609, 0]], dtype=np.int16)
    conducts = np.column_stack(
        [first_b_608 <= 5, first_b_607 <= 14, a_first <= 609]
    )
    successes = np.count_nonzero(conducts, axis=0).astype(np.int32)
    p_hat = successes.astype(float) / TRIALS

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        trial_indices=indices,
        candidates=candidates,
        conducts=conducts,
        successes=successes,
        p_hat=p_hat,
        a_first_passage=a_first,
        first_b_608_5=first_b_608,
        first_b_607_14=first_b_607,
        unresolved_607_14=len(unresolved_indices),
        reused_607_14_trials=reused_trials,
        elapsed_seconds=elapsed,
        base_seed=BASE_SEED,
        random_stream_schema=np.asarray(
            "SeedSequence([20260808, trial_index]); trial_index=10000..12499"
        ),
    )

    print(
        f"trials={TRIALS}; reused first-stage trials={reused_trials}; "
        f"new 607A+14B full mixed trials={len(unresolved_indices)}"
    )
    for (n_a, n_b), success, probability in zip(candidates, successes, p_hat):
        print(
            f"({int(n_a)},{int(n_b)}): successes={int(success)}/{TRIALS}; "
            f"p_hat={probability:.6%}; "
            f"{'keep' if probability >= 0.9 else 'drop'}"
        )
    print(f"elapsed={elapsed:.1f}s")
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
