from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

try:
    from q4_independent_validation_runner import load_q3_engine
except ModuleNotFoundError:
    from Q4.tools.q4_independent_validation_runner import load_q3_engine


BASE_SEED = 20260808
N_A = 609
TOTAL_TRIALS = 20_000
N_JOBS = min(20, os.cpu_count() or 1)

ROOT = Path(__file__).resolve().parents[2]
Q3_ORIGINAL = ROOT / "Q3" / "q3_first_passage_results.npz"
Q3_SEGMENT = ROOT / "Q3" / "q3_first_passage_segment_10000_22500.npz"
Q4_SCREEN = (
    ROOT
    / "Q4"
    / "validation"
    / "q4_candidate_608A_5B_seed20260808_trials10000_19999_q3screen.npz"
)
OUTPUT = (
    ROOT / "Q4" / "validation" / "q4_A609_seed20260808_trials0_19999.npz"
)


def simulate_chunk(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    engine = load_q3_engine()
    values = np.zeros(len(indices), dtype=bool)
    for position, trial_index in enumerate(indices):
        indicators, _ = engine.simulate_nested_checked(
            int(trial_index),
            n_levels=np.asarray([N_A], dtype=int),
            base_seed=BASE_SEED,
        )
        values[position] = bool(indicators[0])
    return np.asarray(indices, dtype=np.int32), values


def main() -> None:
    with np.load(Q3_ORIGINAL, allow_pickle=False) as data:
        assert int(data["base_seed"]) == BASE_SEED
        assert np.array_equal(data["trial_indices"], np.arange(10_000))
        original = data["first_passage"].astype(np.int16) <= N_A

    with np.load(Q3_SEGMENT, allow_pickle=False) as data:
        assert int(data["base_seed"]) == BASE_SEED
        segment_indices = data["trial_indices"].astype(np.int32)
        segment_first = data["first_passage"].astype(np.int16)
    assert np.array_equal(segment_indices, np.arange(10_000, 15_800))
    segment = segment_first <= N_A

    with np.load(Q4_SCREEN, allow_pickle=False) as data:
        assert int(data["validation_seed"]) == BASE_SEED
        assert int(data["trial_start"]) == 10_000
        assert int(data["completed_trials"]) == 10_000
        first_b_608 = data["first_b"].astype(np.int16)

    tail_indices = np.arange(15_800, 20_000, dtype=np.int32)
    tail = first_b_608[5_800:] == 0
    unresolved_positions = np.flatnonzero(~tail)
    unresolved_indices = tail_indices[unresolved_positions]

    started = time.perf_counter()
    if len(unresolved_indices):
        chunks = [
            chunk
            for chunk in np.array_split(unresolved_indices, N_JOBS)
            if len(chunk)
        ]
        outputs = Parallel(n_jobs=len(chunks), backend="loky")(
            delayed(simulate_chunk)(chunk) for chunk in chunks
        )
        lookup = {
            int(index): bool(value)
            for output_indices, output_values in outputs
            for index, value in zip(output_indices, output_values)
        }
        tail[unresolved_positions] = np.asarray(
            [lookup[int(index)] for index in unresolved_indices], dtype=bool
        )
    elapsed = time.perf_counter() - started

    conducts = np.concatenate([original, segment, tail])
    assert len(conducts) == TOTAL_TRIALS
    successes = int(np.count_nonzero(conducts))
    p_hat = successes / TOTAL_TRIALS

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        conducts=conducts,
        trial_indices=np.arange(TOTAL_TRIALS, dtype=np.int32),
        successes=successes,
        p_hat=p_hat,
        base_seed=BASE_SEED,
        n_a=N_A,
        n_b=0,
        unresolved_tail_trials=len(unresolved_indices),
        elapsed_seconds=elapsed,
        random_stream_schema=np.asarray(
            "SeedSequence([20260808, trial_index]); trial_index=0..19999"
        ),
    )
    print(f"unresolved tail trials computed={len(unresolved_indices)}")
    print(f"successes={successes}/{TOTAL_TRIALS}")
    print(f"p_hat={p_hat:.8%}")
    print(f"meets_90_percent={successes >= 18_000}")
    print(f"margin_from_18000={successes - 18_000}")
    print(f"elapsed={elapsed:.1f}s")
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
