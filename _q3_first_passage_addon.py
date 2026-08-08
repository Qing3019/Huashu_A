SEARCH_LEVELS = np.arange(1, 708, dtype=int)
NO_HIT_SENTINEL = int(SEARCH_LEVELS[-1]) + 1


def simulate_first_passage(seed, max_cylinders=707, base_seed=20260808):
    """Return the first complete parent-cylinder count that connects the box.

    This is intentionally a thin wrapper around the Q2 nested-prefix engine.
    The sentinel ``max_cylinders + 1`` means that no connection was observed
    within the simulated upper bound; it is not an observed connection count.
    """
    max_cylinders = int(max_cylinders)
    levels = (
        SEARCH_LEVELS
        if max_cylinders == int(SEARCH_LEVELS[-1])
        else np.arange(1, max_cylinders + 1, dtype=int)
    )
    indicators, original_stats = simulate_nested_checked(
        int(seed), levels, int(base_seed)
    )
    hits = np.flatnonzero(indicators)
    first_passage = int(hits[0] + 1) if len(hits) else max_cylinders + 1
    reconstructed = levels >= first_passage
    if not np.array_equal(reconstructed, indicators):
        raise AssertionError("first-passage reconstruction violated monotonicity")
    stats = {"activated_cylinders": min(first_passage, max_cylinders)}
    stats.update(original_stats)
    return first_passage, stats


def simulate_first_passage_checked(seed, max_cylinders=707, base_seed=20260808):
    try:
        return simulate_first_passage(seed, max_cylinders, base_seed)
    except Exception as exc:
        raise RuntimeError(
            f"Monte Carlo first-passage trial {seed} failed"
        ) from exc


def run_first_passage_monte_carlo(
    trial_count,
    max_cylinders=707,
    base_seed=20260808,
    n_jobs=1,
    trial_start=0,
):
    """Run reproducible independent trials and return first-passage counts."""
    trial_indices = list(
        range(int(trial_start), int(trial_start) + int(trial_count))
    )
    started = time.perf_counter()
    if int(n_jobs) == 1:
        outputs = []
        report_every = max(1, int(trial_count) // 10)
        for local_index, trial_index in enumerate(trial_indices, start=1):
            outputs.append(
                simulate_first_passage_checked(
                    trial_index, max_cylinders, base_seed
                )
            )
            if local_index % report_every == 0 or local_index == trial_count:
                print(f"completed {local_index}/{trial_count} trials")
    else:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(simulate_first_passage_checked)(
                trial_index, max_cylinders, base_seed
            )
            for trial_index in trial_indices
        )

    first_passage = np.asarray([item[0] for item in outputs], dtype=np.int16)
    diagnostics = pd.DataFrame([item[1] for item in outputs])
    sentinel = int(max_cylinders) + 1
    if np.any((first_passage < 1) | (first_passage > sentinel)):
        raise AssertionError("invalid first-passage count")
    return first_passage, diagnostics, time.perf_counter() - started


def one_sided_wilson(successes, trials, z=1.6448536269514722):
    """Return marginal one-sided Wilson lower and upper bounds."""
    return wilson_interval(successes, trials, z=z)

