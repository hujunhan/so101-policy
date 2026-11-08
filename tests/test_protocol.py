"""The committed evaluation list must be exactly what the pre-registered seed produces."""

import csv

import pytest

from so101_policy.protocol import (
    EVAL_STATES_CSV,
    N_TRIALS,
    initial_states,
    mcnemar_exact,
    wilson,
)


def test_committed_list_matches_the_seed():
    with EVAL_STATES_CSV.open() as f:
        rows = list(csv.DictReader(f))
    states = initial_states()
    assert len(rows) == len(states) == N_TRIALS
    for r, s in zip(rows, states, strict=True):
        assert (int(r["trial"]), float(r["u"]), float(r["v"]), float(r["yaw_deg"])) == (
            s.trial,
            s.u,
            s.v,
            s.yaw_deg,
        )
        assert 0 <= s.u <= 1 and 0 <= s.v <= 1 and 0 <= s.yaw_deg < 90


def test_other_seed_gives_another_list():
    assert initial_states(seed=1) != initial_states()


def test_wilson_known_values():
    lo, hi = wilson(24, 30)
    assert lo == pytest.approx(0.627, abs=1e-3) and hi == pytest.approx(0.905, abs=1e-3)
    assert wilson(0, 30)[0] == 0.0 and wilson(30, 30)[1] == 1.0
    with pytest.raises(ValueError):
        wilson(31, 30)


def test_mcnemar_counts_only_discordant_trials():
    a = [True] * 20 + [True] * 8 + [False] * 2
    b = [True] * 20 + [False] * 8 + [True] * 2
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (8, 2)
    assert p == pytest.approx(0.109375)  # 2 * P(X <= 2), X ~ Bin(10, 0.5)
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)
