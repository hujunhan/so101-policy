"""The pre-registered evaluation protocol in code (docs/PROTOCOL.md): initial states, statistics, failures.

Initial states are drawn in normalized region coordinates (u, v in [0, 1]) so the list can be fixed before the
workspace is: u runs along the region's x side and v along its y side, both in the arm's base frame. The
region's corners in millimetres are added to the protocol before the first demonstration is recorded.

    uv run python -m so101_policy.protocol   # (re)writes protocol/eval_states.csv; the output never changes
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

EVAL_SEED = 20261108
N_TRIALS = 30
YAW_RANGE_DEG = 90.0  # drawn for a cube; unused for the tape roll (PROTOCOL amendment 1)
EVAL_STATES_CSV = Path(__file__).resolve().parents[2] / "protocol" / "eval_states.csv"

# First thing that went wrong, labelled from the trial video. Order follows the task.
FAILURES = {
    "grasp_miss": "gripper closed without the tape in it",
    "drop_in_transit": "tape grasped, then dropped before the bowl",
    "place_miss": "tape released but lands outside the bowl",
    "no_release": "tape not released, or left stuck on the bowl rim",
    "stuck": "freezes, oscillates, or runs out of time without one of the above",
    "collision": "hits the bowl, the table or itself hard enough to stop or move things; or a safety stop",
    "other": "anything else (described in the notes)",
}


@dataclass(frozen=True)
class InitialState:
    trial: int  # 1-based
    u: float  # [0, 1] along the region's x side
    v: float  # [0, 1] along the region's y side
    yaw_deg: float  # [0, 90)


def initial_states(n: int = N_TRIALS, seed: int = EVAL_SEED) -> list[InitialState]:
    """Uniform positions and yaws. The evaluation list uses the default seed; any other list (e.g. validation
    states for tuning) must use a different seed."""
    rng = np.random.default_rng(seed)
    uv = rng.uniform(0.0, 1.0, size=(n, 2))
    yaw = rng.uniform(0.0, YAW_RANGE_DEG, size=n)
    return [
        InitialState(i + 1, round(float(u), 3), round(float(v), 3), round(float(y), 1))
        for i, ((u, v), y) in enumerate(zip(uv, yaw, strict=True))
    ]


def write_states(states: list[InitialState], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "u", "v", "yaw_deg"])
        for s in states:
            w.writerow([s.trial, f"{s.u:.3f}", f"{s.v:.3f}", f"{s.yaw_deg:.1f}"])


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, n > 0 (got {successes}/{n})")
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Paired comparison of two policies on the same initial states. Returns (only_a, only_b, two-sided p):
    only the discordant trials carry information; under no difference each is a fair coin flip."""
    if len(a) != len(b):
        raise ValueError("both policies must run the same trials")
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    if only_a + only_b == 0:
        return 0, 0, 1.0
    return only_a, only_b, float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


if __name__ == "__main__":
    write_states(initial_states(), EVAL_STATES_CSV)
    print(f"wrote {EVAL_STATES_CSV}")
