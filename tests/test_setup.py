"""Smoke tests: the pinned dependencies install and agree with each other."""

import numpy as np
import pytest


def test_m0_and_m2_are_importable_and_consistent():
    from so101_calib.budget import ray_plane
    from so101_calib.charuco import Intrinsics
    from so101_core.model import ARM_JOINTS

    assert len(ARM_JOINTS) == 5
    intr = Intrinsics((1280, 720), np.array([[535.0, 0, 640], [0, 535, 360], [0, 0, 1.0]]), np.zeros(5))
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking straight down from 0.5 m
    T[2, 3] = 0.5
    p = ray_plane(np.array([[640.0, 360.0]]), intr, T, 0.0)[0]
    assert np.allclose(p, [0, 0, 0], atol=1e-9)


def test_lerobot_with_the_hw_extra():
    lerobot = pytest.importorskip("lerobot", reason="install with `uv sync --extra hw`")
    assert lerobot.__name__ == "lerobot"
