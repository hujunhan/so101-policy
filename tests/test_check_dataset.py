"""Dataset check helpers on synthetic trajectories."""

import numpy as np

from so101_policy.check_dataset import best_lag, still_time


def test_still_time_from_start_and_end():
    x = np.zeros((90, 6))
    x[30:60, 1] = np.linspace(0, 50, 30)
    x[60:, 1] = 50
    assert still_time(x, 30.0) == 32 / 30  # frame 32 is the first more than 2 deg away (3.4 deg)
    assert still_time(x, 30.0, from_end=True) == 32 / 30  # frames 58..89 within 2 deg of 50; 57 is 46.6


def test_best_lag_recovers_a_known_delay():
    t = np.arange(300) / 30
    action = np.stack([20 * np.sin(2 * np.pi * 0.3 * t + k) for k in range(6)], axis=1)
    state = np.vstack([np.repeat(action[:1], 4, axis=0), action[:-4]]) + 0.5  # 4 frames late, 0.5 deg sag
    lag, err = best_lag(action, state)
    assert lag == 4
    assert np.allclose(err, 0.5, atol=1e-6)
