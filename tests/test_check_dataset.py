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


def test_replay_resample_slows_down_and_keeps_the_ends():
    import pytest

    from so101_policy.replay import resample

    x = np.stack([np.linspace(0, 29, 30)] * 6, axis=1)  # 1 s at 30 fps
    y = resample(x, fps=30.0, rate_hz=30.0, speed=0.5)
    assert len(y) == 59  # 0 .. 1.933 s of play time covers the 0.967 s recording
    assert y[0, 0] == 0 and y[-1, 0] == 29 and np.allclose(np.diff(y[:, 0]), 0.5)
    with pytest.raises(ValueError):
        resample(x, 30.0, 30.0, speed=1.5)


def test_replay_trims_still_time_and_predicts_lag():
    from so101_policy.replay import active_span, predicted_lag_deg

    x = np.zeros((300, 6))
    x[100:200, 1] = np.linspace(0, 60, 100)
    x[200:, 1] = 60
    i0, i1 = active_span(x, fps=30.0, pad_s=0.5)
    assert (i0, i1) == (104 - 15, 300 - 104 + 15)  # 0.606 deg per frame: 4 frames to pass 2 deg
    path = np.radians(x)  # 60 deg over 100 commands at 30 Hz: 18 deg/s
    assert np.isclose(predicted_lag_deg(path, 30.0, 0.1)[1], 1.8, atol=0.05)


def test_offline_eval_chunk_errors_stop_at_the_episode_end():
    from so101_policy.offline_eval import chunk_errors

    actions = np.arange(20, dtype=float)[:, None] * np.ones((1, 6))
    pred = actions[15:20] + 1.0  # a 5-step chunk at t=15, off by 1
    assert np.allclose(chunk_errors(pred, actions, 15), 1.0)
    assert chunk_errors(np.zeros((10, 6)), actions, 15).shape == (5, 6)  # only 5 frames left
