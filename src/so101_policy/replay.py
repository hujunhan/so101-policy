"""`so101-replay`: play one recorded episode's actions back on the follower, slowly, through SafeArm.

What it checks: the recorded actions drive the real arm along the demonstrated path (the units, the joint
order and the calibration are consistent), and how closely the arm repeats the recorded states.

How: the episode's actions (LeRobot degrees, gripper %) are converted to joint angles, slowed down by
`--speed` and resampled at SafeArm's command rate. SafeArm moves to the first pose at 10 deg/s, follows the
path with every command through the safety layer (limit margin, step cap, stall detection), holds the last
pose, then asks before releasing the torque. Poses outside the safety limits (the folded rest pose is one)
are clipped to the limits, so the start and the end of a replay sit a few degrees above the rest pose.

The arm moves. The user is at the arm with power in reach; the script asks for `yes` first.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from so101_policy.check_dataset import JOINTS, STILL_DEG, best_lag, load_frames, still_time
from so101_policy.record import load_settings

REPO = Path(__file__).resolve().parents[2]


def episode_arrays(root: Path, episode: int) -> tuple[float, np.ndarray, np.ndarray]:
    """(fps, action, state) of one episode, in LeRobot units."""
    info, data, _ = load_frames(root)
    d = data[data["episode_index"] == episode]
    if d.empty:
        raise ValueError(f"episode {episode} is not in {root}")
    action = np.stack(d["action"].to_numpy()).astype(float)
    state = np.stack(d["observation.state"].to_numpy()).astype(float)
    return float(info["fps"]), action, state


def resample(x: np.ndarray, fps: float, rate_hz: float, speed: float) -> np.ndarray:
    """Rows of x (recorded at fps) played at `speed` times real time, sampled at rate_hz (linear)."""
    if not 0 < speed <= 1:
        raise ValueError("speed must be in (0, 1]: replays are never faster than the demonstration")
    t_rec = np.arange(len(x)) / fps
    t_play = np.arange(0.0, t_rec[-1] / speed + 1e-9, 1.0 / rate_hz) * speed
    return np.stack([np.interp(t_play, t_rec, x[:, j]) for j in range(x.shape[1])], axis=1)


def active_span(state: np.ndarray, fps: float, pad_s: float = 0.5) -> tuple[int, int]:
    """Frames [i0, i1) without the still time at the start and the end (plus pad_s of it on each side)."""
    pad = int(round(pad_s * fps))
    i0 = max(int(round(still_time(state, fps) * fps)) - pad, 0)
    i1 = min(len(state) - int(round(still_time(state, fps, from_end=True) * fps)) + pad, len(state))
    return i0, max(i1, i0 + 2)


def predicted_lag_deg(path: np.ndarray, rate_hz: float, lag_s: float) -> np.ndarray:
    """Per joint, the largest expected |measured - commanded| (deg): commanded speed times the response lag.

    SafeArm reads the arm right after each command, so a joint moving at v deg/s reads about v * lag behind;
    the stall check fires when that exceeds its threshold for long enough, even though nothing is blocked.
    """
    v = np.abs(np.diff(path, axis=0)) * rate_hz
    return np.degrees(v.max(axis=0) * lag_s)


def to_rad(rows: np.ndarray, jm) -> np.ndarray:
    from so101_core.convert import lerobot_to_rad

    return np.array([lerobot_to_rad(dict(zip(JOINTS, r, strict=True)), jm) for r in rows])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=str(REPO / "data" / "local__so101_s2_dry"), help="dataset dir")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--speed", type=float, default=0.25, help="fraction of the recorded speed (default 0.25)")
    p.add_argument("--keep-still", action="store_true", help="keep the still time at the start and the end")
    p.add_argument("--dry", action="store_true", help="plan only: no connection to the arm")
    args = p.parse_args(argv)

    from so101_core.config import DeviceConfig, JointMap
    from so101_core.model import urdf_limits
    from so101_core.safety import SafetyConfig, SafetyLayer

    from so101_policy import runlog

    log = runlog.start("replay", argv)
    print(f"Log: {log}")

    root = Path(args.root)
    fps, action, state = episode_arrays(root, args.episode)
    lag_frames, _ = best_lag(action, state)  # the follower's response lag in this recording
    i0, i1 = (0, len(action)) if args.keep_still else active_span(state, fps)
    action, state = action[i0:i1], state[i0:i1]
    jm = JointMap.load()
    safety_cfg = SafetyConfig()
    stall_deg = 5.0  # SafeArm.follow's default stall threshold
    path = to_rad(resample(action, fps, safety_cfg.rate_hz, args.speed), jm)

    # Clip to the safety limits up front (SafeArm would clip each command anyway). follow() needs the path to
    # start where move_to ends, and move_to ends at the clipped start pose. Then preview the step cap.
    layer = SafetyLayer(safety_cfg, limits=jm.model_limits(*urdf_limits()))
    clipped = np.array([layer.clip(q)[0] for q in path])
    steps = np.abs(np.diff(clipped, axis=0)).max(axis=1)
    n_fast = int((steps > safety_cfg.max_step_rad + 1e-9).sum())
    p_gain = load_settings()["follower"]["p_gain"]
    clip_deg = np.degrees(np.abs(clipped - path))
    lag_deg = predicted_lag_deg(clipped, safety_cfg.rate_hz, lag_frames / fps)
    print(
        f"Episode {args.episode}: frames {i0}-{i1} of the recording ({(i1 - i0) / fps:.1f} s"
        f"{'' if args.keep_still else f', still time > {STILL_DEG:g} deg trimmed'}),"
        f" replay {len(path) / safety_cfg.rate_hz:.1f} s at {args.speed:g}x, servo P = {p_gain}."
    )
    cap = np.degrees(safety_cfg.max_step_rad)
    print(f"Safety preview: {n_fast} commands faster than the step cap ({cap:.0f} deg per command);")
    print("clipped to the limits (at the rest pose, at the stops):")
    for j, name in enumerate(JOINTS):
        n = int((clip_deg[:, j] > 1e-6).sum())
        if n:
            print(f"  {name:14s} {n:5d} commands, up to {clip_deg[:, j].max():.1f} deg")
    worst = int(np.argmax(lag_deg[:5]))
    print(
        f"Expected lag behind the command (response lag {1000 * lag_frames / fps:.0f} ms x peak speed):"
        f" up to {lag_deg[worst]:.1f} deg ({JOINTS[worst]}); the stall check stops at {stall_deg:g} deg."
    )
    if lag_deg[:5].max() > 0.6 * stall_deg:
        slower = args.speed * 0.6 * stall_deg / lag_deg[:5].max()
        print(
            f"Too fast for the stall check: the arm would stop with a false stall. Use --speed {slower:.2f}."
        )
        return
    if args.dry:
        return

    from so101_core.arm import SafeArm

    print(
        "\nThe follower will move: first to the episode's start pose at 10 deg/s, then along the"
        " recorded path."
        "\nStand at the arm with the power switch in reach. Clear the workspace (or put the object where it"
        "\nwas at the start of the episode)."
    )
    print("Type yes to start: ", end="", flush=True)  # printed, so the prompt is in the log too
    if input().strip() != "yes":
        print("Not started.")
        return

    out_dir = REPO / "outputs" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{root.name}_ep{args.episode}_{datetime.now():%Y%m%d_%H%M%S}.npz"
    ticks: list[tuple[int, np.ndarray, np.ndarray]] = []  # kept even if follow() stops with an error
    meta = {
        "root": str(root),
        "episode": args.episode,
        "frames": [i0, i1],
        "speed": args.speed,
        "p_gain": p_gain,
    }

    def save(status: str) -> None:
        q_meas = np.array([m for _, m, _ in ticks]).reshape(-1, 6)
        q_cmd = np.array([c for _, _, c in ticks]).reshape(-1, 6)
        np.savez_compressed(
            out, q_meas=q_meas, q_cmd=q_cmd, path=path, path_clipped=clipped,
            meta=json.dumps({**meta, "status": status}),
        )  # fmt: skip

    with SafeArm(DeviceConfig.load(), safety=safety_cfg, joint_map=jm, position_p=p_gain) as arm:
        arm.connect(torque=True)
        print("Moving to the start pose...")
        arm.move_to(clipped[0], speed_rad_s=np.radians(10.0))
        print("Following the path...")
        try:
            res = arm.follow(clipped, on_tick=lambda i, q, c: ticks.append((i, q, c)))
        except BaseException as e:
            save(f"stopped: {type(e).__name__}: {e}")
            print(f"Saved what was recorded so far: {out}")
            raise
        save("complete")
        print(f"Done. The arm holds the last pose. Step-capped {res.step_limited} commands.")

    # How closely did the arm follow, and how close is it to what was recorded during the demonstration?
    rec_state = to_rad(resample(state, fps, safety_cfg.rate_hz, args.speed), jm)[: len(res.q_meas)]
    track = np.degrees(np.abs(res.q_meas - res.q_cmd))
    vs_demo = np.degrees(np.abs(res.q_meas - rec_state))
    print("\nPer joint (deg):          " + " ".join(f"{j[:8]:>8s}" for j in JOINTS))
    for name, err in (("|measured - commanded|", track), ("|measured - demo state|", vs_demo)):
        print(f"  {name:24s} median " + " ".join(f"{v:8.1f}" for v in np.median(err, 0)))
        print(f"  {'':24s}    max " + " ".join(f"{v:8.1f}" for v in err.max(0)))
    print("(gripper in rad of the model's gripper joint; it stops on the object in the demo but not here)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
