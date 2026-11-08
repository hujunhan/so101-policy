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

from so101_policy.check_dataset import JOINTS, load_frames
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


def to_rad(rows: np.ndarray, jm) -> np.ndarray:
    from so101_core.convert import lerobot_to_rad

    return np.array([lerobot_to_rad(dict(zip(JOINTS, r, strict=True)), jm) for r in rows])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=str(REPO / "data" / "local__so101_s2_dry"), help="dataset dir")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--speed", type=float, default=0.5, help="fraction of the recorded speed (default 0.5)")
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
    jm = JointMap.load()
    safety_cfg = SafetyConfig()
    path = to_rad(resample(action, fps, safety_cfg.rate_hz, args.speed), jm)

    # Clip to the safety limits up front (SafeArm would clip each command anyway). follow() needs the path to
    # start where move_to ends, and move_to ends at the clipped start pose. Then preview the step cap.
    layer = SafetyLayer(safety_cfg, limits=jm.model_limits(*urdf_limits()))
    clipped = np.array([layer.clip(q)[0] for q in path])
    steps = np.abs(np.diff(clipped, axis=0)).max(axis=1)
    n_fast = int((steps > safety_cfg.max_step_rad + 1e-9).sum())
    p_gain = load_settings()["follower"]["p_gain"]
    clip_deg = np.degrees(np.abs(clipped - path))
    print(
        f"Episode {args.episode}: {len(action) / fps:.1f} s recorded,"
        f" replay {len(path) / safety_cfg.rate_hz:.1f} s at {args.speed:g}x"
        f" ({len(path)} commands at {safety_cfg.rate_hz:g} Hz), servo P = {p_gain}."
    )
    cap = np.degrees(safety_cfg.max_step_rad)
    print(f"Safety preview: {n_fast} commands faster than the step cap ({cap:.0f} deg per command);")
    print("clipped to the limits (at the rest pose, at the stops):")
    for j, name in enumerate(JOINTS):
        n = int((clip_deg[:, j] > 1e-6).sum())
        if n:
            print(f"  {name:14s} {n:5d} commands, up to {clip_deg[:, j].max():.1f} deg")
    if args.dry:
        return

    from so101_core.arm import SafeArm

    print(
        "\nThe follower will move: first to the episode's start pose at 10 deg/s, then along the"
        " recorded path."
        "\nStand at the arm with the power switch in reach. Clear the workspace (or put the object where it"
        "\nwas at the start of the episode)."
    )
    if input("Type yes to start: ").strip() != "yes":
        print("Not started.")
        return

    out_dir = REPO / "outputs" / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{root.name}_ep{args.episode}_{datetime.now():%Y%m%d_%H%M%S}.npz"
    with SafeArm(DeviceConfig.load(), safety=safety_cfg, joint_map=jm, position_p=p_gain) as arm:
        arm.connect(torque=True)
        arm.move_to(clipped[0], speed_rad_s=np.radians(10.0))
        res = arm.follow(clipped)
        print(f"Done. The arm holds the last pose. Step-capped {res.step_limited} commands.")
        np.savez_compressed(
            out,
            t=res.t,
            q_cmd=res.q_cmd,
            q_meas=res.q_meas,
            path=path,
            path_clipped=clipped,
            meta=json.dumps(
                {"root": str(root), "episode": args.episode, "speed": args.speed, "p_gain": p_gain}
            ),
        )

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
