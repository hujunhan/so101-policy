"""`so101-record`: LeRobot's record loop with this setup's devices, cameras and settings filled in.

It reads the device paths from so101-core's `devices.yaml`, the recording settings from
`protocol/recording.yaml`, applies the cameras' V4L2 controls (and checks them), registers the "zed" camera
type, then hands a complete command line to `lerobot-record`. Arguments after `--` go to LeRobot unchanged.

The arm moves: the follower copies the leader. The user is at the leader arm with power in reach.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from so101_policy.runlog import mask_serials

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / "protocol" / "recording.yaml"
TASK = "Put the blue box in the bowl."


def load_settings(path: Path = SETTINGS) -> dict:
    return yaml.safe_load(path.read_text())


def camera_configs(settings: dict, paths: dict[str, str]) -> dict[str, dict]:
    """LeRobot camera configs (as dicts) for the ZED and the wrist camera."""
    fps = settings["fps"]
    cams = settings["cameras"]
    out: dict[str, dict] = {}
    if "zed" in cams:
        z = cams["zed"]
        cfg = {
            "type": "zed",
            "index_or_path": paths["zed"],
            "fps": fps,
            "width": z["capture"][0],
            "height": z["capture"][1],
            "fourcc": z["fourcc"],
        }
        if z.get("crop") is not None:
            cfg["crop"] = list(z["crop"])
        if z.get("out_size") is not None:
            cfg["out_width"], cfg["out_height"] = z["out_size"]
        out["zed"] = cfg
    if "wrist" in cams:
        w = cams["wrist"]
        cfg = {
            "type": "opencv",
            "index_or_path": paths["wrist"],
            "fps": fps,
            "width": w["capture"][0],
            "height": w["capture"][1],
            "fourcc": w["fourcc"],
        }
        if w.get("out_size") is not None and list(w["out_size"]) != list(w["capture"]):
            cfg["type"] = "resized"
            cfg["out_width"], cfg["out_height"] = w["out_size"]
        out["wrist"] = cfg
    return out


def lerobot_args(settings: dict, devices, args: argparse.Namespace) -> list[str]:
    cams = camera_configs(settings, devices.cameras)
    f = settings["follower"]
    root = Path(args.root) if args.root else REPO / "data" / args.repo_id.replace("/", "__")
    return [
        "--robot.type=so101_follower",
        f"--robot.port={devices.follower_port}",
        f"--robot.id={devices.robot_id}",
        f"--robot.position_p_coefficient={f['p_gain']}",
        f"--robot.max_relative_target={f['max_relative_target']}",
        f"--robot.cameras={json.dumps(cams)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={devices.leader_port}",
        f"--teleop.id={devices.teleop_id}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={root}",
        f"--dataset.single_task={args.task}",
        f"--dataset.fps={settings['fps']}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.reset_time_s={args.reset_time}",
        f"--dataset.streaming_encoding={'true' if settings.get('streaming_encoding', True) else 'false'}",
        "--dataset.push_to_hub=false",
        f"--display_data={'true' if args.display else 'false'}",
        f"--resume={'true' if args.resume else 'false'}",
    ]


def pose_differences(leader: dict[str, float], follower: dict[str, float], max_deg: float, max_grip: float):
    """Joints where the leader and the follower differ by more than the limit: [(joint, leader, follower)].

    Readings are LeRobot's (degrees; the gripper in % of its range)."""
    out = []
    for joint, lv in leader.items():
        fv = follower[joint]
        limit = max_grip if joint == "gripper" else max_deg
        if abs(lv - fv) > limit:
            out.append((joint, lv, fv))
    return out


def sync_goal_keep_torque(bus) -> None:
    """Goal = present position on every motor; motors whose torque was off are switched off again."""
    torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=3)
    bus.sync_write("Goal_Position", raw, normalize=False)
    off = [m for m, on in torque.items() if not on]
    if off:
        bus.disable_torque(off)


def read_start_poses(devices) -> tuple[dict[str, float], dict[str, float]]:
    """Read both arms, set the follower's goal to its present position, and leave the torque as it was.

    LeRobot's follower connect() enables torque without syncing the goal first: a goal left over from an
    earlier session would make the arm jump (so101-core, M0 findings). Writing goal = present here removes
    that. The Feetech servos switch their torque on when a goal is written (checked on this arm 2026-11-09),
    so a follower that was limp is made limp again. Returns (leader, follower) readings.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    follower = SO101Follower(SO101FollowerConfig(port=devices.follower_port, id=devices.robot_id))
    leader = SO101Leader(SO101LeaderConfig(port=devices.leader_port, id=devices.teleop_id))
    readings = []
    for dev in (leader, follower):
        bus = dev.bus
        bus.connect()
        try:
            if not bus.is_calibrated:
                raise RuntimeError(f"{dev} calibration does not match its LeRobot calibration file.")
            readings.append(bus.sync_read("Present_Position", num_retry=3))
            if dev is follower:
                sync_goal_keep_torque(bus)
        finally:
            bus.disconnect(disable_torque=False)  # leave the torque as it was
    return readings[0], readings[1]


def apply_v4l2(device: str, controls: dict[str, int]) -> list[str]:
    """Set V4L2 controls and read them back. Returns a list of problems (empty if all applied)."""
    problems = []
    for name, value in controls.items():
        subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{name}={value}"], check=False, capture_output=True)
        res = subprocess.run(
            ["v4l2-ctl", "-d", device, "-C", name], check=False, capture_output=True, text=True
        )
        got = res.stdout.strip().rsplit(":", 1)[-1].strip()
        if res.returncode != 0 or got != str(value):
            problems.append(f"{name}: wanted {value}, got {got or res.stderr.strip()}")
    return problems


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="local/so101_s2_dry", help="dataset name (never pushed from here)")
    p.add_argument("--root", default=None, help="dataset directory (default: data/<repo-id>)")
    p.add_argument("--task", default=TASK)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--episode-time", type=float, default=60.0, help="seconds per episode (press n when done)")
    p.add_argument(
        "--reset-time", type=float, default=120.0,
        help="longest reset between episodes (s); press n to start the next episode sooner",
    )  # fmt: skip
    p.add_argument("--display", action="store_true", help="show both cameras and the joints live")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")
    p.add_argument("--dry", action="store_true", help="print the LeRobot command and exit")
    p.add_argument(
        "--max-start-diff", type=float, default=15.0,
        help="refuse to start if a joint of the leader and the follower differ by more (deg; gripper %%)",
    )  # fmt: skip
    args = p.parse_args(argv)

    from so101_core.config import DeviceConfig

    devices = DeviceConfig.load()
    settings = load_settings()
    cmd = lerobot_args(settings, devices, args) + extra

    if not args.dry:
        from so101_policy import runlog

        log = runlog.start("record", argv)
        print(f"Log: {log}")
    print("lerobot-record \\\n  " + " \\\n  ".join(mask_serials(a) for a in cmd))
    if args.dry:
        return

    for name, cam in settings["cameras"].items():
        problems = apply_v4l2(devices.cameras[name], cam.get("v4l2", {}))
        for msg in problems:
            print(f"WARNING {name}: {msg}")

    leader, follower = read_start_poses(devices)
    far = pose_differences(leader, follower, args.max_start_diff, args.max_start_diff)
    if far:
        print("\nThe leader is not in the follower's pose; the follower would jump. Move the leader:")
        for joint, lv, fv in far:
            print(f"  {joint:14s} leader {lv:7.1f}  follower {fv:7.1f}  -> move the leader by {fv - lv:+.1f}")
        print("Then run the command again.")
        sys.exit(1)
    print(f"Start pose check passed (every joint within {args.max_start_diff:g}).")

    from lerobot.scripts import lerobot_record

    import so101_policy.zed_camera  # noqa: F401  (registers camera type "zed")

    print(
        "\nThe follower will copy the leader, also during the reset phase. While an episode is saved the\n"
        "follower holds still; wait for 'Recording episode' before moving on.\n"
        "Keys in this terminal: n = end the episode (during the reset: start the next one), r = redo the\n"
        "episode, q = stop recording.\n"
        "Each episode: start pose -> pick -> place -> back to the start pose -> n.\n"
    )
    if args.display:
        # LeRobot's rerun viewer hung on this laptop: show our own OpenCV window instead.
        from so101_policy.preview import Preview, install

        install(lerobot_record, Preview())
        print("A window shows both cameras as recorded; the keys also work in that window.\n")
    sys.argv = ["lerobot-record", *cmd]
    lerobot_record.main()


if __name__ == "__main__":
    main()
