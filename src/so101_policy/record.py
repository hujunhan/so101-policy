"""`so101-record`: LeRobot's record loop with this setup's devices, cameras and settings filled in.

It reads the device paths from so101-core's `devices.yaml`, the recording settings from
`protocol/recording.yaml`, applies the cameras' V4L2 controls (and checks them), registers the "zed" camera
type, then hands a complete command line to `lerobot-record`. Arguments after `--` go to LeRobot unchanged.

The arm moves: the follower copies the leader. The user is at the leader arm with power in reach.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / "protocol" / "recording.yaml"
TASK = "Put the cube in the bowl."


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
        out["wrist"] = {
            "type": "opencv",
            "index_or_path": paths["wrist"],
            "fps": fps,
            "width": w["capture"][0],
            "height": w["capture"][1],
            "fourcc": w["fourcc"],
        }
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
        "--dataset.push_to_hub=false",
        f"--display_data={'true' if args.display else 'false'}",
        f"--resume={'true' if args.resume else 'false'}",
    ]


def mask_serials(text: str) -> str:
    """Hide adapter and camera serial numbers before printing (the repo and its logs are public)."""
    text = re.sub(r"(USB_Single_Serial_)[^-\s\"']+", r"\1<SERIAL>", text)
    return re.sub(r"(Camera_)[0-9A-Za-z]{6,}", r"\1<SERIAL>", text)


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
    p.add_argument("--episode-time", type=float, default=30.0, help="seconds per episode")
    p.add_argument("--reset-time", type=float, default=15.0, help="seconds to reset between episodes")
    p.add_argument("--display", action="store_true", help="show cameras and joints in rerun")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")
    p.add_argument("--dry", action="store_true", help="print the LeRobot command and exit")
    args = p.parse_args(argv)

    from so101_core.config import DeviceConfig

    devices = DeviceConfig.load()
    settings = load_settings()
    cmd = lerobot_args(settings, devices, args) + extra

    print("lerobot-record \\\n  " + " \\\n  ".join(mask_serials(a) for a in cmd))
    if args.dry:
        return

    for name, cam in settings["cameras"].items():
        problems = apply_v4l2(devices.cameras[name], cam.get("v4l2", {}))
        for msg in problems:
            print(f"WARNING {name}: {msg}")

    from lerobot.scripts import lerobot_record

    import so101_policy.zed_camera  # noqa: F401  (registers camera type "zed")

    print(
        "\nThe follower will copy the leader. Put the leader in the same pose as the follower first.\n"
        "Keys in this terminal: n = end the episode early, r = redo it, q = stop recording.\n"
    )
    sys.argv = ["lerobot-record", *cmd]
    lerobot_record.main()


if __name__ == "__main__":
    main()
