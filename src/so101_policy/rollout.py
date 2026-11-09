"""`so101-rollout`: run a trained policy on the arm through SafeArm, with the recording's cameras and gain.

Each tick (at the dataset's 30 Hz): read the latest camera frames and the joint state, give them to the policy
in the same form as the recorded dataset (LeRobot degrees, gripper %; RGB images with the recording's crop),
get one action (the policy keeps its own chunk queue), and send it through SafeArm's safety layer (limit
margin, step cap). A watch stops the run when the arm stays far from its command (blocked) for too long.
Everything is logged: per tick the state, the action, the command sent, the policy time; the camera streams
to video in the git-ignored captures/.

`--shadow`: the arm stays limp and is only read; the policy runs on the live cameras and its actions are
logged but not sent. Checks the plumbing and the timing with nothing moving.

The arm moves (without --shadow). The user is at the arm with power in reach; the script asks for `yes`.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from so101_policy.check_dataset import JOINTS
from so101_policy.record import apply_v4l2, camera_configs, load_settings

REPO = Path(__file__).resolve().parents[2]


def training_meta(checkpoint: Path):
    """Metadata of the dataset the policy was trained on (None if unknown or not available locally)."""
    cfg_path = checkpoint / "train_config.json"
    if not cfg_path.exists():
        return None
    ds = json.loads(cfg_path.read_text()).get("dataset", {})
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        return LeRobotDatasetMetadata(ds["repo_id"], root=ds.get("root"))
    except Exception:
        return None


def training_task(checkpoint: Path) -> str:
    """The language instruction the policy was trained with: the first task of its training dataset."""
    meta = training_meta(checkpoint)
    return "" if meta is None else str(meta.tasks.index[0])


class PolicyRunner:
    """A trained LeRobot policy fed with live observations in the dataset's format."""

    def __init__(self, checkpoint: Path, device: str = "cuda", camera_map: dict[str, str] | None = None):
        from so101_policy.offline_eval import load_policy

        self.policy, self.pre, self.post = load_policy(checkpoint, device)
        self.camera_map = camera_map or {}  # our camera name -> the policy's (for policies trained elsewhere)
        # The cameras of the training dataset, by their dataset names: the preprocessor renames them if the
        # policy calls them something else (SmolVLA: camera1, camera2; it pads the cameras it declares but
        # was not trained with). Without the dataset, the policy's own image inputs.
        meta = training_meta(checkpoint)
        self.image_keys = list(meta.camera_keys) if meta is not None else [
            k for k in self.policy.config.input_features if k.startswith("observation.images.")
        ]  # fmt: skip
        self.task = "" if meta is None else str(meta.tasks.index[0])

    def warm_up(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> float:
        """One call before the first tick (CUDA kernels load on the first call: ~0.7 s). Returns its ms."""
        t = time.perf_counter()
        self.act(state, images, task)
        self.reset()
        return (time.perf_counter() - t) * 1000

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> np.ndarray:
        """state: LeRobot units (6,); images: our camera name -> HxWx3 uint8 RGB. Returns one action (6,)."""
        import torch

        batch = {"observation.state": torch.from_numpy(state.astype(np.float32)).unsqueeze(0)}
        for name, img in images.items():
            key = f"observation.images.{self.camera_map.get(name, name)}"
            if key in self.image_keys or not self.image_keys:
                t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
                batch[key] = t.unsqueeze(0)
        missing = [k for k in self.image_keys if k not in batch]
        if missing:
            raise KeyError(f"The policy wants {missing}; map our cameras with --camera-map (e.g. zed=up).")
        batch["task"] = [task]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            obs = self.pre(batch)
            action = self.policy.select_action(obs)
        return self.post(action.float()).detach().cpu().numpy().reshape(-1)


@dataclass
class RunLog:
    t: list[float] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)  # LeRobot units
    action: list[np.ndarray] = field(default_factory=list)  # LeRobot units, as the policy gave it
    q_cmd: list[np.ndarray] = field(default_factory=list)  # rad, after the safety filter (nan in shadow)
    policy_ms: list[float] = field(default_factory=list)
    frame_age_ms: list[dict[str, float]] = field(default_factory=list)

    def save(self, path: Path, meta: dict) -> None:
        np.savez_compressed(
            path,
            t=np.array(self.t),
            state=np.array(self.state),
            action=np.array(self.action),
            q_cmd=np.array(self.q_cmd),
            policy_ms=np.array(self.policy_ms),
            meta=json.dumps(meta),
        )


def open_cameras(settings: dict, devices) -> dict:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    from so101_policy.zed_camera import ZedCamera, ZedCameraConfig

    for name, cam in settings["cameras"].items():
        for msg in apply_v4l2(devices.cameras[name], cam.get("v4l2", {})):
            print(f"WARNING {name}: {msg}")
    cams = {}
    for name, cfg in camera_configs(settings, devices.cameras).items():
        kw = {k: v for k, v in cfg.items() if k != "type"}
        cams[name] = (
            ZedCamera(ZedCameraConfig(**kw))
            if cfg["type"] == "zed"
            else OpenCVCamera(OpenCVCameraConfig(**kw))
        )
    for c in cams.values():
        c.connect()
    return cams


class VideoSink:
    """Writes each camera stream to an mp4 in a background thread, so the control loop does not wait."""

    def __init__(self, folder: Path, fps: float):
        import queue

        self.folder, self.fps = folder, fps
        self.q: queue.Queue = queue.Queue(maxsize=300)
        self.writers: dict = {}
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def put(self, frames: dict[str, np.ndarray]) -> None:
        try:
            self.q.put_nowait(frames)
        except Exception:
            pass  # a dropped video frame never stalls the arm

    def _loop(self) -> None:
        import cv2

        while True:
            frames = self.q.get()
            if frames is None:
                break
            for name, img in frames.items():
                w = self.writers.get(name)
                if w is None:
                    h, wd = img.shape[:2]
                    w = cv2.VideoWriter(
                        str(self.folder / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (wd, h)
                    )
                    self.writers[name] = w
                w.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.q.put(None)
        self.thread.join(timeout=30)
        for w in self.writers.values():
            w.release()


def run_episode(
    runner: PolicyRunner,
    cams: dict,
    arm,
    jm,
    fps: float,
    time_limit_s: float,
    task: str,
    shadow: bool,
    log: RunLog,
    video: VideoSink | None,
    stop: threading.Event,
    stall_deg: float = 15.0,
    stall_s: float = 0.5,
) -> str:
    """One episode. Returns how it ended: 'time_limit', 'stopped', or 'stall: <joint>'."""
    from so101_core.arm import Rate
    from so101_core.convert import lerobot_to_rad, rad_to_lerobot

    runner.reset()
    rate = Rate(fps)
    t0 = time.perf_counter()
    lag_since = None
    while True:
        now = time.perf_counter() - t0
        if now >= time_limit_s:
            return "time_limit"
        if stop.is_set():
            return "stopped"
        images, ages = {}, {}
        for name, cam in cams.items():
            images[name] = cam.read_latest(max_age_ms=500)
            ages[name] = (time.perf_counter() - cam.latest_timestamp) * 1000
        q = arm.read()
        state = np.array(list(rad_to_lerobot(q, jm).values()))
        t_pol = time.perf_counter()
        action = runner.act(state, images, task)
        pol_ms = (time.perf_counter() - t_pol) * 1000
        if shadow:
            q_sent = np.full(len(JOINTS), np.nan)
        else:
            q_sent = arm.command(lerobot_to_rad(dict(zip(JOINTS, action, strict=True)), jm))
            far = np.abs(q - q_sent)
            if np.max(far) > np.radians(stall_deg):
                lag_since = lag_since or time.perf_counter()
                if time.perf_counter() - lag_since > stall_s:
                    worst = JOINTS[int(np.argmax(far))]
                    arm.hold()
                    return f"stall: {worst}"
            else:
                lag_since = None
        log.t.append(now)
        log.state.append(state)
        log.action.append(action)
        log.q_cmd.append(q_sent)
        log.policy_ms.append(pol_ms)
        log.frame_age_ms.append(ages)
        if video is not None:
            video.put(images)
        rate.sleep()


def summarize(log: RunLog, fps: float) -> str:
    t = np.array(log.t)
    dt = np.diff(t) * 1000 if len(t) > 1 else np.array([0.0])
    pm = np.array(log.policy_ms)
    ages = {
        k: np.array([a[k] for a in log.frame_age_ms])
        for k in (log.frame_age_ms[0] if log.frame_age_ms else {})
    }
    lines = [
        f"{len(t)} ticks in {t[-1] if len(t) else 0:.1f} s; tick interval median {np.median(dt):.1f} ms,"
        f" 99th percentile {np.percentile(dt, 99):.1f} ms, max {dt.max():.1f} ms"
        f" (target {1000 / fps:.1f} ms)",
        f"policy time: median {np.median(pm):.1f} ms, max {pm.max():.1f} ms"
        f" ({int((pm > 1000 / fps).sum())} ticks over the budget: new chunks)",
    ]
    for k, a in ages.items():
        lines.append(f"frame age {k}: median {np.median(a):.0f} ms, max {a.max():.0f} ms")
    return "\n".join(lines)


def parse_map(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    return dict(pair.split("=", 1) for pair in text.split(","))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="…/checkpoints/<step>/pretrained_model")
    p.add_argument("--shadow", action="store_true", help="arm limp and read only; actions logged, not sent")
    p.add_argument("--time-limit", type=float, default=30.0, help="seconds")
    p.add_argument("--camera-map", default=None, help="our camera -> the policy's, e.g. zed=up,wrist=side")
    p.add_argument("--task", default=None, help="language instruction (default: the training task)")
    p.add_argument("--name", default=None, help="name of this run (default: a timestamp)")
    args = p.parse_args(argv)

    from so101_core.arm import SafeArm
    from so101_core.config import DeviceConfig, JointMap
    from so101_core.safety import SafetyConfig

    from so101_policy import runlog

    log_path = runlog.start("rollout", argv)
    print(f"Log: {log_path}")
    settings = load_settings()
    fps = float(settings["fps"])
    ro = settings.get("rollout", {})
    safety = SafetyConfig(max_step_rad=np.radians(ro.get("max_step_deg", 2.0)))
    p_gain = settings["follower"]["p_gain"]
    devices, jm = DeviceConfig.load(), JointMap.load()

    ckpt = Path(args.checkpoint)
    runner = PolicyRunner(ckpt, camera_map=parse_map(args.camera_map))
    task = args.task if args.task is not None else runner.task
    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "captures" / "rollout" / name
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"Policy {ckpt} ({runner.policy.config.type}); task {task!r};"
        f" {'SHADOW (arm limp)' if args.shadow else 'LIVE'};"
        f" time limit {args.time_limit:g} s; step cap {np.degrees(safety.max_step_rad):.1f} deg per tick"
        f" ({np.degrees(safety.max_step_rad) * fps:.0f} deg/s); servo P = {p_gain}."
    )

    cams = open_cameras(settings, devices)
    stop = threading.Event()
    listener = None
    try:
        from lerobot.utils.keyboard_input import create_key_listener

        listener = create_key_listener(
            lambda k: stop.set() if k in ("q", "esc") else None, controls_help="q = stop"
        )
    except Exception as e:
        print(f"(no keyboard listener: {e}; Ctrl+C stops)")

    log = RunLog()
    video = VideoSink(out, fps)
    meta = {
        "checkpoint": str(ckpt), "task": task, "shadow": args.shadow, "fps": fps, "p_gain": p_gain,
        "max_step_deg": float(np.degrees(safety.max_step_rad)), "camera_map": parse_map(args.camera_map),
    }  # fmt: skip
    arm = SafeArm(devices, safety=safety, joint_map=jm, position_p=p_gain)
    end = "error"
    try:
        with arm:
            arm.connect(torque=not args.shadow)
            if not args.shadow:
                print("\nThe arm will move under the policy's control. Stand at the arm, power in reach.")
                print("Type yes to start: ", end="", flush=True)
                if input().strip() != "yes":
                    print("Not started.")
                    return
            images = {n: c.read_latest(max_age_ms=500) for n, c in cams.items()}
            from so101_core.convert import rad_to_lerobot

            state0 = np.array(list(rad_to_lerobot(arm.read(), jm).values()))
            print(f"Policy warm-up: {runner.warm_up(state0, images, task):.0f} ms.")
            print("Running. q = stop.")
            end = run_episode(
                runner, cams, arm, jm, fps, args.time_limit, task, args.shadow, log, video, stop
            )
            print(f"Ended: {end}.")
            if not args.shadow:
                arm.hold()
                print("Holding. Support the arm when asked; torque goes off next.")
    finally:
        video.close()
        if listener is not None:
            listener.stop()
        for c in cams.values():
            c.disconnect()
        meta["end"] = end
        if log.t:
            log.save(out / "log.npz", meta)
            print(summarize(log, fps))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
