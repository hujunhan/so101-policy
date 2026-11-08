"""`so101-check-dataset`: check a recorded LeRobot dataset before training on it.

Per episode: length; video frames per camera against data frames; how long the arm sits still at the start
and the end; start and end pose (did the demonstration return to the start pose?); the largest one-frame
jump in the action (a teleoperation glitch or a start-pose jump); how far the follower lags the leader
(frames, from the best shift between action and state) and the remaining tracking error.

`--sheet` also writes a contact sheet (first, middle and last frame of each camera per episode) to the
git-ignored captures/ directory: it shows the room, so it is not committed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STILL_DEG = 2.0  # "the arm moved" = some joint left its first (or last) value by more than this


@dataclass
class EpisodeReport:
    index: int
    frames: int
    seconds: float
    video_frames: dict[str, int] = field(default_factory=dict)
    still_start_s: float = 0.0
    still_end_s: float = 0.0
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    max_jump_deg: float = 0.0
    max_jump_joint: str = ""
    max_jump_t: float = 0.0
    lag_frames: int = 0
    track_err_deg: np.ndarray | None = None  # per joint, median |state(t+lag) - action(t)| while moving


def load_frames(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    info = json.loads((root / "meta" / "info.json").read_text())
    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet")))
    episodes = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("*/*.parquet")))
    return info, data.sort_values("index"), episodes.sort_values("episode_index")


def still_time(x: np.ndarray, fps: float, from_end: bool = False) -> float:
    """Seconds before any joint first leaves its starting value by STILL_DEG (or, from the end, after the
    last time it was farther than that from its final value)."""
    seq = x[::-1] if from_end else x
    moved = np.any(np.abs(seq - seq[0]) > STILL_DEG, axis=1)
    return (int(np.argmax(moved)) if moved.any() else len(seq)) / fps


def best_lag(action: np.ndarray, state: np.ndarray, max_lag: int = 10) -> tuple[int, np.ndarray]:
    """Shift k (frames) that minimizes |state(t+k) - action(t)| over the moving frames, and the per-joint
    median error at that shift. The follower reaches a commanded position some frames later (servo response
    plus the read-command cycle), so k > 0 is expected."""
    moving = np.any(np.abs(np.diff(action, axis=0)) > 0.2, axis=1)
    best = (0, np.inf, None)
    for k in range(0, min(max_lag, len(action) - 2) + 1):
        a = action[: len(action) - k] if k else action
        s = state[k:]
        m = moving[: len(a) - 1]
        if m.sum() < 10:
            continue
        err = np.abs(s[:-1][m] - a[:-1][m])
        score = float(np.median(err[:, :5]))  # the gripper saturates on the object; leave it out of the fit
        if score < best[1]:
            best = (k, score, np.median(err, axis=0))
    if best[2] is None:
        return 0, np.full(action.shape[1], np.nan)
    return best[0], best[2]


def count_video_frames(path: Path, t0: float, t1: float) -> int:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = 0
        for pkt in c.demux(s):
            if pkt.pts is None:
                continue
            t = float(pkt.pts * s.time_base)
            if t0 - 1e-3 <= t < t1 - 1e-3:
                n += 1
        return n


def video_path(root: Path, info: dict, key: str, ep: pd.Series) -> Path:
    chunk = int(ep[f"videos/{key}/chunk_index"])
    file = int(ep[f"videos/{key}/file_index"])
    return root / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)


def analyze(root: Path, count_video: bool = True) -> tuple[dict, list[EpisodeReport]]:
    info, data, episodes = load_frames(root)
    fps = float(info["fps"])
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    reports = []
    for _, ep in episodes.iterrows():
        i = int(ep["episode_index"])
        d = data[data["episode_index"] == i]
        a = np.stack(d["action"].to_numpy()).astype(float)
        s = np.stack(d["observation.state"].to_numpy()).astype(float)
        r = EpisodeReport(index=i, frames=len(d), seconds=len(d) / fps)
        if count_video:
            for key in video_keys:
                t0 = float(ep[f"videos/{key}/from_timestamp"])
                t1 = float(ep[f"videos/{key}/to_timestamp"])
                r.video_frames[key.removeprefix("observation.images.")] = count_video_frames(
                    video_path(root, info, key, ep), t0, t1
                )
        r.still_start_s = still_time(s, fps)
        r.still_end_s = still_time(s, fps, from_end=True)
        r.start, r.end = s[0], s[-1]
        if len(a) > 1:
            jumps = np.abs(np.diff(a, axis=0))
            t, j = np.unravel_index(int(np.argmax(jumps)), jumps.shape)
            r.max_jump_deg, r.max_jump_joint, r.max_jump_t = float(jumps[t, j]), JOINTS[j], (t + 1) / fps
        r.lag_frames, r.track_err_deg = best_lag(a, s)
        reports.append(r)
    return info, reports


def print_report(info: dict, reports: list[EpisodeReport]) -> None:
    fps = info["fps"]
    print(f"{len(reports)} episodes, {sum(r.frames for r in reports)} frames at {fps} fps\n")
    for r in reports:
        vids = ", ".join(f"{k} {n}" for k, n in r.video_frames.items())
        ok = all(n == r.frames for n in r.video_frames.values())
        print(f"episode {r.index}: {r.frames} frames ({r.seconds:.1f} s); video frames: {vids or '-'}"
              f"{'' if ok else '  <-- MISMATCH'}")  # fmt: skip
        print(f"  still at start {r.still_start_s:.1f} s, at end {r.still_end_s:.1f} s")
        print(
            f"  largest one-frame action jump: {r.max_jump_deg:.1f} deg"
            f" ({r.max_jump_joint}, t={r.max_jump_t:.2f} s)"
        )
        print(f"  follower lags the leader by {r.lag_frames} frames ({1000 * r.lag_frames / fps:.0f} ms); "
              "median error after the shift: " + " ".join(f"{e:.1f}" for e in r.track_err_deg))  # fmt: skip
    starts = np.array([r.start for r in reports])
    ends = np.array([r.end for r in reports])
    print("\nPose per joint (deg; gripper %):  " + " ".join(f"{j[:8]:>8s}" for j in JOINTS))
    for name, arr in (("start, mean", starts.mean(0)), ("start, spread (max-min)", np.ptp(starts, 0)),
                      ("end - start, max |.|", np.abs(ends - starts).max(0))):  # fmt: skip
        print(f"  {name:31s} " + " ".join(f"{v:8.1f}" for v in arr))


def contact_sheet(root: Path, info: dict, out: Path, width: int = 320) -> Path:
    """First, middle and last frame of each camera per episode, in one image."""
    import av
    import cv2

    _, _, episodes = load_frames(root)
    keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    rows = []
    for _, ep in episodes.iterrows():
        tiles = []
        for key in keys:
            t0 = float(ep[f"videos/{key}/from_timestamp"])
            t1 = float(ep[f"videos/{key}/to_timestamp"])
            dt = 1.0 / info["fps"]
            targets = [t0, (t0 + t1) / 2, t1 - dt]
            with av.open(str(video_path(root, info, key, ep))) as c:
                s = c.streams.video[0]
                for tt in targets:
                    c.seek(int(max(tt - 1.0, 0) / s.time_base), stream=s, backward=True)
                    img = None
                    for frame in c.decode(s):
                        img = frame.to_ndarray(format="bgr24")
                        if float(frame.pts * s.time_base) >= tt - dt / 2:
                            break
                    h = int(img.shape[0] * width / img.shape[1])
                    tile = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
                    cam = key.removeprefix("observation.images.")
                    label = f"ep{int(ep['episode_index'])} {cam} t={tt - t0:.1f}s"
                    cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    tiles.append(tile)
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 2, cv2.BORDER_CONSTANT) for t in tiles]
        rows.append(np.hstack(tiles))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=str(REPO / "data" / "local__so101_s2_dry"), help="dataset dir")
    p.add_argument("--sheet", action="store_true", help="write a contact sheet to captures/")
    args = p.parse_args(argv)
    root = Path(args.root)
    info, reports = analyze(root)
    print_report(info, reports)
    if args.sheet:
        out = contact_sheet(root, info, REPO / "captures" / "s2" / f"{root.name}_sheet.jpg")
        print(f"\nContact sheet: {out}")


if __name__ == "__main__":
    main()
