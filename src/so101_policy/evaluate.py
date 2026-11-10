"""`so101-eval`: run and report the pre-registered evaluation (docs/PROTOCOL.md) on the real arm.

    so101-eval run <checkpoint> --policy act --condition base     # trials, one row each in trials.csv
    so101-eval report results/act/base [results/smolvla/base]     # Wilson intervals; McNemar for two

`run` goes through `protocol/eval_states.csv` in order. Per trial: the arm moves to the start pose; the
operator places the object at the listed state and presses Enter; the policy runs until the time limit or
until the operator presses q (the task is finished, or the arm is back and still); the arm holds while the
operator judges the result (success, or the first failure label, or void with its cause) and adds notes; the
row is written at once, so a session can stop any time and resume later. Void trials are repeated at the end
of the list, keeping their number. Each trial's camera streams, joint states and actions go to the git-ignored
captures/eval/; the trial log goes to results/<policy>/<condition>/ (committed).

The arm moves under the policy's control through SafeArm. The operator stands at the arm with power in reach.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from so101_policy.protocol import EVAL_STATES_CSV, FAILURES, N_TRIALS, mcnemar_exact, wilson

REPO = Path(__file__).resolve().parents[2]
COLUMNS = ["trial", "success", "failure", "duration_s", "void", "notes"]


# -- the trial log ----------------------------------------------------------------------------------------
def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def final_rows(rows: list[dict]) -> dict[int, dict]:
    """The counted result of each trial: its last row that is not void (a void trial is repeated)."""
    out = {}
    for r in rows:
        if not r["void"]:
            out[int(r["trial"])] = r
    return out


def pending(trials: list[int], rows: list[dict]) -> list[int]:
    """Trials still to run, in list order: never run first, then those whose runs so far were all void."""
    done = final_rows(rows)
    tried = {int(r["trial"]) for r in rows}
    fresh = [t for t in trials if t not in tried]
    voided = [t for t in trials if t in tried and t not in done]
    return fresh + voided


def summary(rows: list[dict]) -> dict:
    done = final_rows(rows)
    n = len(done)
    succ = [r for r in done.values() if r["success"] == "1"]
    times = [float(r["duration_s"]) for r in succ]
    out = {
        "n": n,
        "successes": len(succ),
        "voids": sum(1 for r in rows if r["void"]),
        "median_time_s": float(np.median(times)) if times else None,
        "failures": dict(Counter(r["failure"] for r in done.values() if r["success"] != "1")),
    }
    if n:
        out["rate"] = len(succ) / n
        out["wilson95"] = wilson(len(succ), n)
    return out


def paired(rows_a: list[dict], rows_b: list[dict]) -> tuple[int, int, float, int]:
    """McNemar on the trials both policies completed. Returns (only_a, only_b, p, number of pairs)."""
    a, b = final_rows(rows_a), final_rows(rows_b)
    common = sorted(set(a) & set(b))
    sa = [a[t]["success"] == "1" for t in common]
    sb = [b[t]["success"] == "1" for t in common]
    only_a, only_b, p = mcnemar_exact(sa, sb)
    return only_a, only_b, p, len(common)


def format_summary(name: str, s: dict) -> str:
    if not s["n"]:
        return f"{name}: no completed trials"
    lo, hi = s["wilson95"]
    t = f"{s['median_time_s']:.1f} s" if s["median_time_s"] is not None else "-"
    fails = ", ".join(f"{k} {v}" for k, v in sorted(s["failures"].items(), key=lambda kv: -kv[1])) or "none"
    partial = "" if s["n"] == N_TRIALS else f"  (INCOMPLETE: {s['n']} of {N_TRIALS} trials)"
    return (
        f"{name}: {s['successes']}/{s['n']} = {100 * s['rate']:.0f} %"
        f" (95 % Wilson {100 * lo:.0f}-{100 * hi:.0f} %)"
        f"; voids {s['voids']}; median time to success {t}; failures: {fails}{partial}"
    )


# -- the protocol's initial states ------------------------------------------------------------------------
def eval_states() -> list[dict]:
    with EVAL_STATES_CSV.open(newline="") as f:
        return [{"trial": int(r["trial"]), "u": float(r["u"]), "v": float(r["v"]),
                 "yaw_deg": float(r["yaw_deg"])} for r in csv.DictReader(f)]  # fmt: skip


def placement_text(state: dict, region: dict | None, yaw: bool = True) -> str:
    """Where the object goes. With the region from PROTOCOL section 7 (recording.yaml `eval.region`: `origin`,
    `u_axis`, `v_axis` in mm, base frame), in mm; before that, in normalized coordinates."""
    if region:
        o, eu, ev = (np.asarray(region[k], float) for k in ("origin", "u_axis", "v_axis"))
        x, y = o + state["u"] * eu + state["v"] * ev
        uv = f"u {state['u']:.2f}, v {state['v']:.2f}"
        text = f"x {x:.0f} mm, y {y:.0f} mm ({uv})"
    else:
        text = f"u {state['u']:.2f}, v {state['v']:.2f} (region not fixed yet)"
    return text + (f", yaw {state['yaw_deg']:.0f} deg" if yaw else "")


def parse_trials(text: str | None) -> list[int]:
    if not text:
        return list(range(1, N_TRIALS + 1))
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# -- running ----------------------------------------------------------------------------------------------
def ask(prompt: str) -> str:
    print(prompt, end="", flush=True)  # printed, so the prompt is in the log too
    return input().strip()


def ask_result(ended: str, elapsed_s: float, time_limit_s: float) -> dict:
    labels = list(FAILURES)
    while True:
        a = ask("Result? s = success, f = failure, v = void: ").lower()
        if a == "s":
            if ended == "time_limit":
                print("It ran to the time limit; success means the task finished within it (PROTOCOL 1).")
            row = {
                "success": 1,
                "failure": "",
                "duration_s": f"{min(elapsed_s, time_limit_s):.1f}",
                "void": "",
            }
            break
        if a == "f":
            print("First thing that went wrong:")
            for i, k in enumerate(labels, 1):
                print(f"  {i} {k:16s} {FAILURES[k]}")
            hint = " (a safety stop counts as collision)" if ended.startswith("stall") else ""
            c = ask(f"Number{hint}: ")
            if c.isdigit() and 1 <= int(c) <= len(labels):
                row = {
                    "success": 0,
                    "failure": labels[int(c) - 1],
                    "duration_s": f"{time_limit_s:.1f}",
                    "void": "",
                }
                break
        if a == "v":
            cause = ask(
                "Cause of the void (a fault outside the policy: bus error, camera dropout, person in view): "
            )
            if cause:
                row = {"success": "", "failure": "", "duration_s": f"{elapsed_s:.1f}", "void": cause}
                break
        print("Again.")
    notes = ask("Notes (Enter for none): ")
    row["notes"] = notes if ended in ("time_limit", "stopped") else f"[ended: {ended}] {notes}".strip()
    return row


def check_session(folder: Path, meta: dict) -> None:
    """One checkpoint and one setup per results folder: refuse to mix them."""
    path = folder / "meta.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if k in old and old[k] != v}
        if diff:
            raise SystemExit(f"{folder} was started with other settings: {diff}. Use another --condition.")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=1) + "\n")


def run(args) -> None:
    from so101_core.arm import SafeArm
    from so101_core.config import DeviceConfig, JointMap
    from so101_core.convert import lerobot_to_rad, rad_to_lerobot
    from so101_core.safety import SafetyConfig

    from so101_policy import runlog
    from so101_policy.check_dataset import JOINTS
    from so101_policy.record import load_settings
    from so101_policy.rollout import (
        PolicyRunner,
        RunLog,
        VideoSink,
        open_cameras,
        parse_map,
        run_episode,
        stop_keys,
    )

    log_path = runlog.start("eval", None)
    print(f"Log: {log_path}")
    settings = load_settings()
    ev = settings.get("eval", {})
    time_limit = args.time_limit or ev.get("time_limit_s")
    if not time_limit:
        raise SystemExit("No time limit: PROTOCOL section 7 fixes it (recording.yaml eval.time_limit_s).")
    if args.time_limit and ev.get("time_limit_s") and args.time_limit != ev["time_limit_s"]:
        print(f"WARNING: --time-limit {args.time_limit} differs from the protocol's {ev['time_limit_s']} s.")
    fps = float(settings["fps"])
    ro = settings.get("rollout", {})
    safety = SafetyConfig(max_step_rad=np.radians(ro.get("max_step_deg", 2.0)))
    p_gain = settings["follower"]["p_gain"]
    devices, jm = DeviceConfig.load(), JointMap.load()
    if ev.get("start_pose_deg"):
        start = lerobot_to_rad(dict(zip(JOINTS, ev["start_pose_deg"], strict=True)), jm)
    elif jm.rest is not None:
        start = jm.rest
    else:
        raise SystemExit("No start pose: set recording.yaml eval.start_pose_deg (PROTOCOL section 7).")

    ckpt = Path(args.checkpoint).resolve()
    folder = REPO / "results" / args.policy / args.condition
    meta = {
        "checkpoint": str(ckpt.relative_to(REPO)) if ckpt.is_relative_to(REPO) else str(ckpt),
        "time_limit_s": float(time_limit), "fps": fps, "p_gain": p_gain,
        "max_step_deg": float(np.degrees(safety.max_step_rad)), "camera_map": parse_map(args.camera_map),
        "start_pose_deg": [round(float(x), 1) for x in rad_to_lerobot(start, jm).values()],
    }  # fmt: skip
    check_session(folder, meta)
    csv_path = folder / "trials.csv"
    states = {s["trial"]: s for s in eval_states()}
    todo = pending(parse_trials(args.trials), read_rows(csv_path))
    if not todo:
        print(format_summary(f"{args.policy}/{args.condition}", summary(read_rows(csv_path))))
        return

    runner = PolicyRunner(ckpt, camera_map=parse_map(args.camera_map))
    task = args.task if args.task is not None else runner.task
    print(
        f"Policy {meta['checkpoint']} ({runner.policy.config.type}); task {task!r}; results {folder};"
        f"\ntime limit {time_limit:g} s; step cap {meta['max_step_deg']:.1f} deg per tick;"
        f" servo P = {p_gain};"
        f"\n{len(todo)} trials to run: {todo}"
    )
    cams = open_cameras(settings, devices)
    stop = threading.Event()

    arm = SafeArm(devices, safety=safety, joint_map=jm, position_p=p_gain)
    try:
        with arm:
            arm.connect(torque=True)
            print("\nThe arm will move: to the start pose, then under the policy's control.")
            print("Stand at the arm with the power switch in reach; nobody in the camera views.")
            if ask("Type yes to start: ") != "yes":
                print("Not started.")
                return
            for k in todo:
                st = states[k]
                print(f"\n=== Trial {k} ===")
                print("Moving to the start pose...")
                arm.move_to(start, speed_rad_s=np.radians(20.0))
                print(f"Place the box: {placement_text(st, ev.get('region'), ev.get('use_yaw', True))}")
                a = ask("Enter = run, skip = leave this trial for later, quit = end the session: ").lower()
                if a == "quit":
                    break
                if a == "skip":
                    continue
                images = {n: c.read_latest(max_age_ms=500) for n, c in cams.items()}
                state0 = np.array(list(rad_to_lerobot(arm.read(), jm).values()))
                runner.warm_up(state0, images, task)
                out = (
                    REPO
                    / "captures"
                    / "eval"
                    / args.policy
                    / args.condition
                    / f"trial{k:02d}_{datetime.now():%H%M%S}"
                )
                out.mkdir(parents=True, exist_ok=True)
                log, video = RunLog(), VideoSink(out, fps)
                stop.clear()
                print("Running. q = end the trial (finished, or back and still for 5 s).")
                t0 = time.perf_counter()
                try:
                    with stop_keys(stop, "q = end the trial"):  # only while the policy runs
                        ended = run_episode(
                            runner, cams, arm, jm, fps, time_limit, task, False, log, video, stop
                        )
                except Exception as e:  # a bus or camera fault: hold, record, let the operator void it
                    ended = f"error: {type(e).__name__}: {e}"
                elapsed = time.perf_counter() - t0
                try:
                    arm.hold()
                except Exception as e:
                    print(f"Could not hold ({e}): switch the servo power off if the arm moves.")
                video.close()
                log.save(out / "log.npz", {**meta, "trial": k, "end": ended, "state": st})
                print(f"Ended: {ended} after {elapsed:.1f} s. The arm holds; judge before touching anything.")
                row = {"trial": k, **ask_result(ended, elapsed, time_limit)}
                append_row(csv_path, row)
                print(f"Wrote trial {k}: {row}")
                ask("Take the box out, clear the arm's way back, then Enter: ")
            print("\n" + format_summary(f"{args.policy}/{args.condition}", summary(read_rows(csv_path))))
            print("Holding. Support the arm when asked; torque goes off next.")
    finally:
        for c in cams.values():
            c.disconnect()


def report(args) -> None:
    rows = [read_rows(Path(f) / "trials.csv") for f in args.folders]
    out = {}
    for f, r in zip(args.folders, rows, strict=True):
        name = "/".join(Path(f).parts[-2:])
        s = summary(r)
        print(format_summary(name, s))
        out[name] = s
    if len(rows) == 2:
        only_a, only_b, p, n = paired(*rows)
        a, b = ("/".join(Path(f).parts[-2:]) for f in args.folders)
        print(f"Paired over {n} trials: only {a} succeeds in {only_a}, only {b} in {only_b};"
              f" exact McNemar p = {p:.3f}")  # fmt: skip
        out["mcnemar"] = {"only_a": only_a, "only_b": only_b, "p": p, "pairs": n}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run trials on the arm")
    r.add_argument("checkpoint", help="…/checkpoints/<step>/pretrained_model (the final one, PROTOCOL 3)")
    r.add_argument("--policy", required=True, help="name for results/, e.g. act or smolvla")
    r.add_argument("--condition", default="base", help="name for results/<policy>/, e.g. base")
    r.add_argument("--trials", default=None, help="subset, e.g. 1-5 or 3,7 (default: all 30)")
    r.add_argument("--time-limit", type=float, default=None, help="seconds (default: the protocol's)")
    r.add_argument("--camera-map", default=None, help="our camera -> the policy's, e.g. zed=up,wrist=side")
    r.add_argument("--task", default=None, help="language instruction (default: the training task)")
    r.set_defaults(func=run)
    q = sub.add_parser("report", help="success rates; McNemar for two folders")
    q.add_argument("folders", nargs="+", help="results/<policy>/<condition> (one or two)")
    q.add_argument("--json", default=None, help="also write the numbers to this file")
    q.set_defaults(func=report)
    args = p.parse_args(argv)
    if args.cmd == "report" and len(args.folders) > 2:
        p.error("report takes one or two folders")
    args.func(args)


if __name__ == "__main__":
    main()
