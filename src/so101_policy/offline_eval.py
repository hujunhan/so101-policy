"""`so101-offline-eval`: how well a trained policy predicts the demonstrator's actions on held-out episodes.

Open loop on recorded data: at frame t the policy sees the recorded observation (images and joint state) and
predicts a chunk of the next actions; these are compared with what the demonstrator actually commanded at
t, t+1, ... This checks training and the data pipeline without the arm. It does not measure task success:
on the arm the policy's own actions change what it sees next (closed loop), so small errors can grow or be
corrected.

Reported per joint (deg; gripper %): the mean absolute error by how far ahead in the chunk the action is,
and over the actions a deployed policy would execute (a new chunk every `n_action_steps` frames). Baseline:
"hold still", predicting the current joint state for every future action. A policy that learned anything
beats it clearly at longer horizons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JOINT_LABELS = ["pan", "lift", "elbow", "w_flex", "w_roll", "grip"]
HORIZONS = [1, 10, 25, 50, 100]


def load_policy(path: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, pre, post


def chunk_errors(pred: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
    """|pred[k] - actions[t + k]| for the k that exist in the episode: (K, joints)."""
    k = min(len(pred), len(actions) - t)
    return np.abs(pred[:k] - actions[t : t + k])


def episode_predictions(policy, pre, post, ds, task: str, stride: int, device: str):
    """Chunks predicted every `stride` frames of one episode. Returns (frames t, chunks (n, K, joints))."""
    import torch

    feats = list(policy.config.input_features)
    ts, chunks = [], []
    for t in range(0, len(ds), stride):
        item = ds[t]
        batch = {k: item[k].unsqueeze(0) for k in feats}
        batch["task"] = [task]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            obs = pre(batch)
            chunk = policy.predict_action_chunk(obs)  # (1, K, A), normalized
        chunk = chunk.float()
        k = chunk.shape[1]
        out = post(chunk.reshape(k, -1))  # unnormalize; one row per action
        chunks.append(out.detach().cpu().numpy().reshape(k, -1))
        ts.append(t)
    return np.array(ts), np.stack(chunks)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="…/checkpoints/<step>/pretrained_model")
    p.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    p.add_argument("--root", default=None, help="local dataset directory (default: the Hugging Face cache)")
    p.add_argument("--episodes", default="45-49", help="held-out episodes, e.g. 45-49 or 3,7,9")
    p.add_argument("--stride", type=int, default=5, help="predict a chunk every N frames")
    p.add_argument("--plot-episode", type=int, default=None, help="episode to plot (default: the first)")
    args = p.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_policy.check_dataset import load_frames

    if "-" in args.episodes:
        a, b = map(int, args.episodes.split("-"))
        episodes = list(range(a, b + 1))
    else:
        episodes = [int(e) for e in args.episodes.split(",")]
    ckpt = Path(args.checkpoint)
    device = "cuda"
    policy, pre, post = load_policy(ckpt, device)
    n_exec = int(getattr(policy.config, "n_action_steps", 1))

    meta_ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[episodes[0]])
    _, data, _ = load_frames(Path(meta_ds.root))
    task = meta_ds.meta.tasks.index[0] if hasattr(meta_ds.meta.tasks, "index") else ""

    by_h = {h: [] for h in HORIZONS}
    base_h = {h: [] for h in HORIZONS}
    executed, executed_base = [], []
    plot = None
    for ep in episodes:
        ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep])
        d = data[data["episode_index"] == ep]
        actions = np.stack(d["action"].to_numpy()).astype(float)
        states = np.stack(d["observation.state"].to_numpy()).astype(float)
        ts, chunks = episode_predictions(policy, pre, post, ds, task, args.stride, device)
        for t, c in zip(ts, chunks, strict=True):
            err = chunk_errors(c, actions, t)
            base = chunk_errors(np.repeat(states[t : t + 1], len(c), axis=0), actions, t)
            for h in HORIZONS:
                if h <= len(err):
                    by_h[h].append(err[h - 1])
                    base_h[h].append(base[h - 1])
        # what a deployed policy executes: a new chunk every n_exec frames (open loop on recorded frames)
        seq = np.full_like(actions, np.nan)
        for t, c in zip(ts, chunks, strict=True):
            if t % n_exec == 0:
                seq[t : t + n_exec] = c[: len(seq[t : t + n_exec])]
        ok = ~np.isnan(seq[:, 0])
        executed.append(np.abs(seq[ok] - actions[ok]))
        hold = np.vstack([np.repeat(states[t : t + 1], min(n_exec, len(actions) - t), axis=0)
                          for t in range(0, len(actions), n_exec)])  # fmt: skip
        executed_base.append(np.abs(hold - actions))
        if plot is None or ep == args.plot_episode:
            plot = (ep, actions, states, seq)

    print(f"Checkpoint {ckpt}")
    print(f"Held-out episodes {episodes}; a chunk every {args.stride} frames; executes {n_exec} per chunk.\n")
    print("Mean |predicted - demonstrated| action (deg; gripper %), policy / hold-still baseline:")
    print(f"  {'ahead':>9s} " + " ".join(f"{j:>13s}" for j in JOINT_LABELS))
    rows = {}
    for h in HORIZONS:
        if by_h[h]:
            m, b = np.mean(by_h[h], 0), np.mean(base_h[h], 0)
            rows[h] = {"policy": m.round(2).tolist(), "hold": b.round(2).tolist()}
            print(f"  {h:>3d} frame " + " ".join(f"{x:6.1f}/{y:6.1f}" for x, y in zip(m, b, strict=True)))
    ex, exb = np.vstack(executed).mean(0), np.vstack(executed_base).mean(0)
    print("  executed  " + " ".join(f"{x:6.1f}/{y:6.1f}" for x, y in zip(ex, exb, strict=True)))

    out_dir = REPO / "outputs" / "offline_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{ckpt.parents[2].name}_{ckpt.parent.name}"
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"checkpoint": str(ckpt), "episodes": episodes, "by_horizon": rows,
                    "executed": {"policy": ex.round(2).tolist(), "hold": exb.round(2).tolist()}}, indent=1)
    )  # fmt: skip
    fig_path = out_dir / f"{tag}_ep{plot[0]}.png"
    plot_episode(fig_path, *plot, n_exec)
    print(f"\nWrote {out_dir / (tag + '.json')} and {fig_path}")


def plot_episode(path: Path, ep: int, actions, states, seq, n_exec: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(actions)) / 30.0
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for j, ax in enumerate(axes.T.ravel()):
        ax.plot(t, actions[:, j], color="#1f77b4", lw=1.6, label="demonstrated action")
        ax.plot(t, seq[:, j], color="#d62728", lw=1.2, ls="--", label=f"policy (new chunk every {n_exec})")
        ax.plot(t, states[:, j], color="#7f7f7f", lw=0.9, alpha=0.7, label="joint state")
        for c in np.arange(0, len(actions), n_exec) / 30.0:
            ax.axvline(c, color="#cccccc", lw=0.6, zorder=0)
        ax.set_title(JOINT_LABELS[j], fontsize=10)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Held-out episode {ep}: open-loop prediction on recorded frames (deg; gripper %)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
