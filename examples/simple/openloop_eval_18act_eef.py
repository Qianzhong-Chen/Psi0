"""
Open-loop evaluation for Psi0 fine-tuned with 18-dim EEF actions.

Loads model from a training run directory, runs inference on the training
dataset, and plots GT vs predicted values (absolute + action chunk overlay).

Usage:
    python examples/simple/openloop_eval_18act_eef.py \
        --run-dir .runs/finetune/<run_folder> \
        --ckpt-step 30000 \
        --episode 0-5 \
        --stride 4 \
        --data-root ~/.cache/huggingface/lerobot

Action (18): L_eef_6d(6) + L_grip(1) + R_eef_6d(6) + R_grip(1) + vx + vy + vyaw + height
State  (15): L_eef_6d(6) + L_grip(1) + R_eef_6d(6) + R_grip(1) + height
"""

import argparse
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm

from psi.utils import parse_args_to_tyro_config, seed_everything
from psi.models.psi0 import Psi0Model
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig

ACTION_LABELS = [
    "L_x(m)", "L_y(m)", "L_z(m)", "L_roll(rad)", "L_pitch(rad)", "L_yaw(rad)", "L_grip",
    "R_x(m)", "R_y(m)", "R_z(m)", "R_roll(rad)", "R_pitch(rad)", "R_yaw(rad)", "R_grip",
    "vx", "vy", "vyaw", "height",
]

# Semantic groups for plotting
DIM_GROUPS = [
    ("Left Wrist XYZ", range(0, 3), "m"),
    ("Left Wrist Rot", range(3, 6), "rad"),
    ("Left Grip", [6], ""),
    ("Right Wrist XYZ", range(7, 10), "m"),
    ("Right Wrist Rot", range(10, 13), "rad"),
    ("Right Grip", [13], ""),
    ("Linear Velocity (vx, vy)", [14, 15], "m/s"),
    ("Angular Velocity (vyaw)", [16], "rad/s"),
    ("Height", [17], "m"),
]


def load_model_and_data(run_dir, ckpt_step, device="cuda:0", data_root=None):
    """Load Psi0 model and dataset from a training run directory."""
    run_dir = Path(run_dir)
    config_ = parse_args_to_tyro_config(run_dir / "argv.txt")
    conf = (run_dir / "run_config.json").open("r").read()
    launch_config = config_.model_validate_json(conf)

    seed_everything(launch_config.seed or 42)

    psi0 = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=device)
    psi0.to(device)
    psi0.eval()
    print("Model loaded successfully.")

    data_cfg: LerobotDataConfig = launch_config.data
    if data_root is not None:
        data_cfg.root_dir = data_root
        print(f"Overriding data root to: {data_root}")
    model_cfg: Psi0ModelConfig = launch_config.model
    maxmin = data_cfg.transform.field

    transform_kwargs = dict(vlm_processor=psi0.vlm_processor)
    dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)

    return psi0, dataset, maxmin, model_cfg, device


def eval_episode(psi0, dataset, maxmin, device, ep_idx, stride=4, num_inference_steps=10):
    """Run open-loop evaluation on a single episode. Returns GT, predictions, and errors."""
    raw_ds = dataset.raw_dataset
    start_idx = raw_ds.base_dataset.episode_data_index["from"][ep_idx].item()
    end_idx = raw_ds.base_dataset.episode_data_index["to"][ep_idx].item()
    ep_len = end_idx - start_idx

    print(f"\n--- Episode {ep_idx}: {ep_len} frames (idx {start_idx}-{end_idx}) ---")

    # Collect full GT timeline
    gt_full = []
    for t in range(ep_len):
        frame = dataset[start_idx + t]
        gt_action = frame["raw_actions"]  # (chunk_size, action_dim)
        gt_full.append(gt_action[0])  # first step GT
    gt_full = np.stack(gt_full, axis=0)  # (ep_len, action_dim)
    n_action_dim = gt_full.shape[1]

    l1_errors_first = []
    l1_errors_horizon = []
    pred_first_steps = []
    pred_chunks = []  # (frame_offset, pred_actions) for chunk overlay plot
    timings = []

    for i in tqdm(range(start_idx, end_idx, stride), desc=f"Ep {ep_idx}"):
        frame = dataset[i]
        images = frame["raw_images"]
        instruction = frame["instruction"]
        states = frame["states"]

        batch_images = [images]
        batch_instructions = [instruction]
        batch_states = torch.from_numpy(states).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            pred_actions = psi0.predict_action(
                observations=batch_images,
                states=batch_states,
                instructions=batch_instructions,
                num_inference_steps=num_inference_steps,
                traj2ds=None,
            )
        dt = time.time() - t0
        timings.append(dt * 1000)

        gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(device)
        pred_denormed = maxmin.denormalize(pred_actions)
        pred_np = pred_denormed.detach().cpu().numpy().reshape(-1, n_action_dim)

        frame_offset = i - start_idx

        # First-step error
        l1_first = np.abs(gt_full[frame_offset] - pred_np[0])
        l1_errors_first.append(l1_first)
        pred_first_steps.append(pred_np[0].copy())

        # Horizon-avg error
        gt_horizon = []
        for h in range(min(pred_np.shape[0], end_idx - i)):
            gt_horizon.append(gt_full[frame_offset + h])
        gt_horizon = np.array(gt_horizon)
        valid = gt_horizon.shape[0]
        l1_full = np.abs(gt_horizon - pred_np[:valid])
        l1_errors_horizon.append(np.mean(l1_full, axis=0))

        # Store full chunk for overlay plot
        pred_chunks.append((frame_offset, pred_np.copy()))

    l1_errors_first = np.array(l1_errors_first)
    l1_errors_horizon = np.array(l1_errors_horizon)
    pred_first_steps = np.array(pred_first_steps)

    print(f"Mean inference time: {np.mean(timings):.1f} ms")

    return {
        "gt_full": gt_full,
        "pred_first_steps": pred_first_steps,
        "pred_chunks": pred_chunks,
        "l1_first": l1_errors_first,
        "l1_horizon": l1_errors_horizon,
        "timings": timings,
        "stride": stride,
        "ep_len": ep_len,
    }


def print_summary(results):
    """Print per-dimension error summary."""
    l1_first = results["l1_first"]
    l1_horizon = results["l1_horizon"]
    mean_first = l1_first.mean(axis=0)
    mean_horizon = l1_horizon.mean(axis=0)

    print("\nPer-dimension L1 error (first step | horizon avg):")
    n_dims = min(len(ACTION_LABELS), mean_first.shape[0])
    for i in range(n_dims):
        print(f"  {ACTION_LABELS[i]:18s}: {mean_first[i]:.6f}  |  {mean_horizon[i]:.6f}")


def plot_results(results, ep_idx, save_dir):
    """Generate all plots: L1 bar, GT vs pred values, GT vs pred with action chunks."""
    import matplotlib.pyplot as plt

    gt_full = results["gt_full"]
    pred_first_steps = results["pred_first_steps"]
    pred_chunks = results["pred_chunks"]
    l1_first = results["l1_first"]
    l1_horizon = results["l1_horizon"]
    stride = results["stride"]
    ep_len = results["ep_len"]
    n_dims = min(len(ACTION_LABELS), gt_full.shape[1])
    mean_time = np.mean(results["timings"])

    # ---- Plot 1: L1 bar chart ----
    mean_first = l1_first.mean(axis=0)
    mean_horizon = l1_horizon.mean(axis=0)

    x = np.arange(n_dims)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width / 2, mean_first[:n_dims], width, label="First step", color="steelblue")
    bars2 = ax.bar(x + width / 2, mean_horizon[:n_dims], width, label="Horizon avg", color="coral")
    ax.set_ylabel("Mean L1 Error")
    ax.set_title(f"Open-Loop L1 Error | Episode {ep_idx} | Inference: {mean_time:.0f}ms")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS[:n_dims], rotation=45, ha="right")
    ax.legend()
    for bar, val in zip(bars1, mean_first):
        if val > 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, mean_horizon):
        if val > 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fname = save_dir / f"l1_bar_ep{ep_idx}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()

    # ---- Plot 2: GT vs Predicted (first-step) values over time ----
    sample_frames = np.arange(0, ep_len, stride)[:len(pred_first_steps)]
    active_groups = [(t, [d for d in dims if d < n_dims], u) for t, dims, u in DIM_GROUPS
                     if any(d < n_dims for d in dims)]

    fig, axes = plt.subplots(len(active_groups), 1, figsize=(14, 3.5 * len(active_groups)), sharex=True)
    if len(active_groups) == 1:
        axes = [axes]
    fig.suptitle(f"GT vs Predicted (first step) | Episode {ep_idx}", fontsize=13)

    tab10 = plt.cm.tab10
    all_dims = [d for _, dims, _ in active_groups for d in dims]
    dim_colors = {d: tab10(ci % 10) for ci, d in enumerate(all_dims)}

    for ax, (title, dims, unit) in zip(axes, active_groups):
        for d in dims:
            label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f"dim{d}"
            ax.plot(np.arange(ep_len), gt_full[:, d], "-", color=dim_colors[d],
                    label=f"GT {label}", alpha=0.9, linewidth=1.5)
            ax.plot(sample_frames, pred_first_steps[:, d], "x", color=dim_colors[d],
                    label=f"Pred {label}", alpha=0.7, markersize=4)
        ax.set_title(title)
        ax.set_ylabel(f"Value ({unit})" if unit else "Value")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    fname = save_dir / f"gt_vs_pred_ep{ep_idx}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()

    # ---- Plot 3: GT vs Predicted with action chunk overlay ----
    fig, axes = plt.subplots(len(active_groups), 1, figsize=(14, 3.5 * len(active_groups)), sharex=True)
    if len(active_groups) == 1:
        axes = [axes]
    fig.suptitle(f"GT vs Predicted (action chunk overlay) | Episode {ep_idx}", fontsize=13)

    for ax, (title, dims, unit) in zip(axes, active_groups):
        # GT as solid lines
        for d in dims:
            label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f"dim{d}"
            ax.plot(np.arange(ep_len), gt_full[:, d], "-", color=dim_colors[d],
                    label=f"GT {label}", alpha=0.9, linewidth=1.5)
        # Predicted chunks as dashed lines
        for chunk_idx, (frame_offset, chunk) in enumerate(pred_chunks):
            chunk_len = chunk.shape[0]
            x_chunk = np.arange(frame_offset, min(frame_offset + chunk_len, ep_len))
            valid_len = len(x_chunk)
            for d in dims:
                lbl = f"Pred {ACTION_LABELS[d]}" if chunk_idx == 0 else None
                ax.plot(x_chunk, chunk[:valid_len, d], "--", color=dim_colors[d],
                        alpha=0.3, linewidth=0.8, label=lbl)
        ax.set_title(title)
        ax.set_ylabel(f"Value ({unit})" if unit else "Value")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    fname = save_dir / f"gt_vs_pred_chunks_ep{ep_idx}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()


def parse_episode_arg(episode_str, n_episodes):
    if episode_str is None:
        return [np.random.randint(0, n_episodes)]
    if "-" in episode_str:
        start, end = episode_str.split("-", 1)
        return list(range(int(start), min(int(end) + 1, n_episodes)))
    return [int(episode_str)]


def main():
    parser = argparse.ArgumentParser(description="Open-loop eval for Psi0 18-dim EEF model")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to .runs/finetune/<run_folder>")
    parser.add_argument("--ckpt-step", type=int, required=True, help="Checkpoint step to load")
    parser.add_argument("--episode", type=str, default=None, help="Episode index or range (e.g. 5 or 0-10)")
    parser.add_argument("--stride", type=int, default=4, help="Sample every N frames")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root dir (default: from run_config)")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory (default: <run-dir>/eval)")
    args = parser.parse_args()

    psi0, dataset, maxmin, model_cfg, device = load_model_and_data(
        args.run_dir, args.ckpt_step, args.device, args.data_root
    )

    n_episodes = dataset.raw_dataset.meta.total_episodes
    print(f"Dataset: {n_episodes} episodes")

    ep_indices = parse_episode_arg(args.episode, n_episodes)
    print(f"Evaluating episodes: {ep_indices}")

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.run_dir) / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in ep_indices:
        results = eval_episode(psi0, dataset, maxmin, device, ep_idx, args.stride, args.num_inference_steps)
        print_summary(results)
        plot_results(results, ep_idx, save_dir)

    print(f"\nDone! Plots saved to {save_dir}")


if __name__ == "__main__":
    main()
