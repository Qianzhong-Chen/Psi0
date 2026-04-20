"""
Open-loop evaluation for Psi0 fine-tuned with 18-dim EEF actions.

Loads model from a training run directory, runs inference on the training
dataset, and plots L1 errors per action dimension.

Usage:
    python examples/simple/openloop_eval_18act_eef.py \
        --run-dir .runs/finetune/<run_folder> \
        --ckpt-step 30000 \
        --episode 0-5 \
        --stride 4

Action (18): L_eef_6d(6) + L_grip(1) + R_eef_6d(6) + R_grip(1) + vx + vy + vyaw + height
State  (15): L_eef_6d(6) + L_grip(1) + R_eef_6d(6) + R_grip(1) + height
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from psi.utils import parse_args_to_tyro_config, seed_everything, move_to_device
from psi.models.psi0 import Psi0Model
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig

ACTION_LABELS = [
    "L_x", "L_y", "L_z", "L_roll", "L_pitch", "L_yaw", "L_grip",
    "R_x", "R_y", "R_z", "R_roll", "R_pitch", "R_yaw", "R_grip",
    "vx", "vy", "vyaw", "height",
]

# Semantic groups for plotting
PLOT_GROUPS = [
    ("Left Wrist XYZ (m)", [0, 1, 2]),
    ("Left Wrist Rot (rad)", [3, 4, 5]),
    ("Left Grip", [6]),
    ("Right Wrist XYZ (m)", [7, 8, 9]),
    ("Right Wrist Rot (rad)", [10, 11, 12]),
    ("Right Grip", [13]),
    ("Locomotion (vx, vy, vyaw)", [14, 15, 16]),
    ("Height (m)", [17]),
]

# For norm-based summary
ERROR_GROUPS = {
    "L_eef_6d": slice(0, 6),
    "L_grip": slice(6, 7),
    "R_eef_6d": slice(7, 13),
    "R_grip": slice(13, 14),
    "vx": slice(14, 15),
    "vy": slice(15, 16),
    "vyaw": slice(16, 17),
    "height": slice(17, 18),
}


def load_model_and_data(run_dir, ckpt_step, device="cuda:0"):
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
    model_cfg: Psi0ModelConfig = launch_config.model
    maxmin = data_cfg.transform.field

    transform_kwargs = dict(vlm_processor=psi0.vlm_processor)
    dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)

    return psi0, dataset, maxmin, model_cfg, device


def eval_episode(psi0, dataset, maxmin, device, ep_idx, stride=4, num_inference_steps=10):
    """Run open-loop evaluation on a single episode."""
    raw_ds = dataset.raw_dataset
    start_idx = raw_ds.base_dataset.episode_data_index["from"][ep_idx].item()
    end_idx = raw_ds.base_dataset.episode_data_index["to"][ep_idx].item()
    ep_len = end_idx - start_idx

    print(f"\n--- Episode {ep_idx}: {ep_len} frames (idx {start_idx}-{end_idx}) ---")

    errors_list = []

    for i in tqdm(range(start_idx, end_idx, stride), desc=f"Ep {ep_idx}"):
        frame = dataset[i]
        images = frame["raw_images"]
        instruction = frame["instruction"]
        states = frame["states"]

        batch_images = [images]
        batch_instructions = [instruction]
        batch_states = torch.from_numpy(states).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_actions = psi0.predict_action(
                observations=batch_images,
                states=batch_states,
                instructions=batch_instructions,
                num_inference_steps=num_inference_steps,
                traj2ds=None,
            )

        gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(device)
        pred_denormed = maxmin.denormalize(pred_actions)

        error = (pred_denormed - gt_action).detach().abs().cpu().numpy()
        error = error.reshape(-1, gt_action.shape[-1])
        avg_error = error.mean(axis=0)
        errors_list.append(avg_error)

    errors = np.stack(errors_list, axis=0)
    return errors


def print_summary(errors):
    """Print per-group error summary."""
    mean_errors = errors.mean(axis=0)

    print("\n--- Per-dimension mean L1 error ---")
    n_dims = min(len(ACTION_LABELS), mean_errors.shape[0])
    for i in range(n_dims):
        print(f"  {ACTION_LABELS[i]:15s}: {mean_errors[i]:.6f}")

    print("\n--- Per-group error norm ---")
    for name, sl in ERROR_GROUPS.items():
        group_err = mean_errors[sl]
        print(f"  {name:15s} {group_err.shape}: {np.linalg.norm(group_err):.6f}")


def plot_results(errors, ep_idx, save_dir):
    """Generate and save error plots."""
    import matplotlib.pyplot as plt

    mean_errors = errors.mean(axis=0)
    n_dims = min(len(ACTION_LABELS), mean_errors.shape[0])

    # Bar chart: per-dimension mean L1 error
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_dims)
    bars = ax.bar(x, mean_errors[:n_dims], color="steelblue")
    ax.set_ylabel("Mean L1 Error")
    ax.set_title(f"Open-Loop L1 Error per Dimension | Episode {ep_idx}")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS[:n_dims], rotation=45, ha="right")
    for bar, val in zip(bars, mean_errors):
        if val > 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fname = save_dir / f"l1_bar_ep{ep_idx}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()

    # Time-series: error over time per group
    active_groups = [(title, dims) for title, dims in PLOT_GROUPS if max(dims) < errors.shape[1]]
    fig, axes = plt.subplots(len(active_groups), 1, figsize=(12, 3 * len(active_groups)), sharex=True)
    if len(active_groups) == 1:
        axes = [axes]
    fig.suptitle(f"Open-Loop Error Over Time | Episode {ep_idx}", fontsize=13)

    for ax, (title, dims) in zip(axes, active_groups):
        for d in dims:
            ax.plot(errors[:, d], label=ACTION_LABELS[d])
        ax.set_title(title)
        ax.set_ylabel("L1 Error")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample step")
    plt.tight_layout()
    fname = save_dir / f"l1_over_time_ep{ep_idx}.png"
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
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory (default: <run-dir>/eval)")
    args = parser.parse_args()

    psi0, dataset, maxmin, model_cfg, device = load_model_and_data(
        args.run_dir, args.ckpt_step, args.device
    )

    n_episodes = dataset.raw_dataset.meta.total_episodes
    print(f"Dataset: {n_episodes} episodes")

    ep_indices = parse_episode_arg(args.episode, n_episodes)
    print(f"Evaluating episodes: {ep_indices}")

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.run_dir) / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in ep_indices:
        errors = eval_episode(psi0, dataset, maxmin, device, ep_idx, args.stride, args.num_inference_steps)
        print_summary(errors)
        plot_results(errors, ep_idx, save_dir)

    print(f"\nDone! Plots saved to {save_dir}")


if __name__ == "__main__":
    main()
