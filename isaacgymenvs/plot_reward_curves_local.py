# -*- coding: utf-8 -*-
# 本機執行版本 — 不需要 Colab
# 執行前確認：pip install tensorboard matplotlib numpy
#
# 使用方式：
#   cd /home/erc/isaacgym/python/IsaacGymEnvs/isaacgymenvs
#   python plot_reward_curves_local.py

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── 輸出資料夾 ───────────────────────────────────────────────────────
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 圖表樣式 ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
})

COLORS = [
    '#4363D8',  # Blue  — PPO
    '#3CB44B',  # Green — MoE-PPO
]

# ── Runs 路徑（本機絕對路徑）────────────────────────────────────────
RUNS_ROOT = "/home/erc/isaacgym/python/IsaacGymEnvs/isaacgymenvs/runs"

mix_title = "Mix (3 Objects) - Training Curves"
mix_checkpoints = [
    f"{RUNS_ROOT}/DiabloGraspCustom3_29-22-48-04/summaries/events.out.tfevents.1777474090.erc-rtx3050",
    f"{RUNS_ROOT}/DiabloGraspCustom3_30-14-28-47/summaries/events.out.tfevents.1777530533.erc-rtx3050",
]

drill_title = "Drill P&P - Training Reward Curves"
drill_checkpoints = [
    f"{RUNS_ROOT}/DrillDiabloGraspCustom3/DiabloGraspCustom3_14-08-42-05-drill/summaries/events.out.tfevents.*",
    f"{RUNS_ROOT}/drill_grasp_moe_num_actor_2_seed_1234_15-19-45-45/summaries/events.out.tfevents.*",
]

mug_title = "Mug P&P - Training Reward Curves"
mug_checkpoints = [
    f"{RUNS_ROOT}/MugDiabloGraspCustom3/DiabloGraspCustom3_13-23-11-24/summaries/events.out.tfevents.*",
    f"{RUNS_ROOT}/mug_grasp_moe_num_actor_2_seed_42_14-13-24-55/summaries/events.out.tfevents.*",
]

dumbbell_title = "Dumbbell P&P - Training Reward Curves"
dumbbell_checkpoints = [
    f"{RUNS_ROOT}/dumbbell_grasp_moe_num_actor_1_seed_27404_17-10-37-27/summaries/events.out.tfevents.*",
    f"{RUNS_ROOT}/dumbbell_grasp_moe_num_actor_2_seed_2027_17-09-02-30/summaries/events.out.tfevents.*",
]

labels = [
    "PPO",
    "MoE-PPO",
]

# ── 工具函式 ─────────────────────────────────────────────────────────
def exponential_moving_average(data, smoothing):
    if smoothing == 0:
        return data
    smoothed = np.zeros_like(data)
    last = 0
    debias_weight = 0
    for i, point in enumerate(data):
        last = last * smoothing + (1 - smoothing) * point
        debias_weight = debias_weight * smoothing + (1 - smoothing)
        smoothed[i] = last / debias_weight
    return smoothed


def rolling_std(data, window=200):
    result = np.zeros_like(data, dtype=float)
    half = window // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half)
        result[i] = np.std(data[start:end])
    return result


def load_tensorboard_data(checkpoints, labels, tag="rewards/iter", max_iters=None):
    grouped_rewards = {}

    for checkpoint, label in zip(checkpoints, labels):
        event_files = glob(checkpoint)

        if not event_files:
            print(f"Warning: No files found for pattern: {checkpoint}")
            continue

        if label not in grouped_rewards:
            grouped_rewards[label] = []

        for event_file in event_files:
            print(f"Loading: {event_file}")
            ea = EventAccumulator(event_file)
            ea.Reload()

            if tag in ea.Tags()['scalars']:
                scalar_events = ea.Scalars(tag)
                reward_data = np.array([event.value for event in scalar_events])
                grouped_rewards[label].append(reward_data)
                print(f"  Loaded {len(reward_data)} data points from {label}")
            else:
                print(f"  Warning: '{tag}' tag not found in {event_file}")
                print(f"  Available tags: {ea.Tags()['scalars']}")

    min_length = min(len(data) for group in grouped_rewards.values() for data in group)
    print(f"\nMinimum timesteps across all runs: {min_length}")

    if max_iters is not None:
        final_length = min(min_length, max_iters)
        print(f"Maximum iterations limit: {max_iters}")
        print(f"Final trimmed length: {final_length}")
    else:
        final_length = min_length

    for label in grouped_rewards:
        grouped_rewards[label] = [data[:final_length] for data in grouped_rewards[label]]

    common_timesteps = np.arange(1, final_length + 1)

    print(f"\nGrouped rewards summary:")
    for label, group in grouped_rewards.items():
        print(f"  {label}: {len(group)} run(s), {len(group[0])} timesteps each")

    return grouped_rewards, common_timesteps


from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    return f'{int(x/1000)}'


def plot_training_curves(grouped_rewards, common_timesteps, title,
                         smoothing=0.9, figsize=(12, 6),
                         output_filename="total_rewards.png"):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (label, group) in enumerate(grouped_rewards.items()):
        color = COLORS[i % len(COLORS)]
        raw_group = np.array(group)

        if len(group) > 1:
            raw_mean = np.mean(raw_group, axis=0)
            raw_std  = np.std(raw_group, axis=0)
            mean = exponential_moving_average(raw_mean, smoothing)
            std  = exponential_moving_average(raw_std, smoothing)
        else:
            raw  = raw_group[0]
            mean = exponential_moving_average(raw, smoothing)
            std  = exponential_moving_average(rolling_std(raw, window=200), smoothing)

        print(f"Plotting '{label}': mean final = {mean[-1]:.2f}, std final = {std[-1]:.2f}")

        ax.plot(common_timesteps, mean, label=label, alpha=0.9,
                color=color, linestyle="-", linewidth=2.5)
        ax.fill_between(common_timesteps, mean - std, mean + std,
                        alpha=0.2, color=color)

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    min_x = common_timesteps[0]
    max_x = common_timesteps[-1]
    ax.set_xlim(min_x, max_x)
    x_ticks = np.linspace(min_x, max_x, 6, dtype=int)
    ax.set_xticks(x_ticks)

    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Reward (×10$^3$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_filename}")

    return fig, ax


def load_tensorboard_tag(event_files, tag):
    """從單一 event file（直接路徑，非 glob）讀取指定 tag。"""
    ea = EventAccumulator(event_files)
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        print(f"  Warning: '{tag}' not found in {event_files}")
        return None
    return np.array([e.value for e in ea.Scalars(tag)])


def plot_mix_curves(event_files, labels, tags_info, max_iters=2000,
                    smoothing=0.9, output_prefix="mix"):
    """
    event_files: list of direct file paths (not glob)
    tags_info: list of (tag, ylabel, filename_suffix)
    """
    for tag, ylabel, suffix in tags_info:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, (fpath, label) in enumerate(zip(event_files, labels)):
            data = load_tensorboard_tag(fpath, tag)
            if data is None:
                continue
            data = data[:max_iters]
            xs = np.arange(1, len(data) + 1)
            smoothed = exponential_moving_average(data, smoothing)
            std = exponential_moving_average(rolling_std(data, window=200), smoothing)
            color = COLORS[i % len(COLORS)]
            ax.plot(xs, smoothed, label=label, color=color, linewidth=2.5)
            ax.fill_between(xs, smoothed - std, smoothed + std, alpha=0.2, color=color)

        ax.set_xlim(1, max_iters)
        x_ticks = np.linspace(1, max_iters, 6, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xlabel("Iterations")
        if "10$^3$" in ylabel:
            ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.set_ylabel(ylabel)
        ax.set_title(f"Mix (3 Objects) - Training Reward Curves")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.5, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        out = f"{OUTPUT_DIR}/{output_prefix}_{suffix}.png"
        fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {out}")
        plt.show()


# ── 主程式 ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 80)
    print("MIX (3 OBJECTS) GENERALIZATION")
    print("=" * 80)
    mix_tags = [
        ("rewards/iter",                    r"Reward (×10$^3$)",  "total_reward"),
        ("metrics/success_rate/iter",       "Success Rate",        "success_rate"),
        ("metrics/mean_placement_dist_m/iter", "Placement Dist (m)", "placement_dist"),
    ]
    plot_mix_curves(mix_checkpoints, labels, mix_tags, max_iters=2000, smoothing=0.9)

    print("=" * 80)
    print("DRILL P&P")
    print("=" * 80)
    drill_rewards, drill_timesteps = load_tensorboard_data(
        drill_checkpoints, labels, max_iters=1500)
    plot_training_curves(
        drill_rewards, drill_timesteps, drill_title,
        smoothing=0.9,
        output_filename=f"{OUTPUT_DIR}/drill_total_rewards.png")
    plt.show()

    print("=" * 80)
    print("MUG P&P")
    print("=" * 80)
    mug_rewards, mug_timesteps = load_tensorboard_data(
        mug_checkpoints, labels, max_iters=1500)
    plot_training_curves(
        mug_rewards, mug_timesteps, mug_title,
        smoothing=0.9,
        output_filename=f"{OUTPUT_DIR}/mug_total_rewards.png")
    plt.show()

    print("=" * 80)
    print("DUMBBELL P&P")
    print("=" * 80)
    dumbbell_rewards, dumbbell_timesteps = load_tensorboard_data(
        dumbbell_checkpoints, labels, max_iters=1500)
    plot_training_curves(
        dumbbell_rewards, dumbbell_timesteps, dumbbell_title,
        smoothing=0.9,
        output_filename=f"{OUTPUT_DIR}/dumbbell_total_rewards.png")
    plt.show()

    print("\n全部完成！圖片儲存在 figures/ 資料夾")
