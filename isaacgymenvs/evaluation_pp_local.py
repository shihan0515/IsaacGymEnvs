# -*- coding: utf-8 -*-
# 本機執行版本 - 不需要 Colab

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 設定路徑 ────────────────────────────────────────────────────────
XLSX_PATH = "/home/erc/Downloads/pp_evaluation.xlsx"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 讀取資料 ────────────────────────────────────────────────────────
def _seed_rows(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    return df[pd.to_numeric(df["seed"], errors="coerce").notna()].reset_index(drop=True)

df_drill    = _seed_rows(XLSX_PATH, "drill")
df_mug      = _seed_rows(XLSX_PATH, "mug")
df_dumbbell = _seed_rows(XLSX_PATH, "dumbbell")

print(df_drill.head())

# ── 圖表樣式 ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
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

# ── 計算統計 ────────────────────────────────────────────────────────
def calc_mean_std(df):
    ppo = {
        'ppo_sr':         (df['ppo_sr'].mean(),         df['ppo_sr'].std()),
        'ppo_l2':         (df['ppo_l2'].mean(),         df['ppo_l2'].std()),
        'ppo_avg_reward': (df['ppo_avg_reward'].mean(), df['ppo_avg_reward'].std()),
        'ppo_avg_steps':  (df['ppo_avg_steps'].mean(),  df['ppo_avg_steps'].std()),
    }
    moe = {
        'moe_sr':         (df['moe_sr'].mean(),         df['moe_sr'].std()),
        'moe_l2':         (df['moe_l2'].mean(),         df['moe_l2'].std()),
        'moe_avg_reward': (df['moe_avg_reward'].mean(), df['moe_avg_reward'].std()),
        'moe_avg_steps':  (df['moe_avg_steps'].mean(),  df['moe_avg_steps'].std()),
    }

    print("=== Evaluation Results ===")
    print(f"PPO Success Rate:    {ppo['ppo_sr'][0]:.2f} ± {ppo['ppo_sr'][1]:.2f}")
    print(f"PPO L2 Distance:     {ppo['ppo_l2'][0]:.4f} ± {ppo['ppo_l2'][1]:.4f}")
    print(f"PPO Average Reward:  {ppo['ppo_avg_reward'][0]:.2f} ± {ppo['ppo_avg_reward'][1]:.2f}")
    print(f"PPO Average Steps:   {ppo['ppo_avg_steps'][0]:.2f} ± {ppo['ppo_avg_steps'][1]:.2f}")
    print("--------------------------")
    print(f"MoE-PPO Success Rate:   {moe['moe_sr'][0]:.2f} ± {moe['moe_sr'][1]:.2f}")
    print(f"MoE-PPO L2 Distance:    {moe['moe_l2'][0]:.4f} ± {moe['moe_l2'][1]:.4f}")
    print(f"MoE-PPO Average Reward: {moe['moe_avg_reward'][0]:.2f} ± {moe['moe_avg_reward'][1]:.2f}")
    print(f"MoE-PPO Average Steps:  {moe['moe_avg_steps'][0]:.2f} ± {moe['moe_avg_steps'][1]:.2f}")
    print("==========================\n")
    return ppo, moe

drill_ppo,    drill_moe    = calc_mean_std(df_drill)
mug_ppo,      mug_moe      = calc_mean_std(df_mug)
dumbbell_ppo, dumbbell_moe = calc_mean_std(df_dumbbell)

# ── 共用繪圖函式 ────────────────────────────────────────────────────
def draw_bar(ax, x, width, ppo_means, ppo_stds, moe_means, moe_stds,
             title, ylabel, ppo_label='PPO', moe_label='MoE-PPO'):
    ax.bar(x - width/2, ppo_means, width,
           label=ppo_label, color=COLORS[0], edgecolor='black', linewidth=0.6)
    ax.bar(x + width/2, moe_means, width,
           label=moe_label, color=COLORS[1], edgecolor='black', linewidth=0.6)

    # std overlay
    for i, (m, s) in enumerate(zip(ppo_means, ppo_stds)):
        ax.bar(x[i]-width/2, 2*s, width, bottom=m-s,
               color='white', alpha=0.35, edgecolor='none', zorder=3)
    for i, (m, s) in enumerate(zip(moe_means, moe_stds)):
        ax.bar(x[i]+width/2, 2*s, width, bottom=m-s,
               color='white', alpha=0.35, edgecolor='none', zorder=3)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(["Drill", "Mug", "Dumbbell"])
    ax.grid(True, axis='y', alpha=1.0, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=2, frameon=True, fancybox=False, shadow=False,
              framealpha=0.9, columnspacing=1.5, handlelength=1.5,
              handletextpad=0.5, borderpad=0.4, labelspacing=0.3)

x     = np.arange(3)
width = 0.35

# ── Success Rate ────────────────────────────────────────────────────
ppo_sr_means = [drill_ppo['ppo_sr'][0], mug_ppo['ppo_sr'][0], dumbbell_ppo['ppo_sr'][0]]
ppo_sr_stds  = [drill_ppo['ppo_sr'][1], mug_ppo['ppo_sr'][1], dumbbell_ppo['ppo_sr'][1]]
moe_sr_means = [drill_moe['moe_sr'][0], mug_moe['moe_sr'][0], dumbbell_moe['moe_sr'][0]]
moe_sr_stds  = [drill_moe['moe_sr'][1], mug_moe['moe_sr'][1], dumbbell_moe['moe_sr'][1]]

fig, ax = plt.subplots(figsize=(8, 5))
draw_bar(ax, x, width, ppo_sr_means, ppo_sr_stds, moe_sr_means, moe_sr_stds,
         "Success Rate Comparison", "Success Rate (%)")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/plot_success_rate.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved: {OUTPUT_DIR}/plot_success_rate.png")
plt.show()

# ── L2 Distance ─────────────────────────────────────────────────────
ppo_l2_means = [drill_ppo['ppo_l2'][0], mug_ppo['ppo_l2'][0], dumbbell_ppo['ppo_l2'][0]]
ppo_l2_stds  = [drill_ppo['ppo_l2'][1], mug_ppo['ppo_l2'][1], dumbbell_ppo['ppo_l2'][1]]
moe_l2_means = [drill_moe['moe_l2'][0], mug_moe['moe_l2'][0], dumbbell_moe['moe_l2'][0]]
moe_l2_stds  = [drill_moe['moe_l2'][1], mug_moe['moe_l2'][1], dumbbell_moe['moe_l2'][1]]

fig, ax = plt.subplots(figsize=(8, 5))
draw_bar(ax, x, width, ppo_l2_means, ppo_l2_stds, moe_l2_means, moe_l2_stds,
         "L2 Distance Comparison", "L2 Distance (mm)")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/plot_l2_dist.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved: {OUTPUT_DIR}/plot_l2_dist.png")
plt.show()

print("全部完成！圖片儲存在 figures/ 資料夾")


# ── Generalization 版本（pp_evaluation_generalization.xlsx）──────────
XLSX_GEN = "/home/erc/Downloads/pp_evaluation_generalization.xlsx"

def load_seed_rows(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    # 只保留 seed 為數字的列（排除 Average、improve、NaN）
    return df[pd.to_numeric(df["seed"], errors="coerce").notna()].reset_index(drop=True)

df_gen_drill    = load_seed_rows(XLSX_GEN, "drill")
df_gen_mug      = load_seed_rows(XLSX_GEN, "mug")
df_gen_dumbbell = load_seed_rows(XLSX_GEN, "dumbbell")

print("\n=== Generalization Results ===")
gen_drill_ppo,    gen_drill_moe    = calc_mean_std(df_gen_drill)
gen_mug_ppo,      gen_mug_moe      = calc_mean_std(df_gen_mug)
gen_dumbbell_ppo, gen_dumbbell_moe = calc_mean_std(df_gen_dumbbell)

# Success Rate
ppo_sr_means = [gen_drill_ppo['ppo_sr'][0], gen_mug_ppo['ppo_sr'][0], gen_dumbbell_ppo['ppo_sr'][0]]
ppo_sr_stds  = [gen_drill_ppo['ppo_sr'][1], gen_mug_ppo['ppo_sr'][1], gen_dumbbell_ppo['ppo_sr'][1]]
moe_sr_means = [gen_drill_moe['moe_sr'][0], gen_mug_moe['moe_sr'][0], gen_dumbbell_moe['moe_sr'][0]]
moe_sr_stds  = [gen_drill_moe['moe_sr'][1], gen_mug_moe['moe_sr'][1], gen_dumbbell_moe['moe_sr'][1]]

fig, ax = plt.subplots(figsize=(8, 5))
draw_bar(ax, x, width, ppo_sr_means, ppo_sr_stds, moe_sr_means, moe_sr_stds,
         "Generalization - Success Rate Comparison", "Success Rate (%)")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/plot_gen_success_rate.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved: {OUTPUT_DIR}/plot_gen_success_rate.png")
plt.show()

# L2 Distance
ppo_l2_means = [gen_drill_ppo['ppo_l2'][0], gen_mug_ppo['ppo_l2'][0], gen_dumbbell_ppo['ppo_l2'][0]]
ppo_l2_stds  = [gen_drill_ppo['ppo_l2'][1], gen_mug_ppo['ppo_l2'][1], gen_dumbbell_ppo['ppo_l2'][1]]
moe_l2_means = [gen_drill_moe['moe_l2'][0], gen_mug_moe['moe_l2'][0], gen_dumbbell_moe['moe_l2'][0]]
moe_l2_stds  = [gen_drill_moe['moe_l2'][1], gen_mug_moe['moe_l2'][1], gen_dumbbell_moe['moe_l2'][1]]

fig, ax = plt.subplots(figsize=(8, 5))
draw_bar(ax, x, width, ppo_l2_means, ppo_l2_stds, moe_l2_means, moe_l2_stds,
         "Generalization - L2 Distance Comparison", "L2 Distance (mm)")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/plot_gen_l2_dist.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved: {OUTPUT_DIR}/plot_gen_l2_dist.png")
plt.show()

print("泛化版完成！")
