#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 解析泛化評估 log 資料夾，輸出成 Excel（格式同 pp_evaluation.xlsx）
#
# 使用方式：
#   python parse_generalization_to_excel.py <log_dir> <output_xlsx>
#
# 範例：
#   python parse_generalization_to_excel.py eval_generalization_20260501_111214 /home/erc/Downloads/pp_evaluation_generalization.xlsx

import re
import os
import sys
import pandas as pd

OBJECTS = ["mug", "drill", "dumbbell"]
SEEDS   = ["42", "7", "123", "2021", "0"]


def parse_log(path):
    if not os.path.exists(path):
        return None
    sr = l2 = avg_reward = avg_steps = None
    with open(path) as f:
        content = f.read()

    # 正常格式：av reward: <num>  av steps: <num>
    m = re.search(r"av reward: ([\d.]+)\s+av steps: ([\d.]+)", content)
    if m:
        avg_reward, avg_steps = float(m.group(1)), float(m.group(2))

    # warning 截斷：下一行開頭是數值
    if avg_reward is None:
        m = re.search(r"\n([\d.]+) av steps: ([\d.]+)", content)
        if m:
            avg_reward, avg_steps = float(m.group(1)), float(m.group(2))

    # warning 夾在數值後面：av reward: <num>/home/...
    if avg_reward is None:
        m = re.search(r"av reward: ([\d.]+)/", content)
        if m:
            avg_reward = float(m.group(1))

    # av steps 單獨出現在某行
    if avg_steps is None:
        m = re.search(r"av steps: ([\d.]+)", content)
        if m:
            avg_steps = float(m.group(1))

    m = re.search(r"final success_rate: [\d.]+ \(([\d.]+)%\)", content)
    if m:
        sr = float(m.group(1))

    m = re.search(r"final mean_placement_dist: [\d.]+cm \(([\d.]+)mm\)", content)
    if m:
        l2 = float(m.group(1))

    return {"sr": sr, "l2": l2, "avg_reward": avg_reward, "avg_steps": avg_steps}


def build_sheet(log_dir, obj):
    rows = []
    for seed in SEEDS:
        ppo = parse_log(f"{log_dir}/{obj}_PPO_seed{seed}.log")
        moe = parse_log(f"{log_dir}/{obj}_MoE-PPO_seed{seed}.log")
        if None in (ppo, moe):
            print(f"  Missing file: {obj} seed={seed}")
            continue
        if any(v is None for v in [*ppo.values(), *moe.values()]):
            print(f"  Incomplete: {obj} seed={seed} ppo={ppo} moe={moe}")
            continue
        rows.append({
            "seed":           int(seed),
            "ppo_sr":         ppo["sr"],
            "ppo_l2":         ppo["l2"],
            "ppo_avg_reward": round(ppo["avg_reward"], 3),
            "ppo_avg_steps":  round(ppo["avg_steps"], 3),
            "moe_sr":         moe["sr"],
            "moe_l2":         moe["l2"],
            "moe_avg_reward": round(moe["avg_reward"], 3),
            "moe_avg_steps":  round(moe["avg_steps"], 3),
        })

    df = pd.DataFrame(rows)

    avg = {c: df[c].mean() for c in df.columns if c != "seed"}
    avg["seed"] = "Average"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    std = {c: df[df["seed"].apply(lambda x: str(x).isdigit())][c].std() for c in df.columns if c != "seed"}
    std["seed"] = "Std"
    df = pd.concat([df, pd.DataFrame([std])], ignore_index=True)

    imp = {"seed": "improve"}
    for metric in ["sr", "l2", "avg_reward", "avg_steps"]:
        pv = avg[f"ppo_{metric}"]
        mv = avg[f"moe_{metric}"]
        imp[f"ppo_{metric}"] = (mv - pv) / pv if pv else None
    df = pd.concat([df, pd.DataFrame([imp])], ignore_index=True)

    return df, len(rows)


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_generalization_to_excel.py <log_dir> <output_xlsx>")
        sys.exit(1)

    log_dir = sys.argv[1]
    out     = sys.argv[2]

    if not os.path.isdir(log_dir):
        print(f"Error: {log_dir} is not a directory")
        sys.exit(1)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for obj in OBJECTS:
            df, n = build_sheet(log_dir, obj)
            df.to_excel(writer, sheet_name=obj, index=False)
            print(f"Sheet '{obj}': {n} seeds written")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
