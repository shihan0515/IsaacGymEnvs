#!/bin/bash
# 評估泛化模型在三個物件上的個別成功率
# 使用 task.env.eval_object_name 強制指定單一物件

SEEDS=("42" "7" "123" "2021" "0")
OBJECTS=("mug" "drill" "dumbbell")
GPU_ID=0

# ── 模型設定（切換這裡來比較 PPO vs MoE-PPO）──────────────────────
PPO_ACTORS=1
PPO_CKPT="runs/DiabloGraspCustom3_29-22-48-04/nn/DiabloGraspCustom3.pth"

MOE_ACTORS=2
MOE_CKPT="runs/DiabloGraspCustom3_30-14-28-47/nn/DiabloGraspCustom3.pth"

# ── 建立 log 資料夾 ───────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="eval_generalization_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SUMMARY="$LOG_DIR/summary.txt"
echo "Per-Object Generalization Evaluation" > "$SUMMARY"
echo "PPO:     $PPO_CKPT" >> "$SUMMARY"
echo "MoE-PPO: $MOE_CKPT" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"

run_eval() {
    local label=$1
    local ckpt=$2
    local num_actors=$3
    local obj=$4
    local seed=$5
    local log_file=$6

    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        task=DiabloGraspCustom3 \
        headless=True \
        num_envs=1000 \
        seed="$seed" \
        test=True \
        moe_num_actors="$num_actors" \
        checkpoint="$ckpt" \
        task.env.eval_object_name="$obj" \
        2>&1 | tee "$log_file"
}

for obj in "${OBJECTS[@]}"
do
    echo ""
    echo "=========================================="
    echo "Object: $obj"
    echo "=========================================="
    echo "" >> "$SUMMARY"
    echo "[ Object: $obj ]" >> "$SUMMARY"

    for label in "PPO" "MoE-PPO"; do
        if [ "$label" = "PPO" ]; then
            ckpt=$PPO_CKPT; actors=$PPO_ACTORS
        else
            ckpt=$MOE_CKPT; actors=$MOE_ACTORS
        fi

        echo "  -- $label --"
        echo "  $label:" >> "$SUMMARY"

        for seed in "${SEEDS[@]}"
        do
            log_file="$LOG_DIR/${obj}_${label}_seed${seed}.log"
            echo "    Seed: $seed"
            run_eval "$label" "$ckpt" "$actors" "$obj" "$seed" "$log_file"

            sr=$(grep "final success_rate:" "$log_file" | tail -1)
            dist=$(grep "final mean_placement_dist:" "$log_file" | tail -1)
            echo "    seed=${seed}  $sr  $dist" >> "$SUMMARY"
        done
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Summary saved to: $SUMMARY"
echo "=========================================="
echo ""
cat "$SUMMARY"
