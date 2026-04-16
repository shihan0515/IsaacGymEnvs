#!/bin/bash
args=("42" "7" "123" "2021" "0" "193" "397")

GPU_ID=0  # Specify which GPU to use


# Drill
# NUM_ACTORS=1
# EXP_NAME="runs/DrillDiabloGraspCustom3/DiabloGraspCustom3_14-08-42-05-drill/nn/DiabloGraspCustom3.pth"
NUM_ACTORS=2
EXP_NAME="runs/drill_grasp_moe_num_actor_2_seed_1234_15-19-45-45/nn/drill_grasp_moe_num_actor_2_seed_1234.pth"

# Mug
# NUM_ACTORS=1
# EXP_NAME="runs/MugDiabloGraspCustom3/DiabloGraspCustom3_13-23-11-24/nn/DiabloGraspCustom3.pth"
# NUM_ACTORS=2
# EXP_NAME="runs/mug_grasp_moe_num_actor_2_seed_42_14-13-24-55/nn/mug_grasp_moe_num_actor_2_seed_42.pth"

# Dumbbell
# NUM_ACTORS=1
# EXP_NAME="runs/dumbbell_grasp_moe_num_actor_1_seed_27404_17-10-37-27/nn/dumbbell_grasp_moe_num_actor_1_seed_27404.pth"
# NUM_ACTORS=2
# EXP_NAME="runs/dumbbell_grasp_moe_num_actor_2_seed_2027_17-09-02-30/nn/dumbbell_grasp_moe_num_actor_2_seed_2027.pth"

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="evaluation_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Summary file
SUMMARY_FILE="$LOG_DIR/summary_results.txt"
echo "Evaluation Summary for $EXP_NAME" > "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for arg in "${args[@]}"
do
    echo "Starting Evaluation, Exp: $EXP_NAME, Seed: $arg"
    LOG_FILE="$LOG_DIR/seed_${arg}.log"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        task=DiabloGraspCustom3 \
        headless=True \
        num_envs=1000 \
        seed="$arg" \
        test=True\
        moe_num_actors="$NUM_ACTORS" \
        checkpoint=$EXP_NAME \
        2>&1 | tee "$LOG_FILE"
    
    # Extract and append key metrics to summary
    echo "Seed: $arg" >> "$SUMMARY_FILE"
    grep "final success_rate:" "$LOG_FILE" >> "$SUMMARY_FILE" 2>/dev/null
    grep "final mean_placement_dist:" "$LOG_FILE" >> "$SUMMARY_FILE" 2>/dev/null
    echo "" >> "$SUMMARY_FILE"

done

# Print summary location
echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Logs saved to: $LOG_DIR"
echo "Summary saved to: $SUMMARY_FILE"
echo "=========================================="
echo ""
echo "Summary Results:"
cat "$SUMMARY_FILE"
