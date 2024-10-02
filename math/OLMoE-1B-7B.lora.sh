#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=15:59:00
#SBATCH --job-name=OLMoE-1B-7B.lora.train
#SBATCH --output=log/OLMoE-1B-7B.lora.%j.train.slurm.log
#SBATCH --mail-type=ALL
'''
declare -a configs=(
    "4 8 False True False 2 1 4.2-1"
    "8 16 False True False 2 1 8.2-1"
    "16 32 False True False 2 1 16.2-1"
    "32 64 False True False 2 1 32.2-1"
    "4 8 False True False 2 2 4.2-2"
    "8 16 False True False 2 2 8.2-2"
    "16 32 False True False 2 2 16.2-2"
    "32 64 False True False 2 2 32.2-2"
    "4 8 False True False 4 1 4.4-1"
    "8 16 False True False 4 1 8.4-1"
    "16 32 False True False 4 1 16.4-1"
    "32 64 False True False 4 1 32.4-1"
    "4 8 False True False 4 2 4.4-2"
    "8 16 False True False 4 2 8.4-2"
    "16 32 False True False 4 2 16.4-2"
    "32 64 False True False 4 2 32.4-2"
    "4 8 False True False 4 4 4.4-4"
    "8 16 False True False 4 4 8.4-4"
    "16 32 False True False 4 4 16.4-4"
    "32 64 False True False 4 4 32.4-4"
    "4 8 False True False 8 2 4.8-2"
    "8 16 False True False 8 2 8.8-2"
    "16 32 False True False 8 2 16.8-2"
    "32 64 False True False 8 2 32.8-2"
    "4 8 False True False 8 8 4.8-8"
    "8 16 False True False 8 8 8.8-8"
    "16 32 False True False 8 8 16.8-8"
    "32 64 False True False 8 8 32.8-8"
    "4 8 True False False 1 0 4.1"
    "8 16 True False False 1 0 8.1"
    "16 32 True False False 1 0 16.1"
    "32 64 True False False 1 0 32.1"
    "4 8 True False False 2 0 4.2"
    "8 16 True False False 2 0 8.2"
    "16 32 True False False 2 0 16.2"
    "32 64 True False False 2 0 32.2"
    "4 8 True False False 4 0 4.4"
    "8 16 True False False 4 0 8.4"
    "16 32 True False False 4 0 16.4"
    "32 64 True False False 4 0 32.4"
)
'''
declare -a configs=(
    "4 8 False False True 0 0 4.e"
    "8 16 False False True 0 0 8.e"
    "16 32 False False True 0 0 16.e"
    "32 64 False False True 0 0 32.e"
)
# Function to run training
run_training() {
    local lora_r=$1
    local lora_alpha=$2
    local shared_adapter=$3
    local shared_routing_adapter=$4
    local embedded_routing_adapter=$5
    local adapter_num_experts=$6
    local num_experts_per_tok=$7
    local run_name=$8
    local gpu_id=$9
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    
    local full_run_name="OLMoE-1B-7B.lora_${run_name}"
    local output_dir="checkpoints/OLMoE-1B-7B-0924.lora_${run_name}"
    local log_file="log/${full_run_name}.train.log"
    
    echo "Starting training for ${full_run_name} on GPU ${gpu_id}"
    
    python finetune.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'math_50k.json' \
        --output_dir "${output_dir}" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 1e-5 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --shared_adapter ${shared_adapter} --shared_adapter_num ${adapter_num_experts} \
        --shared_routing_adapter ${shared_routing_adapter} --shared_routing_adapter_num_experts ${adapter_num_experts} \
        --shared_routing_adapter_num_experts_per_tok ${num_experts_per_tok} \
        --adapter_type 'LoRA' --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
        --embedded_routing_adapter ${embedded_routing_adapter} \
        --wandb_project 'peft-moe' \
        --wandb_run_name "${full_run_name}" \
        2>&1 | tee -a "${log_file}"
    
    echo "Finished training for ${full_run_name} on GPU ${gpu_id}"
}


# Run training tasks in parallel
for i in "${!configs[@]}"; do
    IFS=' ' read -ra config <<< "${configs[$i]}"
    run_training "${config[@]}" $i &
done

# Wait for all background jobs to finish
wait

echo "All training jobs completed."