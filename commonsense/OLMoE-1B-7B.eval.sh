#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:59:00
#SBATCH --mail-type=ALL


'''
# Check if the model type parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Model type parameter is required."
    echo "Usage: sbatch $0 <model_type>"
    exit 1
fi

# Get the model type from the command line argument
MODEL_TYPE=$1

# Set job name and output file dynamically
#SBATCH --job-name=OLMoE-1B-7B.${MODEL_TYPE}.eval
#SBATCH --output=log/OLMoE-1B-7B.${MODEL_TYPE}.%j.eval.slurm.log

# Define the configurations
declare -a configs=(
    "lora_4.${MODEL_TYPE}"
    "lora_8.${MODEL_TYPE}"
    "lora_16.${MODEL_TYPE}"
    "lora_32.${MODEL_TYPE}"
    "lora_64.2-1"
    "lora_64.2-2"
    "lora_4.1"
    "lora_2.e"
    "lora_128.2-2"
    "lora_128.4-4"
    "lora_128.8-8"
    "lora_64.4-1"
    "lora_64.e"
    "lora_64.4-4"
    "lora_64.8-8"
    "lora_64.16-8"
    "lora_4.1-1"
    "lora_8.1-1"
    "lora_16.1-1"
    "lora_32.1-1"
    "parallel_4.1-1"
    "parallel_8.1-1"
    "parallel_16.1-1"
    "parallel_32.1-1"
    "parallel_4.2-2"
    "parallel_8.2-2"
    "parallel_16.2-2"
    "parallel_32.2-2"
)
'''
declare -a configs=(
    "parallel_4.4-2"
    "parallel_8.4-2"
    "parallel_16.4-2"
    "parallel_32.4-2"
)

# Function to run evaluation
run_evaluation() {
    local config=$1
    local gpu_id=$2
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    for dataset in "boolq" "piqa" "social_i_qa" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa" "hellaswag"; do
        python commonsense_evaluate.py \
            --dataset $dataset \
            --base_model allenai/OLMoE-1B-7B-0924 \
            --peft_model checkpoints/OLMoE-1B-7B-0924.${config} \
            --name OLMoE-1B-7B.${config} \
            --batch_size 16 --max_new_tokens 4 \
            | tee -a log/OLMoE-1B-7B.${config}.eval.${dataset}.log
    done
}

# Run evaluations in parallel
for i in {0..3}; do
    run_evaluation "${configs[$i]}" $i &
done

# Wait for all background jobs to finish
wait