set -e
source venv/bin/activate

DEFAULT_CUDA_VISIBLE_DEVICES=0
# Check if a specific CUDA device is given as a command-line argument
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
# Export the CUDA_VISIBLE_DEVICES variable
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE

limit=200
word_limit=5

task_names=("gsm8k_adv" "cruxeval_o_adv")

for task in ${task_name[@]}; do

    python3 2_build_steering_vec.py \
        --input_file="visualize/${task}/step0/${task}.json" \
        --model_dir="mymodels/MyQwen2.5-3B" \
        --output_dir="visualize/${task}/qwen3b/step2/steer_${limit}_21" \
        --wait_token_1 "Wait" "Alternatively" "Check" \
        --wait_token_2 "Answer" "Result" "Output" \
        --limit ${limit}

    python3 2_build_steering_vec.py \
        --input_file="visualize/${task}/step0/${task}.json" \
        --model_dir="mymodels/MyQwen2.5-3B" \
        --output_dir="visualize/${task}/qwen3b/step2/steer_${limit}_20" \
        --wait_token_1 "Wait" "Alternatively" "Check" \
        --wait_token_2 "<|endoftext|>" "%" "#" \
        --limit ${limit}

    python3 2_build_steering_vec.py \
        --input_file="visualize/${task}/step0/.json" \
        --model_dir="mymodels/MyQwen2.5-3B" \
        --output_dir="visualize/${task}/qwen3b/step2/steer_${limit}_10" \
        --wait_token_1 "<|endoftext|>" "%" "#" \
        --wait_token_2 "Answer" "Result" "Output" \
        --limit ${limit}

    python3 2_build_steering_vec.py \
        --input_file="visualize/${task}/step0/${task}.json" \
        --model_dir="mymodels/MyQwen2.5-3B" \
        --output_dir="visualize/${task}/qwen3b/step2/steer_baseline" \
        --wait_token_1 "Wait" "Alternatively" "Check" \
        --wait_token_2 ""  \
        --limit ${limit}

    python3 2_reselect.py \
        --input_dir="visualize/${task}/qwen3b/step2/steer_${limit}_21" \
        --word_limit ${word_limit}

    python3 2_reselect.py \
        --input_dir="visualize/${task}/qwen3b/step2/steer_${limit}_20" \
        --word_limit ${word_limit}

    python3 2_reselect.py \
        --input_dir="visualize/${task}/qwen3b/step2/steer_${limit}_10" \
        --word_limit ${word_limit}

    python3 2_reselect.py \
        --input_dir="visualize/${task}/qwen3b/step2/steer_baseline" \
        --word_limit ${word_limit}

done


