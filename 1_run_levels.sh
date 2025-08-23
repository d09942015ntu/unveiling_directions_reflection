source venv/bin/activate

DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE

run_sample_fn() {

  limit=$1
  task=$2
  task_name=$3
  out_dir=$4
  model_path=$5
  model_args_str=$6

  run_str="
  python3 lm_eval --model hf \
      --model_args pretrained=${model_path},dtype=float,trust_remote_code=True,local_files_only=True${model_args_str} \
      --include_path mytasks \
      --tasks ${task} \
      --device cuda:0 \
      --batch_size auto:1 \
      --gen_kwargs max_new_tokens=256 \
      --output_path visualize/${task_name}/step2_1/${out_dir} \
      --limit ${limit} \
      --seed 0 \
      --log_samples
  "
  echo ${run_str}
  ${run_str}

}


limit=2000

task_thread_id=0

task_suffixs=("Wait" "Alternatively" "Check" "<|endoftext|>" "#" "%" "Answer" "Output" "Result")

task_names=("gsm8k_adv" "cruxeval_o_adv")

model_path="./mymodels/MyQwen2.5-3B"

for task_suffix in ${task_suffixs[@]}; do
    for task_name in ${task_names[@]};do
        task="${task_name}_${task_thread_id}"
        echo "run task:${task}_${task_suffix}"
        echo ${task_suffix} > "./mytasks/${task_name}/wait_token_${task_thread_id}.txt"
        cat "./mytasks/${task_name}/wait_token_${task_thread_id}.txt"
        out_dir="gt_${limit}_${task_suffix}"
        run_sample_fn ${limit} ${task} ${task_name} ${out_dir} ${model_path} ""
    done
done
