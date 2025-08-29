#!/bin/bash




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
    out_path=$4
    model_name=$5
    model_args_str=$6

    run_str="
    python3 lm_eval --model hf \
        --model_args pretrained=./mymodels/${model_name},dtype=float,trust_remote_code=True,local_files_only=True${model_args_str} \
        --include_path mytasks \
        --tasks ${task} \
        --device cuda:0 \
        --batch_size auto:1 \
        --gen_kwargs max_new_tokens=256 \
        --output_path ${out_path} \
        --limit ${limit} \
        --seed 0 \
        --log_samples
    "
    echo ${run_str}
    ${run_str}

}


run_sample_gt_fn() {

    limit=$1
    task_name=$2
    task_word=$3
    task_thread_id=$4
    model_name=$5
    step_dest=$6

    task="${task_name}_${task_thread_id}"

    echo "run task:${task}_${task_word}"
    echo ${task_word} > "./mytasks/${task_name}/wait_token_${task_thread_id}.txt"
    cat "./mytasks/${task_name}/wait_token_${task_thread_id}.txt"

    out_dir="gt_${limit}_${task_word}"
    out_path="visualize/${task_name}/${model_name}/${step_dest}/${out_dir}"
    if [ ! -d "$out_path" ]; then
      echo "$out_path does not exist."
      run_sample_fn ${limit} ${task} ${task_name} ${out_path} ${model_name} ""
    fi

}


run_sample_steering_fn (){

    limit=$1
    task_name=$2
    task_word=$3
    task_thread_id=$4
    model_name=$5
    step_dest=$6

    l=$7
    c_scale=$8
    steer_file=$9

    control_type="selected"
    task="${task_name}_${task_thread_id}"

    echo "run task:${task}_${task_word}"
    echo ${task_word} > "./mytasks/${task_name}/wait_token_${task_thread_id}.txt"
    cat "./mytasks/${task_name}/wait_token_${task_thread_id}.txt"

    echo "run task:${task}"
    model_args="steering_vec_file=${steer_file},control_type=selected,control_num=${l},control_scale=${c_scale}"

    out_dir="s${c_scale}_${task_word}_${limit}_${s_suffix}_t${control_type}_l${l}"
    out_path="visualize/${task_name}/${model_name}/${step_dest}/${out_dir}"
    if [ ! -d "$out_path" ]; then
        echo "$out_path does not exist."
        run_sample_fn ${limit} ${task} ${task_name} ${out_path} ${model_name} ","${model_args}
    fi

}

test_fn(){
  content=$1
  echo "content:${content}"
}