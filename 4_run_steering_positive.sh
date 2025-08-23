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
      --output_path ${out_dir} \
      --limit ${limit} \
      --seed 0 \
      --log_samples
  "
  echo ${run_str}
  ${run_str}

}


limit=2000
s_limit=200

task_name="gsm8k_adv"

thread_id=1
wait_token_fid=${thread_id}

model_path="./mymodels/MyQwen2.5-3B"

fidxs=(0 1 2 4 6 8 10 12 14 16 18 20 22 25 28 31 33 35)

steering_suffixs=("20" "21")


task_suffixs2=("answer" "eos")

c_scales=(1)

name_scale="${c_scale//./}"

for l in ${layers[@]}; do
	for task_suffix in ${task_suffixs2[@]}; do

          for s_suffix in ${steering_suffixs[@]}; do
            for c_scale in ${c_scales[@]};do
              control_type="selected"
              task="${task_name}_${task_suffix}"
              echo "run task:${task}"
              steering_vec_file="visualize/${task_name}/step2/steer_${limit_s}_${s_suffix}/seed_avg.json"
              model_args="steering_vec_file=${steering_vec_file},control_type=selected,control_num=${l},control_scale=${c_scale}"

              out_dir="s${c_scale}_${task_suffix}_${limit}_${s_suffix}_t${control_type}_l${l}"
              full_output_path="visualize/${task_name}/step4/${out_dir}"
              if [ ! -d "$full_output_path" ]; then
                echo "$full_output_path does not exist."
                run_sample_fn ${limit} ${task} ${task_name} ${full_output_path} ${model_path} ""
              fi

            done
          done
    done
done

