set -e

source venv/bin/activate

source step0_setup.sh $1

limit=2000
s_limit=200

task_name="gsm8k_adv"

thread_id=${CUDA_VISIBLE_DEVICES}
task_thread_id=${thread_id}

model_name="MyQwen2.5-3B"

layers=(0 1 2 4 6 8 10 12 14 16 18 20 22 25 28 31 33 35)

steering_suffixs=("20" "10")

task_words2=("Wait" "<|endoftext|>")

step_source="step2"
step_dest="step4"
c_scale=-1
task_thread_id=${CUDA_VISIBLE_DEVICES}

for l in ${layers[@]}; do
    for task_word in ${task_words2[@]}; do
        for s_suffix in ${steering_suffixs[@]}; do

            steering_vec_file="visualize/${task_name}/${model_name}/${step_source}/steer_${s_limit}_${s_suffix}/seed_avg.json"
            run_sample_steering_fn ${limit} ${task_name} ${task_word} ${task_thread_id} \
              ${model_name} ${step_dest} ${l} ${c_scale} ${steering_vec_file}

        done
    done
done

