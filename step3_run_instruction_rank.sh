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

# -1: Input Embeddings

steer_types=(steer_${s_limit}_21 steer_${s_limit}_20)
step_source="step2"
step_dest="step3"
for steer_type in ${steer_types}; do
    input_dir="visualize/${task_name}/${model_name}/${step_source}/${steer_type}"

    for layer in ${layers[@]}; do
        echo "-------------------------------------"
        echo "layer :${layer}"

        for steer_type in ${steer_types[@]}; do

            task_words=($(cat "${input_dir}/word_${layer}.txt"))

            for task_word in ${task_words[@]}; do

                run_sample_gt_fn ${limit} ${task_name} ${task_word} ${task_thread_id} ${model_name} ${step_dest}

            done
        done
    done
done