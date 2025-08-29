set -e
source venv/bin/activate

source step0_setup.sh $1


task_thread_id=${CUDA_VISIBLE_DEVICES}


task_words=("Wait" "Alternatively" "Check" "<|endoftext|>" "#" "%" "Answer" "Output" "Result")

task_names=("gsm8k_adv" "cruxeval_o_adv")

model_name="./MyQwen2.5-3B"

limit=2000

for task_word in ${task_words[@]}; do
    for task_name in ${task_names[@]};do
        run_sample_gt_fn ${limit} ${task_name} ${task_word} ${task_thread_id} ${model_name} "step1"
    done
done


