source venv/bin/activate


source step0_setup.sh $1

task_names=("gsm8k_adv" "cruxeval_o_adv")

for task_name in ${task_names[@]}; do

  python3 step0_convert_sample.py  \
      --input_file="mydataset/${task_name}/train.json" \
      --json_out_name="${task_name}.json"

done