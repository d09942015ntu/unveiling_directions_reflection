source venv/bin/activate

python3 0_convert_sample.py  \
    --input_file="mydataset/cruxeval_o_adv/train.json" \
    --json_out_name="cruxeval_o_adv.json"

python3 0_convert_sample.py  \
    --input_file="mydataset/gsm8k_adv/train.json" \
    --json_out_name="gsm8k_adv.json"
