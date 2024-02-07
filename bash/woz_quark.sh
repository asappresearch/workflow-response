#!/bin/bash

python run_a2_train.py  --temperature 0.5 --total_episodes 160000 \
    --dataset ./data/bs_multiwoz_intent.json \
    --reward_model_path ./save/woz_block_evaluator_scorer/2309/random/evaluator-roberta-base-tf-lr2e-5-bs32-steps200-ws0-gas1-1gpu/ \
    --init_model ./save/woz_intent_dist_st/2309/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/ \
    --ref_model ./save/woz_intent_dist_st/2309/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/ \
    --only_standard_data False