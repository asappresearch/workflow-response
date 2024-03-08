#!/bin/bash

function train_model {
if [[ $ngpu -gt 1 ]]; then
    master_port=`shuf -i 20000-35000 -n 1`
    prefix="-m torch.distributed.launch --nproc_per_node=$ngpu --master_port $master_port"
else
    prefix=""
fi



if [ -d ${save_dir}/${run_name} ]; then
overwrite_output_dir=False
else
mkdir -p ${save_dir}/${run_name}
overwrite_output_dir=True
fi

echo "overwrite: $overwrite_output_dir"
# overwrite_output_dir=True

#script=run_workflow_predictor.py
script=run_block_workflow_scorer.py
# script=run_context_block_workflow_scorer.py

python $prefix $script \
    --model_name_or_path $model_name \
    --tokenizer_name $tokenizer_name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --do_train \
    --do_eval \
    --fp16 True \
    --block_size 1024 \
    --one_example_per_block True \
    --mask_context False \
    --run_name $run_name \
    --max_steps $max_steps \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --warmup_steps $warmup_steps \
    --remove_unused_columns False \
    --train_file ${data_prefix}${dataset} \
    --report_to tensorboard\
    --lr_scheduler_type cosine \
    --overwrite_output_dir $overwrite_output_dir \
    --overwrite_cache False \
    --save_total_limit 5 \
    --output_dir ${save_dir}/${run_name} $@ \
    --data_mode $data_mode \
    --do_filter False \
    2>&1 | tee -a ${save_dir}/${run_name}/log.txt
}
# --max_train_samples None \
# --max_eval_samples None \

# case $1 in
# 1)
data_mode=random
# save_dir=save/context_block_evaluator_scorer/230817/${data_mode}/${model}
save_dir=save/woz_block_evaluator_scorer/2309/${data_mode}/${model}
# ;;

# 2)
# data_mode=random
# save_dir=save/context_block_evaluator_scorer/230817/${data_mode}/${model}
# #save_dir=save/block_evaluator_scorer/230817/${data_mode}/${model}

# esac

# workflow prediction
dataset="bs_multiwoz_intent.json"
#dataset="wc_seed_one_convo.json"
data_prefix=data/
#model="roberta-large" #"microsoft/deberta-v3-base" #"bert-base-uncased"
#batch_size=16 # 16 orig, 128 for roberta base

model="roberta-base"
batch_size=32      # 16 orig, 128 for roberta base


logging_steps=50

ngpu=1
tokenizer_name=$model #"bert-base-uncased"
model_name=${model}
model_type="${model}-tf"
num_train_epochs=1     #5 #0   #1 #20
max_steps=200
eval_steps=400
# save_steps=100
warmup_steps=0

lr=2e-5 #1e-4
gradient_accumulation_steps=1
run_name=evaluator-${model_type}-lr${lr}-bs${batch_size}-steps${max_steps}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 


#esac
