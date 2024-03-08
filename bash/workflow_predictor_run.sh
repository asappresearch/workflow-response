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
script=run_workflow_scorer.py


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
    --num_train_epochs $num_train_epochs \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps $eval_steps \
    --logging_steps $logging_steps \
    --warmup_steps $warmup_steps \
    --remove_unused_columns False \
    --train_file ${data_prefix}${train} \
    --validation_file ${data_prefix}${val} \
    --report_to tensorboard\
    --lr_scheduler_type cosine \
    --overwrite_output_dir $overwrite_output_dir \
    --overwrite_cache False \
    --save_total_limit 5 \
    --output_dir ${save_dir}/${run_name} $@ \
    --data_mode $data_mode
    2>&1 | tee -a ${save_dir}/${run_name}/log.txt
}
# --max_train_samples None \
# --max_eval_samples None \

case $1 in
1)
data_mode=neg
save_dir=save/evaluator_scorer/230801/${data_mode}/${model}

;;

2)
data_mode=random
save_dir=save/evaluator_scorer/230801/${data_mode}/${model}

esac

# workflow prediction
#dataset="wc_seed_one_convo.json"
#dataset=::
#train="gen_neg_para_train_3.json" 
train="generated_negatives_train_10000_with_none_with_random.json"
#val="gen_neg_para_dev_3.json" 
val="generated_negatives_dev_1000_with_none_with_random.json"

#model="roberta-large" #"microsoft/deberta-v3-base" #"bert-base-uncased"
#batch_size=16 # 16 orig, 128 for roberta base

model="roberta-base"
batch_size=16      # 16 orig, 128 for roberta base

# model="roberta-large"
# batch_size=8      # 16 orig, 128 for roberta base


#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name=$model #"bert-base-uncased"
model_name=${model}
model_type="${model}-tf"
num_train_epochs=$2     #5 #0   #1 #20
eval_steps=400
# save_steps=100
warmup_steps=0

lr=2e-5 #1e-4
gradient_accumulation_steps=1
run_name=evaluator-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 


#esac
