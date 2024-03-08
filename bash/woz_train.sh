#!/bin/bash

function train_model {
if [[ $ngpu -gt 1 ]]; then
    master_port=`shuf -i 20000-35000 -n 1`
    prefix="-m torch.distributed.launch --nproc_per_node=$ngpu --master_port $master_port"
else
    prefix=""
fi

wf="intent"
type=woz_${wf}_dist_st
save_dir=save/${type}/2309/${model}
#save_dir=save/230614/${model}
data_prefix=data/
#dataset="simple_da_bs_multiwoz_$wf.json"
dataset="bs_multiwoz_$wf.json"
#dataset="multiwoz_$wf.json"

if [ -d ${save_dir}/${run_name} ]; then
overwrite_output_dir=False
else
mkdir -p ${save_dir}/${run_name}
overwrite_output_dir=True
fi

echo "overwrite: $overwrite_output_dir"
# overwrite_output_dir=True


# NOTE: fp16 needs to be false to not get NaN error down the line? ==> Is this true?

#python $prefix run_clm.py \
python $prefix zrun_special_tokens_clm.py \
    --model_name_or_path $model_name \
    --tokenizer_name $tokenizer_name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --do_train \
    --do_eval \
    --fp16 False \
    --block_size 1024 \
    --one_example_per_block False \
    --mask_context False \
    --run_name $run_name \
    --num_train_epochs $num_train_epochs \
    --save_strategy steps \
    --evaluation_strategy steps \
    --eval_steps $eval_steps \
    --save_steps $eval_steps \
    --logging_steps $logging_steps \
    --warmup_steps $warmup_steps \
    --remove_unused_columns False \
    --train_file ${data_prefix}${dataset} \
    --dataset_type ${dataset_type} \
    --report_to tensorboard\
    --lr_scheduler_type cosine \
    --overwrite_output_dir $overwrite_output_dir \
    --overwrite_cache False \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --output_dir ${save_dir}/${run_name} $@ \
    2>&1 | tee -a ${save_dir}/${run_name}/log.txt
}
# --max_train_samples None \
# --max_eval_samples None \

num_train_epochs=2
eval_steps=200
model="distilgpt2"
tokenizer_name="gpt2"

gradient_accumulation_steps=1
lr=5e-4
batch_size=16




logging_steps=50

ngpu=1
model_name=${model}
model_type="${model}-tf"

warmup_steps=0

# do
# 1: conv only
# 2: skip, no action in woz
# 3: wf prediction for cascade but no need, (one model)
# 4: utt prediction (oracle & cascade)

case $1 in
1)
#dataset="b1.json"
dataset_type="b1"
run_name=${dataset_type}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;

2)
# b2
#dataset="b2.json" 
dataset_type="b2"
run_name=${dataset_type}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;



3)
# workflow prediction
#dataset="wc_seed_one_convo.json"
dataset_type="wf_prediction"
run_name=${dataset_type}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 



;;


4)
# utterance prediction (workflow-conditioned)
#dataset="wc_seed_one_convo.json"
dataset_type="utt_prediction"
run_name=${dataset_type}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;


5)
echo "Do nothing, cascade model is not trained separately for now"

;;

6)
# utterance prediction (workflow-conditioned)
#dataset="wc_seed_one_convo.json"
dataset_type="utt_prediction_future_actions"
dataset="wc_seed_future_actions_one_convo.json"
run_name=${dataset_type}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;

7)
# utterance prediction (workflow-conditioned)
#dataset="wc_seed_one_convo.json"
dataset_type="kb"
run_name=${dataset_type}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 
;;
esac
