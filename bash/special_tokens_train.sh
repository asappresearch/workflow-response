#!/bin/bash

function train_model {
if [[ $ngpu -gt 1 ]]; then
    master_port=`shuf -i 20000-35000 -n 1`
    prefix="-m torch.distributed.launch --nproc_per_node=$ngpu --master_port $master_port"
else
    prefix=""
fi

save_dir=save/st/230626/${model}

if [ -d ${save_dir}/${run_name} ]; then
overwrite_output_dir=False
else
mkdir -p ${save_dir}/${run_name}
overwrite_output_dir=True
fi

echo "overwrite: $overwrite_output_dir"
# overwrite_output_dir=True




#python $prefix run_clm.py \
python $prefix run_special_tokens_clm.py \
    --model_name_or_path $model_name \
    --tokenizer_name $tokenizer_name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --do_train \
    --do_eval \
    --fp16 True \
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
    --report_to tensorboard\
    --lr_scheduler_type cosine \
    --overwrite_output_dir $overwrite_output_dir \
    --overwrite_cache False \
    --save_total_limit 5 \
    --load_best_model_at_end \
    --output_dir ${save_dir}/${run_name} $@ \
    2>&1 | tee -a ${save_dir}/${run_name}/log.txt
}
# --max_train_samples None \
# --max_eval_samples None \

num_train_epochs=5
eval_steps=200

case $1 in
1)
dataset="b1.json"
model="gpt2-medium"

#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name="gpt2"
model_name=${model}
model_type="${model}-tf"
#num_train_epochs=1 #20
#eval_steps=400
# save_steps=100
warmup_steps=0
batch_size=8 # 16 orig
lr=1e-4
gradient_accumulation_steps=2
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;

2)
# b2
dataset="b2.json"   
model="gpt2-medium"

#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name="gpt2"
model_name=${model}
model_type="${model}-tf"
#num_train_epochs=1 #20
eval_steps=400
# save_steps=100
warmup_steps=0
batch_size=8 # 16 orig
lr=1e-4
gradient_accumulation_steps=2
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;



3)
# workflow prediction
dataset="wf_prediction.json"
model="gpt2-medium"

#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name="gpt2"
model_name=${model}
model_type="${model}-tf"
#num_train_epochs=4 #1 #20
#eval_steps=400
# save_steps=100
warmup_steps=0
batch_size=8 # 16 orig
lr=1e-4
gradient_accumulation_steps=2
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 



;;


4)
# utterance prediction (workflow-conditioned)
dataset="utt_prediction.json"
model="gpt2-medium"

#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name="gpt2"
model_name=${model}
model_type="${model}-tf"
#num_train_epochs=1 #20
#eval_steps=400
# save_steps=100
warmup_steps=0
batch_size=8 # 16 orig
lr=1e-4
gradient_accumulation_steps=2
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;


5)
echo "Do nothing, cascade model is not trained separately for now"

;;

6)
# utterance prediction (workflow-conditioned)
dataset="utt_prediction_future_actions.json"
model="gpt2-medium"

#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name="gpt2"
model_name=${model}
model_type="${model}-tf"
#num_train_epochs=1 #20
#eval_steps=400
# save_steps=100
warmup_steps=0
batch_size=8 # 16 orig
lr=1e-4
gradient_accumulation_steps=2
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 

;;

8)
# utterance prediction (workflow-conditioned)
dataset="kb_utt_prediction.json"
model="gpt2-medium"

#save_dir=save/230614/${model}
data_prefix=data/

logging_steps=50

ngpu=1
tokenizer_name="gpt2"
model_name=${model}
model_type="${model}-tf"
#num_train_epochs=1 #20
#eval_steps=400
# save_steps=100
warmup_steps=0
batch_size=8 # 16 orig
lr=1e-4
gradient_accumulation_steps=2
run_name=${dataset}-${model_type}-lr${lr}-bs${batch_size}-epoch${num_train_epochs}-ws${warmup_steps}-gas${gradient_accumulation_steps}-${ngpu}gpu

echo $run_name
train_model --gradient_checkpointing True --gradient_accumulation_steps $gradient_accumulation_steps --log_level error --log_level_replica error --fp16 True 
;;
esac
