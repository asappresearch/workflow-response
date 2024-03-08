dataset="workflow-response"

use_special_tokens=False
script=run_special_tokens_evaluate.py #run_evaluate.py
#script=zrun_special_tokens_block_evaluate.py #run_evaluate.py
epochs=10

data_type="workflow"

wf="da"
type=woz_${wf}_dist_st
data_path="data/simple_da_bs_multiwoz_$wf.json"
#data_path="data/multiwoz_$wf.json"

case $1 in
1)
save_path="./test_results/$type/230726/dist_st/b1/epoch$epochs"
model_path="./save/$type/230726/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"

metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'


data_type="b1"

;;
2)
save_path="./test_results/$type/230726/dist_st/b2/epoch$epochs"
model_path="./save/$type/230726/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
data_type="b2"

;;
3)
save_path="./test_results/$type/230726/dist_st/wf_prediction_epochs$epochs/epoch$epochs"
model_path="./save/$type/230726/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='[ "exact_match"]'
dataset="wf-cascade1"
data_type="wf"

;;
4)
# utt-prediction oracle case
save_path="./test_results/$type/230726/dist_st/utt_prediction_oracle_wf/epoch$epochs"
model_path="./save/$type/230726/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

;;
5)
# For this one don't provide num samples
# utt-prediction cascading case 
save_path="./test_results/$type/230726/dist_st/utt_prediction_cascade/epoch$epochs"
model_path="./save/$type/230726/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

dataset="wf-cascade2"
cascade_datapath="./test_results/$type/230726/dist_st/wf_prediction_epochs10/epoch10/evaluation_tf.csv"
;;

6)
save_path="./test_results/$type/230724/dist_st/utt_prediction_future_actions_oracle_wf/epoch$epochs"
model_path="./save/$type/230724/distilgpt2/utt_prediction_future_actions-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
data_path="data/wc_seed_future_actions_one_convo.json"
metrics='["bert_score", "meteor", "bleu"]'
data_type="future"

;;
7)
save_path="./test_results/230711/dist_st/kb_utt_prediction/epoch$epochs"
model_path="./save/dist_st/230626/distilgpt2/kb-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "meteor", "bleu"]'
data_type="kb"

;;
esac

python $script  --num_responses 1 --model_path $model_path --metrics "$metrics" --save_path $save_path \
 --num_samples $2 --data_path $data_path --dataset $dataset --context_len 256 --batch_size 16  --response_len 64 --dataset_type $data_type  --cascade_datapath $cascade_datapath #--use_special_tokens $use_special_tokens


