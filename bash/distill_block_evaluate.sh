dataset="workflow-response"

use_special_tokens=False
# script=zrun_special_tokens_evaluate.py #run_evaluate.py
script=zrun_special_tokens_block_evaluate.py #run_evaluate.py
epochs=10
data_path="data/wc_seed_one_convo.json"
data_type="workflow"

case $1 in
1)
save_path="./test_results/block/230809/dist_st/b1/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#model_path="./save/230614/gpt2-medium/b1.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
#data_path="data/b1.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
# metrics='["bert_score", "meteor", "bleu"]'

data_type="b1"

;;
2)
save_path="./test_results/block/230809/dist_st/b2/epoch$epochs"
#model_path="./save/230614/gpt2-medium/b2.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/dist_st/230809/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#model_path="./save/dist_st/230626/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch5-ws0-gas1-1gpu"
#data_path="data/b2.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
# metrics='["bert_score", "meteor", "bleu"]'
data_type="b2"

;;
3)
save_path="./test_results/block/230809/dist_st/wf_prediction_epochs$epochs/epoch$epochs"
#model_path="./save/230614/gpt2-medium/wf_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch1-ws0-gas2-1gpu"
#model_path="./save/230615/gpt2-medium/wf_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
#model_path="./save/dist_st/230626/distilgpt2/wf_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#data_path="data/wf_prediction.json"
#metrics='["bert_score", "bleurt_score", "meteor", "bleu", "exact_match"]'
metrics='[ "exact_match"]'
dataset="wf-cascade1"
data_type="wf"

;;
4)
# utt-prediction oracle case
save_path="./test_results/block/230809/dist_st/utt_prediction_oracle_wf/epoch$epochs"
#model_path="save/230614/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#data_path="data/utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
# metrics='["bert_score", "meteor", "bleu"]'

;;
5)
# For this one don't provide num samples
# utt-prediction cascading case 
save_path="./test_results/block/230809/dist_st/utt_prediction_cascade/epoch$epochs"
#model_path="save/230614/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
#model_path="./save/dist_st/230626/distilgpt2/utt_prediction.json-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#data_path="data/utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
# metrics='["bert_score", "meteor", "bleu"]'

dataset="wf-cascade2"
;;

6)
save_path="./test_results/230711/dist_st/utt_prediction_future_actions_oracle_wf/epoch$epochs"
#model_path="save/230615/gpt2-medium/utt_prediction_future_actions.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/dist_st/230626/distilgpt2/utt_prediction_future_actions-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#data_path="data/utt_prediction_future_actions.json"
data_path="data/wc_seed_future_actions_one_convo.json"
#metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
metrics='["bert_score", "meteor", "bleu"]'
data_type="future"

;;
7)
save_path="./test_results/230711/dist_st/kb_utt_prediction/epoch$epochs"
model_path="./save/dist_st/230626/distilgpt2/kb-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
#data_path="data/kb_utt_prediction.json"
#metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
metrics='["bert_score", "meteor", "bleu"]'
data_type="kb"

;;
esac

python $script  --num_responses 1 --model_path $model_path --metrics "$metrics" --save_path $save_path \
 --num_samples $2 --data_path $data_path --dataset $dataset --context_len 256 --batch_size 16  --response_len 64 --dataset_type $data_type #--use_special_tokens $use_special_tokens
# conlen 96 ==> 192 ==> 256
# num responses 5 ==> 1
