dataset="workflow-response"

use_special_tokens=False
script=run_special_tokens_evaluate.py 
epochs=10
data_path="data/wc_seed_one_convo.json"
data_type="workflow"

case $1 in
1)
save_path="./test_results/230809/dist_st/b1/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

data_type="b1"

;;
2)
save_path="./test_results/230809/dist_st/b2/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
data_type="b2"

;;
3)
save_path="./test_results/230809/dist_st/wf_prediction_epochs$epochs/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='[ "exact_match"]'
dataset="wf-cascade1"
data_type="wf"

;;
4)
save_path="./test_results/230809/dist_st/utt_prediction_oracle_wf/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

;;
5)
save_path="./test_results/230809/dist_st/utt_prediction_cascade/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

dataset="wf-cascade2"
;;

6)
save_path="./test_results/230809/dist_st/utt_prediction_future_actions_oracle_wf/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/utt_prediction_future_actions-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
data_path="data/wc_seed_future_actions_one_convo.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
data_type="future"

;;
7)
save_path="./test_results/230809/dist_st/kb_utt_prediction/epoch$epochs"
model_path="./save/dist_st/230809/distilgpt2/kb-distilgpt2-tf-lr5e-4-bs16-epoch$epochs-ws0-gas1-1gpu/"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
data_type="kb"

;;
esac

python $script  --num_responses 1 --model_path $model_path --metrics "$metrics" --save_path $save_path \
 --num_samples $2 --data_path $data_path --dataset $dataset --context_len 256 --batch_size 16  --response_len 64 --dataset_type $data_type #--use_special_tokens $use_special_tokens

