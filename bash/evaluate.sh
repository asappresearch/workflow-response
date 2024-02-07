dataset="workflow-response"

use_special_tokens=False
script=run_evaluate.py
epochs=5

case $1 in
1)
save_path="./test_results/nost/b1/epoch$epochs"
model_path="./save/nost/230626/gpt2-medium/b1.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu/"
#model_path="./save/230614/gpt2-medium/b1.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/b1.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

;;
2)
save_path="./test_results/nost/b2/epoch$epochs"
#model_path="./save/230614/gpt2-medium/b2.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/nost/230626/gpt2-medium/b2.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu/"
data_path="data/b2.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

;;
3)
save_path="./test_results/nost/wf_prediction_epochs$epochs/epoch$epochs"
#model_path="./save/230614/gpt2-medium/wf_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch1-ws0-gas2-1gpu"
#model_path="./save/230615/gpt2-medium/wf_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/nost/230626/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu/"
data_path="data/wf_prediction.json"
#metrics='["bert_score", "bleurt_score", "meteor", "bleu", "exact_match"]'
metrics='[ "exact_match"]'
dataset="wf-cascade1"
;;
4)
# utt-prediction oracle case
save_path="./test_results/nost/utt_prediction_oracle_wf/epoch$epochs"
#model_path="save/230614/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/nost/230626/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu/"
data_path="data/utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

;;
5)
# For this one don't provide num samples
# utt-prediction cascading case 
save_path="./test_results/nost/utt_prediction_cascade_wf_epochs4/epoch$epochs"
#model_path="save/230614/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/nost/230626/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu/"
data_path="data/utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

dataset="wf-cascade2"
;;
6)
save_path="./test_results/nost/utt_prediction_future_actions_oracle_wf/epoch$epochs"
#model_path="save/230615/gpt2-medium/utt_prediction_future_actions.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
model_path="./save/nost/230626/gpt2-medium/utt_prediction_future_actions.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu/"
data_path="data/utt_prediction_future_actions.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'

;;
7)
save_path="./test_results/nost/wf_prediction_by_utt_predictor/epoch$epochs"
model_path="save/nost/230626/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/wf_prediction.json"
metrics='["exact_match"]'

# horrible accuracy ~50%


;;
11)
save_path="./test_results/st/b1/epoch$epochs"
model_path="./save/st/230626/gpt2-medium/b1.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/b1.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
script=run_special_tokens_evaluate.py

;;
12)
save_path="./test_results/st/b2/epoch$epochs"
model_path="./save/st/230626/gpt2-medium/b2.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/b2.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
script=run_special_tokens_evaluate.py

;;

13)
# end2end model wf eval
save_path="./test_results/st/wf_prediction_e2e/epoch$epochs"
model_path="./save/st/230626/gpt2-medium/wf_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
#model_path="save/st/230621/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/wf_prediction.json"
metrics='[ "exact_match"]'
#use_special_tokens=True
script=run_special_tokens_evaluate.py
dataset="wf-cascade1"
;;

14)
# utt-prediction oracle case
save_path="./test_results/st/utt_prediction_oracle_e2e_wf/epoch$epochs"
model_path="save/st/230626/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
script=run_special_tokens_evaluate.py
;;

15)
# end2end model utt eval
save_path="./test_results/st/utt_prediction_e2e/epoch$epochs"
model_path="save/st/230626/gpt2-medium/utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
#use_special_tokens=True
script=run_special_tokens_evaluate.py
dataset="wf-cascade2"

;;
16)
save_path="./test_results/st/utt_prediction_future_actions_oracle_wf/epoch$epochs"
model_path="save/st/230626/gpt2-medium/utt_prediction_future_actions.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/utt_prediction_future_actions.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
script=run_special_tokens_evaluate.py


;;
18)
save_path="./test_results/st/kb_utt_prediction/epoch$epochs"
model_path="save/st/230626/gpt2-medium/kb_utt_prediction.json-gpt2-medium-tf-lr1e-4-bs8-epoch$epochs-ws0-gas2-1gpu"
data_path="data/kb_utt_prediction.json"
metrics='["bert_score", "bleurt_score", "meteor", "bleu"]'
script=run_special_tokens_evaluate.py

;;
esac

python $script  --num_responses 1 --model_path $model_path --metrics "$metrics" --save_path $save_path \
 --num_samples $2 --data_path $data_path --dataset $dataset --context_len 256 --batch_size 16 #--use_special_tokens $use_special_tokens
# conlen 96 ==> 192 ==> 256
# num responses 5 ==> 1
