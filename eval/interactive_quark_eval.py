from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv, json
from model.constants import *
import numpy as np
import random
from tqdm import tqdm

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

eval_model_path = "./save/block_evaluator_scorer/230809/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/"

device = torch.device("cuda")
evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
 

LEN = 2#00000 #20#000 
EVAL_BATCHSIZE = 8
TRAIN_BATCHSIZE = 16
"""
Eval TODO
(1) Interactive generation with b1 as user simulator

when no none it doesnt match? ==> cause b2 was getting N/A as wf
"""

eval_data = {}
if True:
    import glob
    from Quark.main import *
    tf_paths = [ 
        ("./save/dist_st/230809/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",True, "b2"), \
    ("./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/", True, "a2"), \
    #( "./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/", False, "a2") 
    ]
    quark_paths = [ 
        ("./outputs/08-09-2023_21:43:57/model/", True, "a2"),  
    #("./outputs/08-09-2023_21:43:57/model/", False, "a2")
    ] 

    user_simulator = "./save/dist_st/230809/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/"
    


    for i, path in enumerate(tf_paths + quark_paths):

        tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(5)] + \
                        [' _TREE_TOKEN_ZERO_COMMENTS']
        oracle = path[1]
        data_type = path[-1]
        path = path[0]
        if path in tf_paths:
            policy = Policy(model_name=path, temperature=1.0, device=device, oracle=oracle, reward_mode="block")
        else:
            policy = Policy(model_name=path, temperature=1.0, device=device,
                        reward_cond=True, tree_tokens=tree_tokens, oracle=False, reward_mode="block")


        policy.tokenizer.padding_side = "left"
        policy.tokenizer.truncation_side = 'left'
        # Define PAD Token = EOS Token = 50256
        policy.tokenizer.pad_token = policy.tokenizer.eos_token
        policy.model.config.pad_token_id = policy.model.config.eos_token_id

        test_dataset = PromptDataset(dataset_type=data_type, path="data/wc_seed_one_convo.json", split="test", data_len=LEN, \
        oracle=oracle, reward_mode="block", limit_none=False, no_none=False)

        prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=False, collate_fn=prompt_collator)

        best_cat = tree_tokens[0]
        best_cat_id = policy.tokenizer.convert_tokens_to_ids(best_cat)

        def add_control_code(input_ids, attention_mask):
            input_ids = torch.cat([input_ids.new([best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
            attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
            return input_ids, attention_mask

        if not oracle:
            path = path+"_cascade"

        print(f"{path} generating..")
        quark_res = []
        for k, (input_ids, attention_mask, workflow, gt_response) in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                if "model" in quark_paths: # need this 
                   input_ids, attention_mask = add_control_code(input_ids, attention_mask)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                generated = policy.model.generate(input_ids = input_ids, attention_mask = attention_mask, \
                max_new_tokens=64, temperature=1.0, top_p=1.0,  do_sample=True)
                
                generated = generated[:, input_ids.shape[-1]:]
                responses  = policy.tokenizer.batch_decode(generated, skip_special_tokens=False)
                queries = policy.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                queries = [output.replace(policy.tokenizer.eos_token,"") for output in queries]
                if tree_tokens is not None:
                    for tt in tree_tokens:
                        queries = [output.replace(tt,"") for output in queries]

                if not oracle:
                    predicted_workflow = [ x.split(WORKFLOW_END)[0].strip() if WORKFLOW_END in x else x for x in responses]
                    responses = [ x.split(WORKFLOW_END)[1] if WORKFLOW_END in x else x for x in responses]
                    responses = [ x.split(RESPONSE)[1] if RESPONSE in x else x for x in responses ]
                    
                else:
                    predicted_workflow = [ "Oracle" for x in responses ]
                responses = [ x.split(ACTION_END)[0] for x in responses ] #

                for q,r, w,g, p in zip(queries, responses, workflow, gt_response, predicted_workflow):
                    dic = {"context": q, "response": r, "subflow":None, "true_response":g, "true_wf":w, "predicted_wf":p}
                    if False and not oracle:
                        print("="*30)
                        print(path)
                        print(dic)
                        print()
                        input() 
                    quark_res.append(dic)

        eval_data[path] = []
        
        for line in quark_res:
            context = line["context"]#+line["response_1"].strip()+"\nsystem: "
            response = line["response"].strip()
            subflow = line["subflow"]
            predicted_wf = line["predicted_wf"]
            dic = { "context": context, "response": response, "subflow":subflow}
            if "true_response" in line:
                true_response = line["true_response"]
            else:
                true_response = None
            if "true_wf" in line:
                true_wf = line["true_wf"]
            else:
                true_wf = "Oracle"
            dic["true_response"] = true_response
            dic["true_wf"] = true_wf
            dic["predicted_wf"] = predicted_wf
            eval_data[path].append(dic)
            #print(dic)
#rint(eval_data)
#print(eval_data[quark_paths[1]])

assert(len(set([len(eval_data[x]) for x in eval_data.keys()])) == 1), f"{set([len(eval_data[x]) for x in eval_data.keys()])}"
LEN = len(eval_data[list(eval_data.keys())[0]])

parallel_data = []
for i in range(LEN):
    # if i not in chosen:
    #     continue
    dic = {}
    for k, v in eval_data.items():
        
        dic["output_true_response"] = v[i]["true_response"]
        
        dic["input_"+k] = v[i]["context"]
        dic["output_"+k] = v[i]["response"]
        dic["predicted_wf_"+k] = v[i]["predicted_wf"]

        context = v[i]["context"]
        # the cascade model has the correct true_wf info, not the oracle
        if "true_wf" in v[i]: # and "oracle" in k:
            gt_wf = v[i]["true_wf"]    
        if k != "quark":
            subflow = v[i]["subflow"]
    for stoken in SPECIAL_TOKEN_SET:
        gt_wf = gt_wf.replace(stoken, "")
    dic["true_wf"] = gt_wf
    dic["subflow"] = subflow

    for k in list(eval_data.keys()) + ["true_response"]:
        context = dic["output_"+k].replace(USER, "\nClient: ")
        context = context.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")
        
        context = context.strip()

        r = "Agent: "+context
        new = []
        rsplit = r.split("\n")
        temp = []
        for z, rs in enumerate(rsplit):
            #if z == len(rsplit) -1:
            if rs.startswith("Action:"):
                pass
            elif rs.startswith("Next Action:"):
                pass
            else:
                temp.append(rs)
        r = "\n".join(temp)

        dic["context_"+k] = f"{r}\nWorkflow Action: {dic['true_wf']}"
        
    parallel_data.append(dic)
    

eval_paths = list(eval_data.keys()) + ["true_response"]
# TODO: eval loop
eval_data = {}
# for k in output_paths[1:] + quark_paths + ["true_response"]: #[ "quark","true_response"]:
#for k in output_paths+ quark_paths + ["true_response"]: #[ "quark","true_response"]:
for k in eval_paths: #[ "quark","true_response"]:
    eval_data[k] = []
    for dat in parallel_data:
        eval_data[k].append(dat)
        # print(dat)
        # input()

#print(eval_data.keys())




from bert_score import BERTScorer
import evaluate 

bertscore = BERTScorer(lang="en", rescale_with_baseline=True)
if False:
    bleurt = evaluate.load("bleurt", "BLEURT-20")
    perplexity = evaluate.load("perplexity", module_type="metric")
    meteor = evaluate.load("meteor")
    bleu = evaluate.load("bleu")

eval_result = {}

for k, v in eval_data.items():
    # if k not in quark_paths:
    #    continue
    #v = v[:10]
    count = 0
    print("Evaluating ", k)
    eval_result[k] = { "compliance": [], "block_bert": [] } #,"perplexity":[], "bertscore": [], "bleurt": [],"meteor": [], "bleu": [] }
    for batch in tqdm(list(chunk(v, size=EVAL_BATCHSIZE))): #16
        texts = [x["context_"+k] for x in batch]
        gold_flat = [ x["output_true_response"] for x in batch]
        preds_flat = [ x["output_"+k] for x in batch]
        # for t,g,p in zip(texts, gold_flat, preds_flat):
        #     print("="*30)
        #     print("text\n", t)
        #     print("\ngt", g)
        #     print("\npred",p)
        #     print()
        #     input()
        tokenized = eval_tok(texts, truncation=True, padding="longest", return_tensors="pt").to(device)

        output = evaluator(**tokenized)

        scores = output.logits.sigmoid().flatten()
        compliance = scores.tolist()


        # Block bertscore
        pred_system, gt_system = [], []
        for gt in [ x["context_true_response"] for x in batch]:
            temp = []
            for t in gt.split("\n"):
                if t.startswith("Agent:"):
                    temp += [t]
            gt_system.append(temp)
        for pred in texts:
            temp = []
            for t in pred.split("\n"):
                if t.startswith("Agent:"):
                    temp += [t]
            pred_system.append(temp)
        
        block_bert = []
        for pred, gt in zip(pred_system, gt_system):
            scores = []
            for p in pred:
                preds_flat = [ p for g in gt ]
                P, R, F1 = bertscore.score(preds_flat, gt)
                bert_scores = F1.tolist()
                bert_score = max(bert_scores)
                scores.append(bert_score)
            block_bert.append(np.average(scores))        


        for i in range(len(compliance)):
            parallel_data[count]["compliance_"+k] = compliance[i]
            parallel_data[count]["block_bert_"+k] = block_bert[i]
            count += 1

        eval_result[k]["compliance"] += compliance
        eval_result[k]["block_bert"] += block_bert
        #continue
        if False:
            P, R, F1 = bertscore.score(preds_flat, gold_flat)
            bert_scores = F1.tolist()
            bleurt_scores = bleurt.compute(predictions=preds_flat, references=gold_flat)['scores']
            perplexity_scores = perplexity.compute(predictions=preds_flat, model_id='gpt2')['perplexities']
            bleu_scores, meteor_scores = [], []
            for pred, gold in zip(preds_flat, gold_flat):
                #pred = "\n"
                results = bleu.compute(predictions=[pred], references=[gold])["bleu"] if pred.strip() != "" else 0.
                bleu_scores.append(results)
                results = meteor.compute(predictions=[pred], references=[gold])["meteor"] if pred.strip() != "" else 0.
                meteor_scores.append(results)
        
            eval_result[k]["bertscore"] +=  bert_scores
            eval_result[k]["bleurt"] += bleurt_scores
            eval_result[k]["perplexity"] += perplexity_scores
            eval_result[k]["bleu"] += bleu_scores
            eval_result[k]["meteor"] += meteor_scores

total = { k:{kk:np.average(vv) for kk,vv in v.items() } for k,v in eval_result.items() }
parallel_data = [ total ] + parallel_data

with open("./test_results/quark_eval_results.json", "w") as fh:
    json.dump(parallel_data, fh, indent = 4)

# for i in range(LEN):
#     print("="*30)
#     print("Context: ", parallel_data[i]["context"])
#     for k,v in eval_result.items():
#         print("Model: ", k)
#         print("Response: ", parallel_data[i]["context_"+k])
#         print("Prediction: ", eval_result[k][i])

#     print()

for k, v in eval_result.items():
    print(k)
    for kk, vv in v.items():
        print(f"{kk}: {np.average(vv)}")
    print()
exit()

#only_include = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "true_response" ]
only_include = eval_data.keys()

random.shuffle(parallel_data)
parallel_data = parallel_data[:100]
LEN = 100
with open("eval.csv", "w") as fh:
    header1 = [  str(i)+ "_response" for i,x in enumerate(only_include)]
    header2 = [] #[  x+ "_pred" for x in list(eval_data.keys())]
    header3 = [] #[  x+ "_score" for x in list(eval_data.keys())]
    header4 = [ "keys", "subflow"]
    header5 = [  str(i)+ "_score" for i,x in enumerate(only_include)]
    writer = csv.DictWriter(fh,  ["context", "true_wf"] + header1 + header2 + header3 + header4)
    writer.writeheader()
    
    row = {}
    for i in range(LEN):
        # if i not in chosen:
        #     continue
        # print(i, len(parallel_data))
        # print(continue)
        row["context"] = parallel_data[i]["context"]
        original_data = parallel_data[i]
        keys = list(only_include) #list(original_data.keys())#list(range(len(original_data)))
        #print("keys", keys)
        random.shuffle(keys)
        shuffled_data =  { key:original_data[key] for key in keys}
        for z, k in enumerate(keys):
        #for k,v in eval_result.items():
            row[str(z)+"_response"] = shuffled_data[k] #parallel_data[i][k]
            #row[k+"_pred"] = eval_result[k][0][i]
            #row[k+"_score"] = eval_result[k][1][i]
        row["true_wf"] = parallel_data[i]["true_wf"]
        row["keys"] = [ x.split("/")[3] if x!="true_response" and x!="quark" else x for x in keys ]
        row["subflow"] = parallel_data[i]["subflow"]
        #print()
        writer.writerow(row)
        # print(parallel_data[i]["true_wf"])
        # print(parallel_data[i]["subflow"])