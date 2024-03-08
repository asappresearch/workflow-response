from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv, json
from model.constants import *
import numpy as np
import random
from tqdm import tqdm
import re

from transformers import set_seed
from Quark.utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness

import fire
import glob
from Quark.main import *
from bert_score import BERTScorer
import evaluate 

from model.call_openai import batch_evaluate_fluency_coherence, batch_llm_generate
import asyncio
from model.retrieve_openai import *

def main(
    LEN = 20000,
    block = True,
    use_scorer = False,
    temperature = 1.0,
    prefix = "block",
    context_scoring = False,
    only_standard_data = True
    ):
    num_len = LEN
    set_seed(42)
    """
    TODO:
    """
    def chunk(l, size=16):
        # looping till length l
        for i in range(0, len(l), size): 
            yield l[i:i + size]

    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(5)] + \
                            [' _TREE_TOKEN_ZERO_COMMENTS']
    best_cat = tree_tokens[0]

    def add_control_code(input_ids, attention_mask, best_cat_id):
        input_ids = torch.cat([input_ids.new([best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        return input_ids, attention_mask


    def sub(string, start, end):
        start_substring = start
        end_substring = end
        final_string = string
        while start_substring in final_string and end_substring in final_string:
            start_index = final_string.find(start_substring)
            end_index = final_string.find(end_substring)
            if start_index < end_index:
                final_string = final_string[:start_index] + final_string[end_index + len(end_substring):]
            else:
                break
        return final_string

    def format_for_llm(string):
        context = string.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")
        context = context.replace(USER, "\nClient: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")

        #r = "Agent: "+context
        r = context
        #new = []
        rsplit = r.split("\n")
        temp = []
        for z, rs in enumerate(rsplit):
            if rs.startswith("Next Action:"):
                pass
            else:
                temp.append(rs)
        r = "\n".join(temp)

        context = r

        return context 

    if context_scoring:
        eval_model_path = "./save/context_block_evaluator_scorer/230817/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/"
    else:
        eval_model_path = "./save/block_evaluator_scorer/230809/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/"

    device = torch.device("cuda")
    evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
    eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
    

    LEN = num_len #20#0#000 #200000 #20#000 
    EVAL_BATCHSIZE = 8
    TRAIN_BATCHSIZE = 16
    split = "test" 
    """
    Eval TODO:
    1. get llm prompting to work with this
    2. Why so few examples are filtered?
    cascade determinism wrt oracle model
    when it predicted correct wf, why isn't the rest of the output same as oracle's?
    """

    eval_data = {}
    if True:
        path = "llm-gpt-3.5-turbo-16k-0613"
        user_path = "./save/dist_st/230809/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/"
        ref_policy = Policy(model_name=user_path, temperature=1.0, device=device, oracle=False, reward_mode="block") # changing user temperature

        if True:
            oracle = True #path[1]
            data_type = "b2" #path[-1]

            data_path = "data/wc_seed_one_convo.json"

            test_dataset = PromptDataset(dataset_type=data_type, path=data_path, split=split, data_len=LEN, \
            oracle=oracle, reward_mode="block", limit_none=False, no_none=False, only_standard_data=only_standard_data)

            train = PromptDataset(dataset_type=data_type, path=data_path, split="dev", data_len=200000, \
            oracle=oracle, reward_mode="block", limit_none=False, no_none=False, only_standard_data=only_standard_data)

            dev = PromptDataset(dataset_type=data_type, path=data_path, split="train", data_len=200000, \
            oracle=oracle, reward_mode="block", limit_none=False, no_none=False, only_standard_data=only_standard_data)

            example_prompts = train.prompts + dev.prompts
            example_workflows = train.workflows + dev.workflows
            example_gt_responses = train.gt_responses + dev.gt_responses
            example_subflows = train.subflows + dev.subflows
            example_gt_responses = [ "Agent: " + format_for_llm(x) for x in example_gt_responses]

            examples = []
            for ep, ew, eg, es in zip(example_prompts, example_workflows, example_gt_responses, example_subflows):
                examples += [ [ es, ew, ep, eg]]


            prompt_collator = PromptCollator(tokenizer=ref_policy.tokenizer)
            prompt_collator.return_subflow = True

            test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=False, collate_fn=prompt_collator)
        
            input_contexts = []
            workflows, subflows = [], []
            guidelines = []
            for k, (input_ids, attention_mask, workflow, gt_response, subflow) in enumerate(tqdm(test_dataloader)):
                input_context = ref_policy.tokenizer.batch_decode(input_ids)
                input_context =  [ ic.replace(ref_policy.tokenizer.pad_token, "") for ic in input_context ]
                guidelines += [ retrieve_guideline_text_action(s, w) for s,w in zip(subflow, workflow) ]
                input_contexts += input_context
                workflows += workflow
                subflows += subflow
            MAX_INTERACTION = 3
            
            generated = ["" for x in input_contexts]
            for i in range(MAX_INTERACTION):
                llm_input_contexts = [ format_for_llm(x) for x in input_contexts ]
                llm_outputs = asyncio.run(batch_llm_generate(llm_input_contexts, guidelines, workflows, subflows, examples)) #, ref_policy, input_ids, attention_mask, top_p=1.0, reward_cond=reward_cond, oracle = oracle, data_type=data_type)
                llm_outputs = [x.lower() for x in llm_outputs]
                user_policy_input_contexts = [ x + y  + USER for x,y in zip(input_contexts, llm_outputs)]

                generated = [ g  + y for g, y in zip(generated, llm_outputs)]
                generated = [ g.strip() for g in generated ]
                
                if i == MAX_INTERACTION -1:
                    break

                user_responses = []
                for batch in chunk(user_policy_input_contexts, size=TRAIN_BATCHSIZE):
                    #ref_policy.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")#.to(device)
                    #user_query =  [ q + r  + USER  for q,r in zip(agent_query, agent_response) ]
                    #must remove WORKFLOW, ACTION: only when user model is b1
                    temp = []
                    for uq in batch:
                        result = sub(uq, ACTION, ACTION_END)
                        result = sub(result, WORKFLOW, WORKFLOW_END)
                        #print(result)
                        temp.append(result)
                    ref_user_query = temp

                    # ref_user_query = user_query # when user model is a1 
                    with torch.no_grad():
                        encodings_dict = ref_policy.tokenizer(ref_user_query, return_tensors="pt", padding=True)
                        input_ids = encodings_dict['input_ids']
                        attention_mask = encodings_dict['attention_mask']

                        user_rollouts = ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=1.0, max_len = 64)
                        user_response = user_rollouts['response/text']
                        user_response = [x.split(USER_END)[0].lower() for x in user_response]
                        #print("User response:", user_response)
                        user_responses += user_response
                input_contexts = [ x+  y for x,y in zip(user_policy_input_contexts, user_responses )]
                generated = [  x + "\nClient: " + y + "\nAgent: " for x, y in zip(generated, user_responses)]
                
            quark_res = []

            for k, (input_ids, attention_mask, workflow, gt_response, subflow) in enumerate(tqdm(test_dataloader)):
                contexts = ref_policy.tokenizer.batch_decode(input_ids)
                for ic , w,g, s in zip(contexts,  workflow, gt_response, subflow):
                    guideline = retrieve_guideline_text_action(s, w)
                    
                    #print("Q:", q)
                    q = ic.replace(ref_policy.tokenizer.pad_token, "")
                    dic = {"context": q, "response": None, "subflow":s, "true_response":g, "true_wf":w, "predicted_wf":None, "guideline": guideline}

                    quark_res.append(dic)

            assert len(quark_res) == len(generated), f"{len(quark_res)} != {len(generated)}"
            
            for g, dic in zip(generated, quark_res):
                dic["response"] = g
                #print(dic)

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
                dic["guideline"] = line["guideline"]
                eval_data[path].append(dic)


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
            
            guideline = v[i]["guideline"]
        context = context.replace(USER, "\nClient: ")
        context = context.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")
        
        context = context.strip()

        dic["context"] = context

        for stoken in SPECIAL_TOKEN_SET:
            gt_wf = gt_wf.replace(stoken, "")
        dic["true_wf"] = gt_wf
        dic["subflow"] = subflow
        
        dic["guideline"] = guideline

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

            if context_scoring:
                # f"{context}\n{pr}\nWorkflow Action: {pos_act}" for context, pr, pos_act
                dic["context_"+k] = f"{dic['context']} {dic['true_wf']}\n{r}\nWorkflow Action: {dic['true_wf']}"
                # print("="*30)
                # print(dic["context_"+k])
                # print("-"*30)
            else:
                dic["context_"+k] = f"{r}\nWorkflow Action: {dic['true_wf']}"
        
        parallel_data.append(dic)


    """
    TODO get a way to just read the already generated ones from test_result/*.json
    """
        

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






    bertscore = BERTScorer(lang="en", rescale_with_baseline=True)
    if not block:
        bleurt = evaluate.load("bleurt", "BLEURT-20")
        #perplexity = evaluate.load("perplexity", module_type="metric")
        meteor = evaluate.load("meteor")
        bleu = evaluate.load("bleu")

    eval_result = {}

    for k, v in eval_data.items():
        # if k not in quark_paths:
        #    continue
        #v = v[:10]
        count = 0
        print("Evaluating ", k)
        eval_result[k] = { "compliance": [],  "block_bert": [],
        "wf_accuracy":[], "dist_1":[], "dist_2":[], "dist_3":[],
        "meteor": [],
        "bleu": [],
        "bleurt": [],
        "bertscore": [],
        "llm_fluency": [],
        "llm_compliance": []
        } #,"perplexity":[], "bertscore": [], "bleurt": [],"meteor": [], "bleu": [] }
        for batch in tqdm(list(chunk(v, size=EVAL_BATCHSIZE))): #16
            texts = [x["context_"+k] for x in batch]
            gold_flat = [ x["output_true_response"] for x in batch]
            preds_flat = [ x["output_"+k] for x in batch]

            #batch_evaluate_fluency_coherence
            # batch_evaluate_fluency_coherence(prompts, responses, workflows, subflows):
            prompts = [x["context"] for x in batch]
            responses = preds_flat
            workflows = [ x["true_wf"] for x in batch]
            subflows = [ x["subflow"] for x in batch]

            
            # Block bertscore
            pred_system, gt_system = [], []
            for gt in [ x["context_true_response"] for x in batch]:
                temp = []
                for t in gt.split("\n"):
                    if t.startswith("Agent:"):
                        temp += [t.strip("Agent:").strip()]
                gt_system.append(temp)
            for pred in texts:
                temp = []
                for t in pred.split("\n"):
                    if t.startswith("Agent:"):
                        temp += [t.strip("Agent:").strip()]
                pred_system.append(temp)

            wf_correct = []
            if "cascade" in k:
                for predicted_wf, gt_wf in zip([ x["predicted_wf_"+k] for x in batch], [ x["true_wf"] for x in batch]):
                    wf_correct.append(float(predicted_wf == gt_wf))

            if block:

                #llm_fluency = asyncio.run(batch_evaluate_fluency_coherence(prompts, responses, workflows, subflows))
                # llm_compliance = batch_evaluate_compliance(prompts, responses, workflows, subflows)
            

                tokenized = eval_tok(texts, truncation=True, padding="longest", return_tensors="pt").to(device)

                output = evaluator(**tokenized)

                scores = output.logits.sigmoid().flatten()
                compliance = scores.tolist()
                block_bert = [] #, meteor, bleu, bleurt = [], [ ], [], []
                dist1s, dist2s, dist3s = [], [], []
                    
                for pred, gt in zip(pred_system, gt_system):
                    scores = []
                    for p in pred:
                        preds_flat = [ p for g in gt ]
                        P, R, F1 = bertscore.score(preds_flat, gt)
                        bert_scores = F1.tolist()
                        bert_score = max(bert_scores)
                        scores.append(bert_score)
                    block_bert.append(np.average(scores))        
                
                    dist_1, dist_2, dist_3 = distinctness(pred)
                    dist1s += [dist_1]
                    dist2s += [dist_2]
                    dist3s += [dist_3]


                for i in range(len(compliance)):
                    parallel_data[count]["compliance_"+k] = compliance[i]
                    #parallel_data[count]["llm_fluency_"+k] = llm_fluency[i]
                    count += 1

                eval_result[k]["compliance"] += compliance
                eval_result[k]["block_bert"] += block_bert
                eval_result[k]["wf_accuracy"] += wf_correct
                eval_result[k]["dist_3"] += dist3s
                #eval_result[k]["llm_fluency"] += llm_fluency


            #continue
            if not block:
                preds_flat = [ x[0] for x in pred_system]
                gold_flat = [x[0] for x in gt_system]
                P, R, F1 = bertscore.score(preds_flat, gold_flat)
                bert_scores = F1.tolist()
                bleurt_scores = bleurt.compute(predictions=preds_flat, references=gold_flat)['scores']
                #perplexity_scores = perplexity.compute(predictions=preds_flat, model_id='gpt2')['perplexities']
                bleu_scores, meteor_scores = [], []
                for pred, gold in zip(preds_flat, gold_flat):
                    #pred = "\n"
                    results = bleu.compute(predictions=[pred], references=[gold])["bleu"] if pred.strip() != "" else 0.
                    bleu_scores.append(results)
                    results = meteor.compute(predictions=[pred], references=[gold])["meteor"] if pred.strip() != "" else 0.
                    meteor_scores.append(results)
            
                eval_result[k]["bertscore"] +=  bert_scores
                eval_result[k]["bleurt"] += bleurt_scores
                #eval_result[k]["perplexity"] += perplexity_scores
                eval_result[k]["bleu"] += bleu_scores
                eval_result[k]["meteor"] += meteor_scores

    total = { k:{kk:np.average(vv) for kk,vv in v.items() } for k,v in eval_result.items() }

    write_data = [ total ] + parallel_data

    with open(f"./test_results/llm_{prefix}_{split}_quark_eval_results.json", "w") as fh:
        json.dump(write_data, fh, indent = 4)



    for k, v in eval_result.items():
        print(k)
        for kk, vv in v.items():
            print(f"{kk}: {np.average(vv)}")
        print()
    #exit()

    #only_include = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "true_response" ]
    only_include = eval_result.keys() #parallel_data.keys() #eval_data.keys()

    #random.shuffle(parallel_data)
    #parallel_data = parallel_data[:100]
    #LEN = 100

    def cleanup(string):
        context = string.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")
        context = context.replace(USER, "\nClient: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")

        r = "Agent: "+context
        #new = []
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

        context = r

        return context 

    with open(f"./test_results/{prefix}quark_eval.csv", "w") as fh:
        header1 = [  [str(i)+ "_response", str(i)+ "_score"] for i,x in enumerate(only_include)]
        header1 = [ x for y in header1 for x in y ]
        header2 = [ ] #[  x+ "_pred" for x in list(eval_data.keys())]
        header3 = [] #[  x+ "_score" for x in list(eval_data.keys())]
        header4 = [ "keys", ]
        header5 = [  str(i)+ "_model_score" for i,x in enumerate(only_include)]
        header6 = [  str(i)+ "_score_context" for i,x in enumerate(only_include)]
        writer = csv.DictWriter(fh,  ["context", "subflow", "true_wf", "guideline"] + header1 + header2 + header3 + header4 + header5 + header6)
        writer.writeheader()
        
        row = {}
        for i in range(LEN):
            # if i not in chosen:
            #     continue
            # print(i, len(parallel_data))
            # print(continue)
            row["context"] = cleanup(parallel_data[i]["context"])
            original_data = parallel_data[i]
            keys = list(only_include) #list(original_data.keys())#list(range(len(original_data)))
            #print("keys", keys)
            random.shuffle(keys)
            shuffled_data =  { key:original_data["output_"+ key] for key in keys}
            for z, k in enumerate(keys):
            #for k,v in eval_result.items():
                row[str(z)+"_response"] = cleanup(shuffled_data[k]) #parallel_data[i][k]
                row[str(z)+"_model_score"] = parallel_data[i]["compliance_"+str(k)]
                row[str(z)+"_score_context"] = parallel_data[i]["context_"+str(k)]
                #row[k+"_pred"] = eval_result[k][0][i]
                #row[k+"_score"] = eval_result[k][1][i]
            row["true_wf"] = parallel_data[i]["true_wf"]
            #row["keys"] = [ x.split("/")[3] if x!="true_response" and x!="quark" else x for x in keys ]
            row["keys"] = [  x for x in keys ]
            row["subflow"] = parallel_data[i]["subflow"]
            row["guideline"] = parallel_data[i]["guideline"]
            
            #print()
            writer.writerow(row)
            # print(parallel_data[i]["true_wf"])
            # print(parallel_data[i]["subflow"])


if __name__ == "__main__":
    fire.Fire(main)
