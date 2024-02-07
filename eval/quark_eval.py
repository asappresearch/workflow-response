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

from model.call_openai import batch_evaluate_fluency_coherence
import asyncio
from model.retrieve_openai import *

def main(
    LEN = 20000,
    block = True,
    use_scorer = False,
    temperature = 1.0,
    prefix = "woz",
    context_scoring = False,
    only_standard_data = False
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

    def interactive_sample(policy, ref_policy, input_ids, attention_mask, top_p=1.0, max_len=64, reward_cond=True, oracle=True, data_type="a2"): #step, depleted=False):

        PRINT = False
        MAX_INTERACTION = 1 #3 #1 for multiwoz
        best_cat_id = policy.tokenizer.convert_tokens_to_ids(best_cat)

        for i in range(MAX_INTERACTION):
            context_len = 512
            input_ids = input_ids[:, :context_len] # not ideal
            attention_mask = attention_mask[:, :context_len] # not ideal, can lead to bigger loss in context (no left padding here)

            if reward_cond:
                input_ids, attention_mask = add_control_code(input_ids, attention_mask, best_cat_id)
                        
            agent_rollouts = policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=top_p, max_len = max_len)
            
            agent_response = agent_rollouts['response/text']

            agent_response = [ x.split(USER)[0] for x in agent_response ]
            
            temp = []
            for r in agent_response:
                if ACTION in r and ACTION_END in r.split(ACTION)[1]:
                    r = r.split(ACTION_END)[0] +ACTION_END
                else:
                    r = r.split(ACTION)[0]
                    r = RESPONSE_END.join(r.split(RESPONSE_END)[:-1]) + RESPONSE_END
                temp.append(r)
            agent_response = temp


            agent_query = agent_rollouts['query/text']

            if i == 0 : #sor (i ==1 and not oracle): # condition remove later
                first_query = agent_query
            
            if i == MAX_INTERACTION -1:
                # no need to generate user response
                if False:
                    for a in agent_response:
                        print("="*30)
                        print("Iter:",i)
                        print(a)
                break

            user_query =  [ q + r  + USER  for q,r in zip(agent_query, agent_response) ]
            #must remove WORKFLOW, ACTION: only when user model is b1 Note: this? TODO:
            temp = []
            for uq in user_query:
                result = sub(uq, ACTION, ACTION_END)
                result = sub(result, WORKFLOW, WORKFLOW_END)
                #print(result)
                temp.append(result)
            ref_user_query = temp

            # ref_user_query = user_query # when user model is a1 

            encodings_dict = ref_policy.tokenizer(ref_user_query, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

            user_rollouts = ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=top_p, max_len = max_len)
            user_response = user_rollouts['response/text']
            #user_query = user_rollouts['query/text']
            #user_query, user_response = self.generate(self.ref_policy, input_ids, attention_mask)

            for stoken in [RESPONSE, WORKFLOW, ACTION, CONTEXT]:
                user_response = [ x.split(stoken)[0] for x in user_response ]
            #prompt = self.decode(rollouts['query/input_ids'][:, 1:])
            if PRINT:
                for a,u in zip(agent_response, user_response):
                    print("="*30)
                    print("Iter:",i)
                    print(a)
                    print(u)
                    print()

            if data_type == "b2":
                agent_query = [ q + r + RESPONSE for q,r in zip(user_query, user_response) ]
            else:
                agent_query = [ q + r + WORKFLOW for q,r in zip(user_query, user_response) ]

            encodings_dict = policy.tokenizer(agent_query, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

        # TODO: need to remake the roll out
        full_texts = [ q + r for q, r in zip(agent_query, agent_response)]

        query, response = [], []
        for i,f in enumerate(full_texts):
            assert(f.startswith(first_query[i])), f"something's wrong {f} // {first_query[i]}"
            q = first_query[i]
            r = f[len(first_query[i]):]
            if ACTION in r:
                if ACTION_END in r:
                    try:
                        r = r.split(ACTION)[0] + ACTION + r.split(ACTION)[1].split(ACTION_END)[-2] + ACTION_END
                    except:
                        print(r)
                        r = r.split(ACTION)[0] + ACTION 
                else:
                    r = r.split(ACTION)[0] + ACTION 
            """
            Note: workflow has to be reattached to the response to make the loss work
            """
            # wf = q.split(WORKFLOW)[-1].split(WORKFLOW_END)[0]#.strip()
            # q = WORKFLOW.join(q.split(WORKFLOW)[:-1]) + WORKFLOW
            # r = wf + WORKFLOW_END + r

            query.append(q)
            response.append(r)
            #############
            if PRINT:
                    print("="*30)
                    print("Query:", q)
                    print("Response:", r)
                    print()

        # encodings_dict = policy.tokenizer(query, return_tensors="pt", padding=True)
        # query_input_ids = encodings_dict['input_ids']
        # query_attention_mask = encodings_dict['attention_mask']   

        # encodings_dict = policy.tokenizer(response, return_tensors="pt", padding=True)
        # response_input_ids = encodings_dict['input_ids']
        # response_attention_mask = encodings_dict['attention_mask']   


        interactive_rollouts = {
            #'query/input_ids': query_input_ids,
            'query/text': query,
            #'query/mask': query_attention_mask,
            #'response/input_ids': response_input_ids,
            'response/text': response,
            #'response/mask': response_attention_mask.to(self.policy.device),
        }

        return interactive_rollouts

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
        if False:
            # abcd set
            tf_paths = [ 
                ("./save/dist_st/230809/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",True, "b2"), \
                ("./save/dist_st/230809/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",True, "b1"), \
                ("./save/dist_st/230809/distilgpt2/kb-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",True, "kb"), \
                ("./save/dist_st/230809/distilgpt2/utt_prediction_future_actions-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",True, "future"), \
                ("./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/", True, "a2"), \
                ("./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/", False, "a2") ,
            ]
            tf_paths = []
            quark_paths = [ 
                #("./outputs/08-09-2023_21:43:57/model/", True, "a2"),  
                #("./outputs/08-09-2023_21:43:57/model/", False, "a2")
                #
                #("./outputs/08-10-2023_21:02:02/model/", True, "a2"),  # longer trained model, longer sample interval
                #("./outputs/08-10-2023_21:02:02/model/", False, "a2")
                # ("./outputs/08-14-2023_19:55:25/model/", True, "a2"),  # temp 1.0 / adaptive kl 320000  
                # ("./outputs/08-14-2023_19:55:25/model/", False, "a2")
                # ("./outputs/08-14-2023_20:00:27/model/", True, "a2"),  # temp 0.5 / adaptive kl 320000  
                # ("./outputs/08-14-2023_20:00:27/model/", False, "a2")
                # ("./outputs/08-16-2023_20:05:28/model/", True, "a2"),  
                # ("./outputs/08-16-2023_20:05:28/model/", False, "a2")
                ("./outputs/09-19-2023_14:26:29/model/", True, "a2"), # current main   # 0.5 ==> this seems to do better in automatic evaluations
                ("./outputs/09-19-2023_14:26:29/model/", False, "a2")
                # ("./outputs/08-17-2023_22:03:40/model/", True, "a2"),  # standard only trained model 0.5
                # ("./outputs/08-17-2023_22:03:40/model/", False, "a2")
                # TODO: last a2 model on standard_only (refigured)
            ] 
            user_path = "./save/dist_st/230809/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/"


            if context_scoring:
                eval_model_path = "./save/context_block_evaluator_scorer/230817/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/"
            else:
                eval_model_path = "./save/block_evaluator_scorer/230809/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/"

            device = torch.device("cuda")
            evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
            eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
            

        else:
            tf_paths = [ 
                ("./save/woz_intent_dist_st/2309/distilgpt2/b2-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/",True, "b2"), \
                ("./save/woz_intent_dist_st/2309/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/",True, "b1"), \
                ("./save/woz_intent_dist_st/2309/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/",True, "a2"), \
                ("./save/woz_intent_dist_st/2309/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/",False, "a2"), \
            ]
            #tf_paths = []
            quark_paths = [ 
                ("./outputs/09-20-2023_19:40:54/model/", True, "a2"), # current main   # 0.5 ==> this seems to do better in automatic evaluations
                ("./outputs/09-20-2023_19:40:54/model/", False, "a2")
            ] 
            #quark_paths = []

            user_path = "./save/woz_intent_dist_st/2309/distilgpt2/b1-distilgpt2-tf-lr5e-4-bs16-epoch1-ws0-gas1-1gpu/" #

            eval_model_path = "./save/woz_block_evaluator_scorer/2309/random/evaluator-roberta-base-tf-lr2e-5-bs32-steps200-ws0-gas1-1gpu/"

            device = torch.device("cuda")
            evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
            eval_tok = AutoTokenizer.from_pretrained(eval_model_path)


        #user_path = "./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/" # just testing
        ref_policy = Policy(model_name=user_path, temperature=1.0, device=device, oracle=False, reward_mode="block") # changing user temperature


        for i, path in enumerate(tf_paths + quark_paths):

            
            oracle = path[1]
            data_type = path[-1]
            path = path[0]
            if path in tf_paths:
                policy = Policy(model_name=path, temperature=temperature, device=device, oracle=oracle, reward_mode="block")
                reward_cond = False
            else:
                policy = Policy(model_name=path, temperature=temperature, device=device,
                            reward_cond=True, tree_tokens=tree_tokens, oracle=oracle, reward_mode="block")
                reward_cond = True


            if "future" in path:
                data_path = "data/wc_seed_future_actions_one_convo.json"
            else:
                #data_path = "data/wc_seed_one_convo.json"
                data_path = "./data/bs_multiwoz_intent.json"
            if block:
                test_dataset = PromptDataset(dataset_type=data_type, path=data_path, split=split, data_len=LEN, \
                oracle=oracle, reward_mode="block", limit_none=False, no_none=False, only_standard_data=only_standard_data)
            else:
                test_dataset = PromptDataset(dataset_type=data_type, path=data_path, split=split, data_len=LEN, \
                oracle=oracle, reward_mode="single", limit_none=False, no_none=False, only_standard_data=only_standard_data)

            prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
            prompt_collator.return_subflow = True

            test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=False, collate_fn=prompt_collator)
        
            best_cat_id = policy.tokenizer.convert_tokens_to_ids(best_cat)
            #continue 

            if not oracle:
                path = path+"_cascade"

            print(f"{path} generating..")
            quark_res = []
            for k, (input_ids, attention_mask, workflow, gt_response, subflow) in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    if block:
                        interactive_rollouts = interactive_sample(policy, ref_policy, input_ids, attention_mask, top_p=1.0, reward_cond=reward_cond, oracle = oracle, data_type=data_type)
                    else:
                        context_len = 512
                        input_ids = input_ids[:, -context_len:] # not ideal
                        attention_mask = attention_mask[:, -context_len:] # not ideal, can lead to bigger loss in context (no left padding here)

                        interactive_rollouts = policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=1.0, max_len = 64)
                    responses = interactive_rollouts['response/text']
                    queries = interactive_rollouts['query/text']

                    if tree_tokens is not None:
                        for tt in tree_tokens:
                            queries = [output.replace(tt,"") for output in queries]

                    if not oracle: # remove false
                        """
                        doing this to input format for reward model
                        """
                        predicted_workflow = [ x.split(WORKFLOW_END)[0].strip() if WORKFLOW_END in x else x for x in responses]
                        responses = [ WORKFLOW_END.join(x.split(WORKFLOW_END)[1:]) if WORKFLOW_END in x else x for x in responses]
                        responses = [ RESPONSE.join(x.split(RESPONSE)[1:]) if RESPONSE in x else x for x in responses ]
                        #queries = [ q + ]
                    else:
                        predicted_workflow = [ "Oracle" for x in responses ]
                    responses = [ x.split(ACTION_END)[0] for x in responses ] #

                    if not block:
                        responses = [ x.split(RESPONSE_END)[0] for x in responses]

                    
                    for q,r, w,g, p, s in zip(queries, responses, workflow, gt_response, predicted_workflow, subflow):
                        if data_type == "future":
                            guideline = retrieve_guideline_text_action(s, w[0])
                        else:
                            guideline = "NA" #retrieve_guideline_text_action(s, w)

                        dic = {"context": q, "response": r, "subflow":s, "true_response":g, "true_wf":w, "predicted_wf":p, "guideline": guideline}
                        #if  not oracle:
                        if False:
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
                dic["guideline"] = line["guideline"]
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



    block = False # for multiwoz test


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

            eval_result[k]["wf_accuracy"] += wf_correct

            tokenized = eval_tok(texts, truncation=True, padding="longest", return_tensors="pt").to(device)

            output = evaluator(**tokenized)

            scores = output.logits.sigmoid().flatten()
            compliance = scores.tolist()
            for i in range(len(compliance)):
                parallel_data[count]["compliance_"+k] = compliance[i]
                #parallel_data[count]["llm_fluency_"+k] = llm_fluency[i]
                count += 1

            eval_result[k]["compliance"] += compliance

            if block:

                #llm_fluency = asyncio.run(batch_evaluate_fluency_coherence(prompts, responses, workflows, subflows))
                # llm_compliance = batch_evaluate_compliance(prompts, responses, workflows, subflows)
            

                # tokenized = eval_tok(texts, truncation=True, padding="longest", return_tensors="pt").to(device)

                # output = evaluator(**tokenized)

                # scores = output.logits.sigmoid().flatten()
                # compliance = scores.tolist()
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


                # for i in range(len(compliance)):
                #     parallel_data[count]["compliance_"+k] = compliance[i]
                #     #parallel_data[count]["llm_fluency_"+k] = llm_fluency[i]
                #     count += 1

                # eval_result[k]["compliance"] += compliance
                eval_result[k]["block_bert"] += block_bert
                
                eval_result[k]["dist_3"] += dist3s
                #eval_result[k]["llm_fluency"] += llm_fluency


            #continue
            if not block:



                # eval_result[k]["compliance"] += compliance

                preds_flat = [ x[0] for x in pred_system]
                gold_flat = [x[0] for x in gt_system]

                # print("="*30)
                # print(k)
                # print(preds_flat)
                # print(gold_flat)


                P, R, F1 = bertscore.score(preds_flat, gold_flat)
                bert_scores = F1.tolist()
                bleurt_scores = bleurt.compute(predictions=preds_flat, references=gold_flat)['scores']
                #perplexity_scores = perplexity.compute(predictions=preds_flat, model_id='gpt2')['perplexities']
                bleu_scores, meteor_scores = [], []
                dist1s, dist2s, dist3s = [], [], []
                for pred, gold in zip(preds_flat, gold_flat):
                    #pred = "\n"
                    results = bleu.compute(predictions=[pred], references=[gold])["bleu"] if pred.strip() != "" else 0.
                    bleu_scores.append(results)
                    results = meteor.compute(predictions=[pred], references=[gold])["meteor"] if pred.strip() != "" else 0.
                    meteor_scores.append(results)
                    #print("pred:", pred)
                    try:
                        dist_1, dist_2, dist_3 = distinctness([pred])
                        #print(pred)
                        #print(dist_3)
                    except:
                        dist_1, dist_2, dist_3 = 0.0, 0.0, 0.0

                    dist1s += [dist_1]
                    dist2s += [dist_2]
                    dist3s += [dist_3]

                eval_result[k]["bertscore"] +=  bert_scores
                eval_result[k]["bleurt"] += bleurt_scores
                #eval_result[k]["perplexity"] += perplexity_scores
                eval_result[k]["bleu"] += bleu_scores
                eval_result[k]["meteor"] += meteor_scores
                eval_result[k]["dist_3"] += dist3s

    total = { k:{kk:np.average(vv) for kk,vv in v.items() } for k,v in eval_result.items() }

    write_data = [ total ] + parallel_data

    with open(f"./test_results/woz_models_{prefix}_{split}_quark_eval_results.json", "w") as fh:
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
