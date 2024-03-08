"""
this script is for running the evaluator model on A1, B2

- Need to get output from A1, B2 evaluation_tf.csv s
"""

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
#from constants import SPECIAL_TOKEN_SET
from model.constants import *

from ast import literal_eval
# need to convert this to list
# block_responses = literal_eval(block_responses)
import random

from sklearn.metrics import cohen_kappa_score
from statsmodels.stats import inter_rater as irr
from scipy import stats
from model.call_openai import batch_evaluate_reward
import asyncio

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

# eval_model_path = "./save/evaluator_ab_rand/230706/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch5-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_no_neg/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_no_random/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"

#eval_model_path = "./save/evaluator/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs64-epoch10-ws0-gas1-1gpu/"

#eval_model_path = "./save/evaluator_scorer/230707/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs64-epoch20-ws0-gas1-1gpu/"

#eval_model_path = "./save/block_evaluator_scorer/230801/random/evaluator-roberta-base-tf-lr2e-5-bs16-epoch1-ws0-gas1-1gpu/"
#eval_model_path = "./save/evaluator_scorer/230731/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs16-epoch10-ws0-gas1-1gpu/"
eval_model_path = "./save/block_evaluator_scorer/230809/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/"
device = torch.device("cuda")
evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
 
# TODO: read the outputs
#output_paths = [ "./test_results/dist_st/b1/epoch10/evaluation_tf.csv", "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv" ]

#output_paths = [ "./test_results/annotator_c.csv", "./test_results/annotator_a.csv" ] #, "./test_results/annotator_b.csv"]
#output_paths = [ "./woz_annotator_c.csv", "./woz_annotator_a.csv" ] #, "./test_results/annotator_b.csv"]
#output_paths = [ "./test_results/annotator_c.csv", "./test_results/annotator_a.csv" ,"./test_results/annotator_b.csv"]
#output_paths =  [ "./test_results/block_annotator_c.csv", "./test_results/block_annotator_b.csv", "./test_results/block_annotator_a.csv"]
# output_paths =  [ "./test_results/new_annotator_c.csv" , "./test_results/new_annotator_b.csv", "./test_results/new_annotator_a.csv"]
output_paths =  [ "./test_results/human_annotator_1_coherence.csv" , "./test_results/human_annotator_3_coherence.csv"] #"./test_results/human_annotator_2.csv", 

#output_paths = ["./test_results/annotator_c.csv"]
KEY_SET = []

MODEL_NUM = 6

eval_data = {}
for datapath in output_paths:
    eval_data[datapath] = []
    #datapath = fprefix + model + "/evaluation_tf.csv"
    with open(datapath, 'r') as data:
        count = 0 
        for line in csv.DictReader(data):
            context = line["context"]
            responses, scores = [], []
            degenerations = []
            for i in range(MODEL_NUM):
                
                response = line[f"{i}_response"].strip()
                try:
                
                    degeneration = 0
                    
                    if line[f"{i}_score"][-1].lower() == "x":
                        score = int(line[f"{i}_score"][:-1].strip())
                        degeneration = 1
                    else:
                        score = int(line[f"{i}_score"].strip())
                    
                    # if degeneration: 
                    #    score = -1
                except:
                #else:
                    score = None
                responses.append(response)
                scores.append(score)
                degenerations.append(degeneration)
            keys = literal_eval(line["keys"])
            KEY_SET.extend(keys)
            assert type(keys) == list, "buig"

            subflow = line["subflow"]
            true_wf = line["true_wf"]
            #dic["true_response"] = true_response
            
            res_dic = {}
            for r,s,k , d in zip(responses, scores,keys, degenerations):
                res_dic[k] = {"response": r, "score":s, "context":context, "wf":true_wf, "subflow":subflow, "degeneration": d}
            dic = { "context": context, "response": response, "subflow":subflow, "result":res_dic, "workflow": true_wf}
            dic["gt_response"] = res_dic["true_response"]["response"]
            dic["true_wf"] = true_wf
            
            #print(true_wf)

            if true_wf == None or true_wf == "None":
                #print(true_wf)
                continue
            eval_data[datapath].append(dic)
            count += 1
            #print(count)
            #print(duc)
            if count >= 100:
                break
        #print(count)
        #dic = { "context": context, "response": response, "true_wf": true_wf }

print(eval_data.keys())

print([len(x) for x in eval_data.values()])    
assert len(set([len(x) for x in eval_data.values()])) == 1, "the models being compared do not have equal-sized generation sets!"

LEN = len(eval_data[list(eval_data.keys())[0]])
#print(eval_data[list(eval_data.keys())[0]][22])
print(LEN)

# random train sample: {'text': "Next Action: None\nAgent: hi how can i help you\nClient: i am thinking about buying some trousers, but i am not sure of the fit. \
#  can i return them if they are not what i need?\nNext Action: membership\nAgent: sure let me check\nAction: membership bronze\nAction: search-faq \nAction: search-policy \
#  \nNext Action: None\nAgent: depends on your membership level, what is your membership?\nClient: bronze\nNext Action: None\nAgent: bronze members are allowed returns for purchases made\
#   in the last 90 days.\nClient: is this true even if i buy them on sale?\nNext Action: None\nAgent: yes, that;'s true!\nClient: okay great. \
#  do you include a shipping label?\nAgent: you can come back here and i can send you one if you need it\nWorkflow Action: None", 'label': 1, 'type': 'positive', 'action': 'None'}


# './test_results/dist_st/b2/epoch10/evaluation_tf.csv': 'Next Action: None Agent: hi, how may i help you this morning? Client: yea, i had a quick question \
# Client: i was checking my email and it says my subscription was removed Client: is that true?  i still want it there Next Action: pull-up-account \
# Agent: sure, i can check that for you Next Action: pull-up-account Agent: what is your account id? Client: umm, not sure Next Action: pull-up-account\
#  Agent: can i have your full name or account id? Workflow Action: pull-up-account', 

original = eval_data # naming failure

KEYS = set(KEY_SET) #keys
print(KEYS)
# TODO: eval loop
eval_data = {}
for i in KEYS:
    eval_data[i] = []

for d in original[list(original.keys())[0]]:
    for i in KEYS:
        #print(d["result"].keys())
        context = d["context"]
        response = d["result"][i]["response"]
        true_wf = d["true_wf"]
        model_input = context + "\nAgent: " + response + "\nWorkflow Action: " + true_wf
        
        context = response.replace(USER, "\nClient: ")
        context = context.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")
        
        context = context.strip()
        # 
        #context = "\nNext Action: ".join(context.split("\nNext Action: ")[:-1]).strip()

        response = "Agent: "+context
        new = []
        rsplit = response.split("\n")
        temp = []
        for k,rs in enumerate(rsplit):
            #if k == len(rsplit) -1:
            if rs.startswith("Action:"):
                pass
            elif rs.startswith("Next Action:"):
                pass
            else:
                temp.append(rs)
        r = "\n".join(temp)
        #pos_responses.append(r)
        #print(r)
        response = r

        model_input = f"{response}\nWorkflow Action: {true_wf}"

        eval_data[i].append(model_input)

#print(eval_data.keys())



eval_result = {}
for k, v in eval_data.items():
    print("Evaluating ", k)
    eval_result[k] = [  ]
    for batch in chunk(v, size=16):

        # prompts = [ x["context"] for x in v]
        # new_prompts = []
        # for p in prompts:
        #     temp = []
        #     split = p.split("\n")
        #     for s in split:
        #         if s.startswith("Next Action:"):
        #             pass
        #         else:
        #             temp += [s]
        #     new_prompts.append("\n".join(temp))
        # prompts = new_prompts
        # workflows = [ x["workflow"] for x in v]
        # responses = [ x["response"] for x in v]
        # subflows = [ x["subflow"] for x in v]     
        if False:
            scores = [0.0 for x in batch]
            eval_result[k] += scores#.tolist()
            continue

        tokenized = eval_tok(batch, truncation=True, padding="longest", return_tensors="pt").to(device)
        output = evaluator(**tokenized)
        scores = output.logits.sigmoid().flatten()
        eval_result[k] += scores.tolist() #[ x[1] for x in scores.tolist() ] 

import numpy as np
for k, v in eval_result.items(): # ignore this...
    print(k)
    print(np.average(v))
    print()

#exit()

"""
TODO: Gather all the results
"""
all_together = { k:[] for k in eval_result.keys()}
# must be lsit of dicts { model_score: int, annotator_score: list}
# for annotator, v in original.items():
#     print(annotator)
#     print(len(v))
#     for dic in v:
#         res_dic = dic["result"]
        # scores, responses = [], []
        # for k,vv in res_dic.items():
        #     response = vv["response"]
        #     score = vv["score"]
        #     responses.append(response)
        #     scores.append(score)


#

unanimous =  { model:[] for model in eval_result.keys()}
thresholds =  { model:[] for model in eval_result.keys()}
for model, scores in eval_result.items():
    try:
        model_name = "_".join(model.split("/")[-4:2])
    except:
        model_name = model
    with open(f"./test_results/block_{model_name}.csv", "w") as fh:
        header1 = ["context", "subflow","workflow", "response","model_score", "annotator_scores", "gt_response", "average", "degeneration" ]
        writer = csv.DictWriter(fh,  header1 )
        writer.writeheader()
        for i, score in enumerate(scores):
            row = { "model_score":score,  "annotator_scores": [], "degeneration": [] }
            for annotator, v in original.items():
                item = v[i]
                res_dic = item["result"][model]
                annotator_score = res_dic["score"]
                if annotator_score == None:
                    annotator_score = -0
                else:
                    annotator_score = annotator_score / 3.0
                # elif annotator_score == 0:
                #     #annotator_score = -1
                #     annotator_score = 0
                # elif annotator_score == -1:
                #     annotator_score = 0
                # elif annotator_score == 1:
                #     annotator_score = 1
                degeneration = res_dic["degeneration"]
                row["annotator_scores"].append(annotator_score)
                row["degeneration"].append(degeneration)
            row["gt_response"] = v[i]["gt_response"]
            if False and "model" in model:
                anno = row["annotator_scores"]
                if (np.average(anno) > 0.5 and score <= 0.5) or (np.average(anno) <= 0.5 and score > 0.5) :
                    print("="*30)
                    print(model)
                    print("context:", res_dic["context"])
                    print("workflow:", res_dic["wf"])
                    print("response:", res_dic["response"])
                    print("gt_response:", row["gt_response"])
                    print(score)
                    print(row["annotator_scores"])
                    input()
            if False:            
                if len(set(row["annotator_scores"])) == 1:
                    dic = {"human_score": list(set(row["annotator_scores"]))[0], \
                    "model_score":score, "response":res_dic["response"], "context": res_dic["context"]}
                    unanimous[model].append(dic)
                    print("="*30)
                    print("context:", res_dic["context"])
                    print("workflow:", res_dic["wf"])
                    print("response:", res_dic["response"])
                    print(row["annotator_scores"])
                    print(score)
                    print()
                    input()

                if score >= 0.75 or score <= 0.25:
                    dic = {"human_scores": row["annotator_scores"], \
                    "model_score":score, "response":res_dic["response"], "context": res_dic["context"]}
                    thresholds[model].append(dic)
                
            row["average"] =  np.average([x for x in row["annotator_scores"] if x!=-2])
            #row["average"] = np.average([x for x in row["annotator_scores"] if x!=-2])
            all_together[model].append(row)
            row["response"] = res_dic["response"]
            row["context"] = res_dic["context"]
            row["workflow"] = res_dic["wf"]
            row["subflow"] = res_dic["subflow"]
            #print(row)
            #input()
            writer.writerow(row)

if False:
    for model, v in thresholds.items():
        top = [ x for x in v if x["model_score"] >= 0.75]
        bottom = [ x for x in v if x["model_score"] <= 0.75]
        for t in top+bottom:
            print("="*30)
            print(t["model_score"])
            print(t["human_scores"])
            print(t["context"])
            print(t["response"])
            print()
            input()
#exit()
if False:
    for model, v in unanimous.items():
        zero = [ x for x in v if x["human_score"] == 0]
        non = [ x for x in v if x["human_score"] == -1]
        com = [ x for x in v if x["human_score"] == 1]
        print("="*30)
        print(model)
        print("zero")
        print(len(zero))
        print(np.average([x["model_score"] for x in zero]))
        print("non")
        print(len(non))
        print(np.average([x["model_score"] for x in non]))
        print("com")
        print(len(com))
        print(np.average([x["model_score"] for x in com]))
        print("pearson:", stats.pearsonr([x["human_score"] for x in v], [x["model_score"] for x in v]))
        print("spearman:", stats.spearmanr([x["human_score"] for x in v], [x["model_score"] for x in v]))
        print()

    

#exit()


for k,v in all_together.items():
    print("="*30)
    print(k)
    #print(v)
    avg_annotator = [vv["average"] for vv in v]
    model = [ vv["model_score"] for vv in v]
    #print("avg annotator:", avg_annotator)
    print("avg judgement:", np.average([vv["average"] for vv in v if not np.isnan([vv["average"]])]))
    print("avg model:", np.average(model))
    print("degeneration:", np.average([ np.average(vv["degeneration"]) for vv in v ]))
    arr = [ vv["annotator_scores"] for vv in v]
    
    print("Percent model matching judgement")
    # print(np.average([ int(score>=0.5) == j for j, score in zip(avg_annotator, model)  ]))
    #print(np.average([ int(score>=0.5) == max(j) for j, score in zip(arr, model)  ]))
    matched = []
    human_strict = []
    for j, score in zip(arr, model):
        #print(arr)
        if 0 in j or -1 in j:
            gt = 0
        else:
            gt = 1
        model_score = int(score>=0.5)
        matched.append(model_score == gt)
        human_strict.append(gt)
    print("matched:", np.average(matched))
    print("annotator score:", arr)
    print("degeneration score:", [ vv["degeneration"] for vv in v] )
    agg = irr.aggregate_raters(arr)
    # print(arr)
    # print(agg)
    
    fleish = irr.fleiss_kappa(agg[0],method='fleiss')
    print("fleiss kappa:", fleish)
    print("pearson:", stats.pearsonr(model, avg_annotator))
    print("spearman:", stats.spearmanr(model, avg_annotator))
    print(human_strict)
    print("pearson:", stats.pearsonr(model, human_strict))
    print("spearman:", stats.spearmanr(model, human_strict))

    assert len(v) == len(eval_result[k]), f"{len(v)} != {len(eval_result[k])}"

    for i in range(1,4):
        # TODO:
        compliant_i = [ sum([ y >=1 for y in  x]) >= i for x in arr]
        noncompliant_i = [ sum([ y <=-1 for y in  x]) >= i for x in arr]
        print(f"Avg fraction of generations where at least {i} annotators marked as compliant = 1")
        print(np.average(compliant_i))
        print(f"Avg fraction of generations where at least {i} annotators marked as noncompliant = -1")
        print(np.average(noncompliant_i))
    print()
    #exit()

exit()






#only_include = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "true_response" ]
only_include = eval_data.keys()

with open("eval.csv", "w") as fh:
    header1 = [  [x+ "_response", x+"_score"] for x in list(eval_data.keys())]
    header1 = [ x for sublist in header1 for x in sublist]
    header2 = [] #[  x+ "_pred" for x in list(eval_data.keys())]
    header3 = [  ] # x+ "_score" for x in list(eval_data.keys())]
    header4 = [ ] # "subflow"]
    header5 = [] #[  str(i)+ "_score" for i,x in enumerate(only_include)]
    writer = csv.DictWriter(fh,  ["context", "subflow", "true_wf"] + header1 + header2 + header3 + header4)
    writer.writeheader()
    
    row = {}
    for i in range(LEN):
        # if i not in chosen:
        #     continue
        # print(i, len(parallel_data))
        # print(continue)
        row["context"] = parallel_data[i]["context"]
        original_data = parallel_data[i]
        #keys = list(only_include) #list(original_data.keys())#list(range(len(original_data)))
        #print("keys", keys)
        #random.shuffle(keys)
        #shuffled_data =  { key:original_data[key] for key in keys}
        #for z, k in enumerate(keys):
        for k,v in eval_result.items():
            row[k+"_response"] = parallel_data[i][k]
            #row[k+"_pred"] = eval_result[k][0][i]
            row[k+"_score"] = eval_result[k][1][i]
        row["true_wf"] = parallel_data[i]["true_wf"]
        #row["keys"] = [ x.split("/")[3] if x!="true_response" else x for x in keys ]
        row["subflow"] = parallel_data[i]["subflow"]
        #print()
        writer.writerow(row)
        # print(parallel_data[i]["true_wf"])
        # print(parallel_data[i]["subflow"])