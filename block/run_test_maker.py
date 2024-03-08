"""
this script is for running the evaluator model on A1, B2

- Need to get output from A1, B2 evaluation_tf.csv s
"""

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv
#from constants import SPECIAL_TOKEN_SET
from model.constants import *

import random

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

# eval_model_path = "./save/evaluator_ab_rand/230706/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch5-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_no_neg/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_no_random/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"

#eval_model_path = "./save/evaluator/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs64-epoch10-ws0-gas1-1gpu/"

eval_model_path = "./save/evaluator_scorer/230707/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs64-epoch20-ws0-gas1-1gpu/"

device = torch.device("cuda")
evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
# evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path, torch_dtype=torch.float16).to(device)
eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
 
# TODO: read the outputs
#output_paths = [ "./test_results/dist_st/b1/epoch10/evaluation_tf.csv", "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv" ]

output_paths = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv" ]



"""
Running once and getting the stats first
"""
datapath = "./test_results/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv"
stat_data = []
with open(datapath, 'r') as data:
    for line in csv.DictReader(data):
        context = line["context"]#+line["response_1"].strip()+"\nsystem: "
        response = line["response_1"].strip()
        subflow = line["subflow"]
        dic = { "context": context, "response": response, "subflow":subflow}
        if "true_response" in line:
            true_response = line["true_response"]
        else:
            true_response = None
        if "true_wf" in line:
            true_wf = line["true_wf"]
        else:
            true_wf = "Oracle"
        for stoken in SPECIAL_TOKEN_SET:
            true_wf = true_wf.replace(stoken, "")
        dic["true_response"] = true_response
        dic["true_wf"] = true_wf
        stat_data.append(dic)
    #dic = { "context": context, "response": response, "true_wf": true_wf }
wf = [ v["true_wf"] for v in stat_data]
sf = [ v["subflow"] for v in stat_data]
sf2idx = {}
wf2idx = {}

for i, ssf in enumerate(sf):
    if ssf not in sf2idx:
        sf2idx[ssf] = [i]
    else:
        sf2idx[ssf] += [i]

for i, ssf in enumerate(wf):
    if ssf not in wf2idx:
        wf2idx[ssf] = [i]
    else:
        wf2idx[ssf] += [i]

from collections import OrderedDict
sf2idx = OrderedDict(sorted(sf2idx.items(), key=lambda i: -len(i[1])))
wf2idx = OrderedDict(sorted(wf2idx.items(), key=lambda i: len(i[1])))

# print(sf2idx)
# print(wf2idx)
# exit()
chosen = []
while len(chosen) < 95:
    for k,v in wf2idx.items():
        if k == "None":
            continue
        pool = list(set(v) - set(chosen))
        if len(pool) == 0:
            continue
        else:
            chosen.append(random.choice(pool))
        if len(chosen) >= 95:
            break

chosen += random.sample(wf2idx["None"],5)

# for k,v in sf2idx.items():
#     chosen.append(random.choice(v))
# for k,v in sf2idx.items():
#     if len(chosen) >= 100:
#         break
#     chosen.append(random.choice(list(set(v) - set(chosen))))
#print(chosen)
print(wf2idx["None"])
print(len(chosen))
from collections import Counter
print(Counter(wf))
print(Counter(sf))
print(len(Counter(wf)))
print(len(Counter(sf)))
# exit()

# Counter({'None': 451, 'pull-up-account': 150, 'enter-details': 48, 'verify-identity': 47, \
# 'search-faq': 45, 'validate-purchase': 41, 'membership': 30, 'select-faq': 25, 'promo-code': 17, \
# 'log-out-in': 16, 'make-purchase': 16, 'update-order': 15, 'record-reason': 15, 'instructions': 11, \
# 'offer-refund': 11, 'subscription-status': 9, 'shipping-status': 9, 'try-again': 9, 'notify-team': 9, \
# 'ask-the-oracle': 9, 'update-account': 7, 'send-link': 7, 'make-password': 3})


eval_data = {}
for datapath in output_paths:
    eval_data[datapath] = []
    #datapath = fprefix + model + "/evaluation_tf.csv"
    with open(datapath, 'r') as data:
        for line in csv.DictReader(data):
            context = line["context"]#+line["response_1"].strip()+"\nsystem: "
            response = line["response_1"].strip()
            subflow = line["subflow"]
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
            eval_data[datapath].append(dic)
        #dic = { "context": context, "response": response, "true_wf": true_wf }

print(eval_data.keys())

print([len(x) for x in eval_data.values()])    
assert len(set([len(x) for x in eval_data.values()])) == 1, "the models being compared do not have equal-sized generation sets!"

for k,v in eval_data.items():
    eval_data[k] = [ x for i,x in enumerate(v) if i in chosen]
# TODO: format the outputs in evaluator-format
#print(eval_data)
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

parallel_data = []
for i in range(LEN):
    # if i not in chosen:
    #     continue
    dic = {}
    for k, v in eval_data.items():
        dic[k] = v[i]["response"]
        dic["true_response"] = v[i]["true_response"]
    
        context = v[i]["context"]
        # the cascade model has the correct true_wf info, not the oracle
        if "true_wf" in v[i]: # and "oracle" in k:
            gt_wf = v[i]["true_wf"]    
        subflow = v[i]["subflow"]
    for stoken in SPECIAL_TOKEN_SET:
        gt_wf = gt_wf.replace(stoken, "")
    dic["true_wf"] = gt_wf
    dic["subflow"] = subflow
    #print(gt_wf, subflow)
    context = context.replace(USER, "\nClient: ")
    context = context.replace(RESPONSE, "\nAgent: ")
    context = context.replace(WORKFLOW, "\nNext Action: ")
    context = context.replace(ACTION, "\nAction: ")

    for stoken in SPECIAL_TOKEN_SET:
        context = context.replace(stoken, "")
    
    context = context.strip()
    # 
    context = "\nNext Action: ".join(context.split("\nNext Action: ")[:-1]).strip()

    dic["context"] = context

    for k in list(eval_data.keys()) + ["true_response"]:
        dic["context_"+k] = context + "\nAgent: " + dic[k] + "\nWorkflow Action: " + dic["true_wf"]
    #dic["true_response"] = context + " " + dic["true_response"] + " Workflow Action: " + dic["true_wf"]
    parallel_data.append(dic)


#print(parallel_data[10])
# for i in parallel_data:
#     print(i)
#     input()

# TODO: eval loop
eval_data = {}
for k in output_paths + ["true_response"]:
    eval_data[k] = []
    for dat in parallel_data:
        eval_data[k].append(dat["context_"+k])

#print(eval_data.keys())


#print(eval_tok.model_max_length)
eval_result = {}
for k, v in eval_data.items():
    print("Evaluating ", k)
    eval_result[k] = [ [], [] ]
    for batch in chunk(v, size=16):
        tokenized = eval_tok(batch, truncation=True, padding="longest", return_tensors="pt").to(device)

        output = evaluator(**tokenized)

        preds = output.logits.sigmoid().flatten()
        scores = output.logits.sigmoid().flatten()

        #preds = output.logits.argmax(-1)
        #scores = output.logits.softmax(-1)

        eval_result[k][0] += preds.tolist()
        eval_result[k][1] += scores.tolist() #[ x[1] for x in scores.tolist() ] 
# for i in range(LEN):
#     print("="*30)
#     print("Context: ", parallel_data[i]["context"])
#     for k,v in eval_result.items():
#         print("Model: ", k)
#         print("Response: ", parallel_data[i]["context_"+k])
#         print("Prediction: ", eval_result[k][i])

#     print()
import numpy as np
for k, v in eval_result.items():
    print(k)
    print(np.average(v[0]))
    print(np.average(v[1]))
    print()


only_include = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "true_response" ]
#only_include = eval_data.keys()

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
        row["keys"] = [ x.split("/")[3] if x!="true_response" else x for x in keys ]
        row["subflow"] = parallel_data[i]["subflow"]
        #print()
        writer.writerow(row)
        # print(parallel_data[i]["true_wf"])
        # print(parallel_data[i]["subflow"])