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
from tqdm import tqdm

from transformers import set_seed

set_seed(42)

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]


#eval_model_path = "./save/evaluator_scorer/230801/neg/evaluator-roberta-base-tf-lr2e-5-bs16-epoch1-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_scorer/230801/random/evaluator-roberta-base-tf-lr2e-5-bs16-epoch1-ws0-gas1-1gpu/"
eval_model_path = "./save/block_evaluator_scorer/230801/random/evaluator-roberta-base-tf-lr2e-5-bs16-epoch1-ws0-gas1-1gpu/"

device = torch.device("cuda")
evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
# evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path, torch_dtype=torch.float16).to(device)
eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
 
# TODO: read the outputs
#output_paths = [ "./test_results/dist_st/b1/epoch10/evaluation_tf.csv", "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv" ]

#output_paths = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv" ]

output_paths = [ "./test_results/block/230802/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/block/230802/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv"]


"""
Running once and getting the stats first
"""
if True:
    datapath = output_paths[1] #"./test_results/woz_da_dist_st/230726/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv"
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

if False:
    for k,v in eval_data.items():
        eval_data[k] = [ x for i,x in enumerate(v) if i in chosen]
# TODO: format the outputs in evaluator-format
#print(eval_data)
LEN = len(eval_data[list(eval_data.keys())[0]])
#print(eval_data[list(eval_data.keys())[0]][22])
print(LEN)


# import quark model and generate
# quark_paths = [  "./outputs/08-03-2023_19:55:07/model/"] 
quark_paths = [ "./outputs/08-08-2023_17:10:43/model/", "./outputs/08-08-2023_18:31:19/model/"  ]# "./outputs/08-04-2023_21:39:50/model/"] 
#                       1.0                                 #   0.5
#quark_paths = [ ]# "./outputs/08-04-2023_21:39:50/model/"] 

#quark_path = "./outputs/top_oracle/model" #"./outputs/07-28-2023_17:28:41/model/"
from Quark.main import *

for quark_path in quark_paths:
    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(5)] + \
                    [' _TREE_TOKEN_ZERO_COMMENTS']
    policy = Policy(model_name=quark_path, temperature=0.5, device=device,
                    reward_cond=True, tree_tokens=tree_tokens, oracle=True, reward_mode = "block")
    test_dataset = PromptDataset(dataset_type="a2", path="data/wc_seed_one_convo.json", split="dev", data_len=2000, oracle=True, reward_mode ="block")

    policy.tokenizer.padding_side = "left"
    policy.tokenizer.truncation_side = 'left'
    # Define PAD Token = EOS Token = 50256
    policy.tokenizer.pad_token = policy.tokenizer.eos_token
    policy.model.config.pad_token_id = policy.model.config.eos_token_id
        
    oracle_path = output_paths[1]
    cascade_path = output_paths[1]
    p, w, g = [] , [], []
    for o,c in zip(eval_data[oracle_path], eval_data[cascade_path]):
        if True: #"oracle" in quark_path:
            context = o["context"]
        else:
            context = c["context"]
        # print(context)
        # input()
        response = o["true_response"]
        subflow = o["subflow"]
        true_wf = c["true_wf"]
        p.append(context)
        w.append(true_wf)
        g.append(response)
    test_dataset.prompts = p
    test_dataset.gt_responses = g
    test_dataset.workflows = w

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=prompt_collator)

    best_cat = tree_tokens[0]
    best_cat_id = policy.tokenizer.convert_tokens_to_ids(best_cat)

    def add_control_code(input_ids, attention_mask):
        input_ids = torch.cat([input_ids.new([best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        return input_ids, attention_mask

    print(f"{quark_path} generating..")
    quark_res = []
    for i, (input_ids, attention_mask, workflow, gt_response) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            input_ids, attention_mask = add_control_code(input_ids, attention_mask)
            # rollouts = policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=1.0, max_len = 64)
            # forward_inputs = {'query_input_ids': rollouts['query/input_ids'][:, 1:],
            #                     'query_mask': rollouts['query/mask'][:, 1:],
            #                     'response_input_ids': rollouts['response/input_ids'],
            #                     'response_mask': rollouts['response/mask']}
            
            # queries = rollouts["query/text"]
            # responses = rollouts["response/text"]

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

            responses = [ x.split(ACTION_END)[0] for x in responses ] #

            for q,r, w,g in zip(queries, responses, workflow, gt_response):
                dic = {"context": q, "response": r, "subflow":None, "true_response":g, "true_wf":w}
                #print(dic)
                quark_res.append(dic)

    eval_data[quark_path] = []
    for line in quark_res:
        context = line["context"]#+line["response_1"].strip()+"\nsystem: "
        response = line["response"].strip()
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
        eval_data[quark_path].append(dic)

#print(len(eval_data["quark"]))
        #dic = { "context": context, "response": response, "true_wf": true_wf }
# import multiwoz files
#output_paths = [ "./test_results/woz_da_dist_st/230726/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/woz_da_dist_st/230726/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "./test_results/woz_da_dist_st/230726/dist_st/utt_prediction_cascade/epoch10/evaluation_tf.csv" ]




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
        #dic[k] = v[i]["response"]
        response = v[i]["response"]

        context = response.replace(USER, "\nClient: ")
        context = context.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")
        response = context
        dic[k] = "Agent: " +response

    
        #dic["true_response"] = 
        gt = v[i]["true_response"]
        context = gt.replace(USER, "\nClient: ")
        context = context.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")
        gt = context
        dic["true_response"] =  "Agent: " +gt

    
        context = v[i]["context"]
        # the cascade model has the correct true_wf info, not the oracle
        if "true_wf" in v[i]: # and "oracle" in k:
            gt_wf = v[i]["true_wf"]    
        #if k != "quark":
        if k not in quark_paths:
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
    #context = "\nNext Action: ".join(context.split("\nNext Action: ")[:-1]).strip()

    dic["context"] = context

    #for k in list(eval_data.keys()) + ["true_response"]:
    #    dic["context_"+k] = context + "\nAgent: " + dic[k] + "\nWorkflow Action: " + dic["true_wf"]
    #dic["true_response"] = context + " " + dic["true_response"] + " Workflow Action: " + dic["true_wf"]
    parallel_data.append(dic)


#print(parallel_data[10])
# for i in parallel_data:
#     print(i)
#     input()

# TODO: eval loop
eval_data = {}
for k in output_paths + quark_paths + [ "true_response"]:
    eval_data[k] = []
    for dat in parallel_data:
        #context = dat["context"]
        response = dat[k] #d["result"][i]["response"]
        true_wf = dat["true_wf"]

        # context = response.replace(USER, "\nClient: ")
        # context = context.replace(RESPONSE, "\nAgent: ")
        # context = context.replace(WORKFLOW, "\nNext Action: ")
        # context = context.replace(ACTION, "\nAction: ")

        # for stoken in SPECIAL_TOKEN_SET:
        #     context = context.replace(stoken, "")
        
        # context = context.strip()
        
        #context = "\nNext Action: ".join(context.split("\nNext Action: ")[:-1]).strip()

        #response = "Agent: "+response
        new = []
        rsplit = response.split("\n")
        temp = []
        for i,rs in enumerate(rsplit):
            #if i == len(rsplit) -1:
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
        #print(model_input)
        eval_data[k].append(model_input)

#print(eval_data.keys())


#print(eval_tok.model_max_length)
eval_result = {}
for k, v in eval_data.items():
    print("Evaluating ", k)
    eval_result[k] = [ [], [] ]
    for batch in chunk(v, size=8):
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


#only_include = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "true_response" ]
only_include = [ x for x in eval_data.keys() ] #if "cascade" not in x ]

random.shuffle(parallel_data)
parallel_data = parallel_data#[:100]
#LEN = 100
with open("block_qqeval.csv", "w") as fh:
    header1 = [  str(x)+ "_response" for i,x in enumerate(only_include)]
    header2 = [] #[  x+ "_pred" for x in list(eval_data.keys())]
    header3 = [] #[  x+ "_score" for x in list(eval_data.keys())]
    header4 = [ "keys", "subflow"]
    header5 = [  str(x)+ "_score" for i,x in enumerate(only_include)]
    writer = csv.DictWriter(fh,  ["context", "true_wf"] + header1 + header2 + header3 + header4 + header5)
    writer.writeheader()
    
    row = {}
    for i in range(LEN):
        # if i not in chosen:
        #     continue
        
        row["context"] = parallel_data[i]["context"]
        original_data = parallel_data[i]
        keys = list(only_include) #list(original_data.keys())#list(range(len(original_data)))
        
        #random.shuffle(keys)
        shuffled_data =  { key:original_data[key] for key in keys}
        for z, k in enumerate(keys):
        #for k,v in eval_result.items():
            #response = shuffled_data[k]
            temp = []
            for r in response.split("\n"):
                if r.startswith("Action:"):
                    pass
                elif r.startswith("Next Action:"):
                    pass
                else:
                    temp += [r]
            response = "\n".join(temp)
            row[str(k)+"_response"] = response #shuffled_data[k] #parallel_data[i][k]
            #row[k+"_pred"] = eval_result[k][0][i]
            row[str(k)+"_score"] = eval_result[k][1][i]
        row["true_wf"] = parallel_data[i]["true_wf"]
        row["keys"] = [ x.split("/")[-3] if x!="true_response" and x not in quark_paths else x for x in keys ]
        row["subflow"] = parallel_data[i]["subflow"]
        
        writer.writerow(row)