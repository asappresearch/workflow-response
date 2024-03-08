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

from collections import Counter, OrderedDict

set_seed(42)

def filter_examples(data, min_length=200, none_frac = 0.0125/2.0):
    print("Original workflows:" ,Counter([x["true_wf"] for x in data]))
    print("Original unique workflows:" ,len(set([x["true_wf"] for x in data])))
    print("Original subflows:" ,Counter([x["subflow"] for x in data]))
    print("Original unique subflows:" ,len(set([x["subflow"] for x in data])))
    print(len(data))
    wf = [ v["true_wf"] for v in data]
    sf = [ v["subflow"] for v in data]
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

    sf2idx = OrderedDict(sorted(sf2idx.items(), key=lambda i: -len(i[1])))
    wf2idx = OrderedDict(sorted(wf2idx.items(), key=lambda i: len(i[1])))

    # print(sf2idx)
    # print(wf2idx)
    # exit()
    chosen = []
    while len(chosen) < min_length - int(none_frac * min_length) :
        for k,v in wf2idx.items():
            if k == "end-dialog":
                continue
            pool = list(set(v) - set(chosen))
            if len(pool) == 0:
                continue
            else:
                chosen.append(random.choice(pool))
            if len(chosen) >= min_length - int(none_frac* min_length):
                break

    chosen += random.sample(wf2idx["end-dialog"], int(none_frac * min_length))
    
    data = [ x for i,x in enumerate(data) if i in chosen]

    print("Included workflows:" ,Counter([x["true_wf"] for x in data]))
    print("Included unique workflows:" ,len(set([x["true_wf"] for x in data])))
    print("Included subflows:" ,Counter([x["subflow"] for x in data]))
    print("Included unique subflows:" ,len(set([x["subflow"] for x in data])))
    print(len(data))

    return data

def main(
    #LEN = 20000,
    block = True,
    use_scorer = False,
    temperature = 1.0,
    prefix = "block",
    context_scoring = False,
    only_standard_data = True,
    result_path = None,
    split="test"
    ):
    #num_len = LEN
    

    if result_path == None:
        #result_path = f"./test_results/models_{prefix}_{split}_quark_eval_results.json"
        result_path = "./test_results/models_test_results.json"
    with open(result_path, "r") as fh:
        data = json.load(fh)

    stat = data[0]
    print(stat)
    parallel_data = data[1:]
    parallel_data = filter_examples(parallel_data[1:])
    
    
    LEN = len(parallel_data)
    #only_include = [ "./test_results/dist_st/b2/epoch10/evaluation_tf.csv", "./test_results/dist_st/utt_prediction_oracle_wf/epoch10/evaluation_tf.csv", "true_response" ]
    only_include = stat.keys() #eval_result.keys() #parallel_data.keys() #eval_data.keys()  
    only_include = [ x for x in only_include if "kb" not in x and "future" not in x and "b1" not in x]
    #random.shuffle(parallel_data)
    #parallel_data = parallel_data[:100]
    #LEN = 100

    with open("./test_results/filtered_test_results.json", "w") as fh:
        # this tat is not rpresentative of parallel data (that is)
        json.dump([stat] + parallel_data, fh, indent =4)

    def cleanup(string, add_agent=True):
        context = string.replace(RESPONSE, "\nAgent: ")
        context = context.replace(WORKFLOW, "\nNext Action: ")
        context = context.replace(ACTION, "\nAction: ")
        context = context.replace(USER, "\nClient: ")

        for stoken in SPECIAL_TOKEN_SET:
            context = context.replace(stoken, "")

        if add_agent:
            r = "Agent: "+context
        else:
            r = context
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

    with open(f"./test_results/{prefix}_quark_eval.csv", "w") as fh:
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

            row["context"] = cleanup(parallel_data[i]["context"], add_agent=False)
            original_data = parallel_data[i]
            keys = list(only_include) #list(original_data.keys())#list(range(len(original_data)))
            #print("keys", keys)
            random.shuffle(keys)
            shuffled_data =  { key:original_data["output_"+ key] for key in keys}
            for z, k in enumerate(keys):

                row[str(z)+"_response"] = cleanup(shuffled_data[k]) #parallel_data[i][k]
                row[str(z)+"_model_score"] = parallel_data[i]["compliance_"+str(k)]
                row[str(z)+"_score_context"] = parallel_data[i]["context_"+str(k)]

            row["true_wf"] = parallel_data[i]["true_wf"]
            #row["keys"] = [ x.split("/")[3] if x!="true_response" and x!="quark" else x for x in keys ]
            row["keys"] = [  x for x in keys ]
            row["subflow"] = parallel_data[i]["subflow"]
            row["guideline"] = parallel_data[i]["guideline"]
            
            writer.writerow(row)


if __name__ == "__main__":
    fire.Fire(main)
