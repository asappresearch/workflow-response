import json
from model.call_openai import *
import fire

import numpy as np
from collections import Counter

import asyncio

from scipy import stats
from dataproc.make_human_eval import filter_examples

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

def evaluate_compliance(result_file = "./test_results/filtered_test_results.json", LEN=200, context=False, filter_samples=False):

    with open(result_file, "r") as fh:
        data =json.load(fh)

    print("Read:", result_file)
    
    stat = data[0]

    model_names = [x for x in  stat.keys() ]
    
    data = data[1:]
    print("Original Row size:", len(data))
    if filter_samples:
        data =   filter_examples(data)
        LEN = len(data)
    else:
        data = data[:LEN] # check
    print("Used Row size:", len(data))
    
    model_dic = {}
    for dat in data:
        wf = dat["true_wf"]
        sf = dat["subflow"]
        guideline = dat["guideline"]

        #wfs += [wf]
        #   sf_wfs += [sf+"_"+wf]

        result = {}

        fixed_model_name = model_names[0] #list(dat.keys())[-1] # so that it's fixed and not true_response
        for k,v in dat.items():
            if "context_" in k:
                model_name = k[len("context_"):] #].strip("context")
                model_input = cleanup(dat["input_"+fixed_model_name])

                compliance_score = dat["compliance_"+model_name]
                if model_name not in model_dic:
                    model_dic[model_name] =[ [v, guideline, wf, sf, model_input, compliance_score ] ]
                else:
                    model_dic[model_name] += [ [v, guideline, wf, sf, model_input, compliance_score ] ]

    dics = []
    for k,v in model_dic.items():
        #only_include = [ x for x in only_include if "kb" not in x and "future" not in x and "b1" not in x]
        if "kb" in k or "future" in k or "b1" in k:
            continue
        print(k)
        gens = [ vv[0] for vv in v]
        guidelines = [ vv[1] for vv in v]
        wfs = [ vv[2] for vv in v]
        sfs = [ vv[3] for vv in v]
        inputs = [ vv[4] for vv in v]
        compliance_scores = [vv[5] for vv in v]

        if not context:
            #inputs = None
            results = asyncio.run(batch_evaluate_compliance(guidelines, gens, wfs, sfs, None))
        else:
            results = asyncio.run(batch_evaluate_compliance(guidelines, gens, wfs, sfs, inputs))
        print("="*30)
        print("Model:", k)
        print("Result:", np.average(results))
        pearson = stats.pearsonr(compliance_scores, results)
        spearman = stats.spearmanr(compliance_scores, results)
        print("pearson:", pearson)
        print("spearman:", spearman)
        print()

        dic = { "model_name":k, "input": inputs, "target": gens, "guidelines": guidelines, \
        "workflow":wfs, "subflows":sfs, "llm_scores":results, "model_scores":compliance_scores,\
        "pearson": pearson, "spearman":spearman, "llm_average":np.average(results)}
        dics.append(dic)

    with open(f"./test_results/compliance_new_prompt_context-{context}_{LEN}.json", "w") as fh:
        json.dump(dics, fh, indent=4)

    
    return 


def evaluate_fluency(result_file = "./test_results/mother_eval_results.json", LEN=20, context=False):

    with open(result_file, "r") as fh:
        data =json.load(fh)

    print("Read:", result_file)
    
    stat = data[0]

    model_names = [x for x in  stat.keys() ]
    
    data = data[1:]
    print("Original Row size:", len(data))
    data = data[:LEN] # check
    print("Used Row size:", len(data))
    
    model_dic = {}
    for dat in data:
        wf = dat["true_wf"]
        sf = dat["subflow"]
        guideline = dat["guideline"]

        #wfs += [wf]
        #   sf_wfs += [sf+"_"+wf]

        result = {}

        fixed_model_name = model_names[0] #list(dat.keys())[-1] # so that it's fixed and not true_response
        for k,v in dat.items():
            if "context_" in k:
                model_name = k[len("context_"):] #].strip("context")
                model_input = cleanup(dat["input_"+fixed_model_name])

                compliance_score = dat["compliance_"+model_name]
                if model_name not in model_dic:
                    model_dic[model_name] =[ [v, guideline, wf, sf, model_input, compliance_score ] ]
                else:
                    model_dic[model_name] += [ [v, guideline, wf, sf, model_input, compliance_score ] ]

    dics = []
    for k,v in model_dic.items():
        # if k != "true_response":
        #     continue
        # print(k)
        gens = [ vv[0] for vv in v]
        guidelines = [ vv[1] for vv in v]
        wfs = [ vv[2] for vv in v]
        sfs = [ vv[3] for vv in v]
        inputs = [ vv[4] for vv in v]
        compliance_scores = [vv[5] for vv in v]

        if not context:
            #inputs = None
            results = asyncio.run(batch_evaluate_fluency(guidelines, gens, wfs, sfs, None))
        else:
            results = asyncio.run(batch_evaluate_fluency(guidelines, gens, wfs, sfs, inputs))
        print("="*30)
        print("Model:", k)
        print("Result:", np.average(results))
        #pearson = stats.pearsonr(compliance_scores, results)
        #spearman = stats.spearmanr(compliance_scores, results)
        #print("pearson:", pearson)
        #print("spearman:", spearman)
        print()

        dic = { "model_name":k, "input": inputs, "target": gens, "guidelines": guidelines, \
        "workflow":wfs, "subflows":sfs, "llm_scores":results,\
        "llm_average":np.average(results)}
        dics.append(dic)

    with open(f"./test_results/fluency_new_prompt_context-{context}_{LEN}.json", "w") as fh:
        json.dump(dics, fh, indent=4)

    
    return 

def breakdown_by_wf(result_file = "./test_results/mother_eval_results.json"):

    with open(result_file, "r") as fh:
        data =json.load(fh)

    stat = data[0]
    data = data[1:]

    print("Read:", result_file)
    print("Row size:", len(data))

    #wf_dic = {}
    #sf_wf_dic = {}
    wfs = []
    sf_wfs = []
    model_dic = {}
    for dat in data:
        wf = dat["true_wf"]
        sf = dat["subflow"]

        wfs += [wf]
        sf_wfs += [sf+"_"+wf]

        result = {}
        for k,v in dat.items():
            if "compliance" in k:
                model_name = k.strip("compliance_")
                if model_name not in model_dic:
                    model_dic[model_name] =[ [v, wf, sf+"_"+wf ] ]
                else:
                    model_dic[model_name] += [ [v, wf, sf+"_"+wf ] ]
    wfs = set(wfs)
    sf_wfs = set(sf_wfs)
    for k,v in model_dic.items():
        print("="*30)
        print(k)
        for wf in wfs:
            wf_sub = []
            for vv in v:
                if vv[1] == wf:
                    wf_sub += [vv[0]]
            print(f"Average for {wf}:", np.average(wf_sub))
            print("Sample size:", len(wf_sub))
        # for sf_wf in sf_wfs:
        #     sf_sub = []
        #     for vv in v:
        #         if vv[2] == sf_wf:
        #             sf_sub += [vv[0]]
        #     print(f"Average for {sf_wf}:", np.average(sf_sub))
        # print()
        
        


if __name__ == "__main__":
    #fire.Fire(breakdown_by_wf)
    fire.Fire(evaluate_compliance)
    # fire.Fire(evaluate_fluency) 