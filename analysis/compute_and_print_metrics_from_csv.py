import csv
import numpy as np
import fire
import glob

METRICS = [ "bert_score", "bleurt_score", "meteor", "bleu", "exact_match" ]

def compute_and_print(path = "230616test_results/wf_prediction_epochs2/evaluation_tf.csv"):
    
    with open(path, "r") as fh:

        scores = { metric:[] for metric in METRICS}
        
        for line in csv.DictReader(fh):
            for i in range(1,2):
                for metric in METRICS:
                    name = f"{metric}_{i}"
                    #print(name, line)
                    if name in line:
                        #print("asd")
                        scores[metric] += [ float(line[name])]
    
    
    for k,v in scores.items():
        print(k, np.mean(v))


def display_scores(folder_path = "test_results/"):
    paths = glob.glob(folder_path+"/**/*.csv")
    print(paths)
    for path in paths:
        print("="*30)
        print(path)
        compute_and_print(path)


import os
import json
import fire

from typing import Optional
from multiprocessing import Pool
from typing import List, Dict
from tqdm.auto import tqdm, trange


from bert_score import BERTScorer
import evaluate 
from datasets import load_dataset

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import pandas as pd



def evaluat_openai(path = "open_ai_results.json", metrics=["bert_score", "bleurt_score", "meteor", "bleu"]):

    with open(path, "r") as fh:
        result = json.load(fh)

        avg = np.average([x["response"]["usage"]["prompt_tokens"] for x in result])
        print(avg)
        #exit()

    preds_flat, gold_flat = [], []
    for r in result:
        finish_reason = r["response"]["choices"][0]["finish_reason"]
        if finish_reason == "stop":
        #if True:
            preds_flat += [r["response"]["choices"][0]["text"].lower()]
            gold_flat += [ r["target"]]


    #preds_flat = [ x["response"]["choices"][0]["text"].lower() for x in result]
    #gold_flat = [ x["target"] for x in result]

    print("Len:", len(preds_flat))
 
    #preds_flat = [p for pred_responses in all_pred_responses for p in pred_responses]
    #all_gold_responses = [gold_response for gold_response in all_gold_responses  ]#for i in range(num_responses)]

    p,g = [], []
    for pp,gg in zip(preds_flat, gold_flat):
        pp = pp.split("\n")[0].strip()
        gg = gg.strip()
        p.append(pp)
        g.append(gg)
    pred_flat, gold_flat = p, g    



    #all_metric_values: Dict[str, List] = {} # metric -> metric_values (list of lists, n_examples x n_responses)
    for metric in metrics:
        print(metric)
        if metric == 'bert_score':
            print("loading bert scorer")
            scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            P, R, F1 = scorer.score(preds_flat, gold_flat)
            bert_f1_flat = F1.tolist() # list of (n_examples * n_responses) elements
            #bert_f1_values = [bert_f1_flat[i:i+num_responses] for i in range(0, len(bert_f1_flat), num_responses)]
            #all_metric_values[metric] = bert_f1_values
            res = bert_f1_flat
        elif (metric == 'bleurt_score'):
            print("loading bleurt scorer")
            bleurt = evaluate.load("bleurt", "BLEURT-20")
            bleurt_flat = bleurt.compute(predictions=preds_flat, references=gold_flat)['scores']
            #bleurt_values = [bleurt_flat[i:i+num_responses] for i in range(0, len(bleurt_flat), num_responses)]
            #all_metric_values[metric] = bleurt_values
            res = bleurt_flat
        elif (metric == 'perplexity'):
            perplexity = evaluate.load("perplexity", module_type="metric")
            perplexity_flat = perplexity.compute(predictions=preds_flat, model_id='gpt2')['perplexities']
            #perplexity_values = [perplexity_flat[i:i+num_responses] for i in range(0, len(perplexity_flat), num_responses)]
            #all_metric_values[metric] = perplexity_values
            res = perplexity_flat
        elif (metric == 'exact_match'):
            em = [1.0 if  p.strip()== g.strip() else 0.0 for p,g in zip(preds_flat, gold_flat)]
            res = em
            #em = [em[i:i+num_responses] for i in range(0, len(em), num_responses)]
            #all_metric_values[metric] = em
        else: # "meteor", "bleu"
            evaluate_metric = evaluate.load(metric)
            metric_values_flat = []
            for pred, gold in zip(preds_flat, gold_flat):
                #pred = "\n"
                results = evaluate_metric.compute(predictions=[pred], references=[gold])[metric] if pred.strip() != "" else 0.
                metric_values_flat.append(results)
            res = metric_values_flat
            #metric_values = [metric_values_flat[i:i+num_responses] for i in range(0, len(metric_values_flat), num_responses)]
            #all_metric_values[metric] = metric_values
        
        print(np.average(res))
        print(np.std(res))
        print()

if __name__ == "__main__":
    fire.Fire(evaluat_openai)
    #fire.Fire(compute_and_print)
    #fire.Fire(display_scores)
