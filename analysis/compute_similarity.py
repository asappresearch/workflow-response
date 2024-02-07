import json

from bert_score import BERTScorer
import evaluate 

from tqdm import tqdm
import numpy as np

EVAL_BATCHSIZE = 8
# result_path = "./test_results/models_test_results.json"
result_path = "test_results/llm_block_test_quark_eval_results.json"

def chunk(l, size=16):
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

with open(result_path, "r") as fh:
    data = json.load(fh)

stat = data[0]
# data = data[1:][:2]
data = data[1:]

eval_paths = list(stat.keys()) #+ ["true_response"]
# TODO: eval loop
eval_data = {}
# for k in output_paths[1:] + quark_paths + ["true_response"]: #[ "quark","true_response"]:
#for k in output_paths+ quark_paths + ["true_response"]: #[ "quark","true_response"]:
for k in eval_paths: #[ "quark","true_response"]:
    eval_data[k] = []
    for dat in data:
        eval_data[k].append(dat)


bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
bleurt_scorer = evaluate.load("bleurt", "BLEURT-20")
meteor_scorer = evaluate.load("meteor")
bleu_scorer = evaluate.load("bleu")

eval_result = {}

for k, v in eval_data.items():
    # if k not in quark_paths:
    #    continue
    #v = v[:10]
    count = 0
    print("Evaluating ", k)
    eval_result[k] = { 
    "bert":[],
    "meteor": [],
    "bleu": [],
    "bleurt": [],
    } 
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

       



        bert, meteor, bleu, bleurt = [], [ ], [], []
            
        for pred, gt in zip(pred_system, gt_system):
            bert_scores, meteor_scores, bleu_scores, bleurt_scores = [], [], [], []
            for p in pred:
                preds_flat = [ p for g in gt ]

                P, R, F1 = bert_scorer.score(preds_flat, gt)
                F1 = F1.tolist()
                bert_score = max(F1)
                bert_scores.append(bert_score)

                bleurt_sc = bleurt_scorer.compute(predictions=preds_flat, references=gt)['scores']
                bleurt_score = max(bleurt_sc)
                bleurt_scores.append(bleurt_score)

                bs, ms = [], []
                for pp, gold in zip(preds_flat, gt):
                    #pred = "\n"
                    results = bleu_scorer.compute(predictions=[pp], references=[gold])["bleu"] if pp.strip() != "" else 0.
                    bs.append(results)
                    results = meteor_scorer.compute(predictions=[pp], references=[gold])["meteor"] if pp.strip() != "" else 0.
                    ms.append(results)

                bleu_score = max(bs)
                bleu_scores.append(bleu_score)
                meteor_score = max(ms)
                meteor_scores.append(meteor_score)

            bert.append(np.average(bert_scores))        
            bleurt.append(np.average(bleurt_scores))
            bleu.append(np.average(bleu_scores))
            meteor.append(np.average(meteor_scores))
        

        eval_result[k]["bert"] += bert
        eval_result[k]["meteor"] += meteor
        eval_result[k]["bleurt"] += bleurt
        eval_result[k]["bleu"] += bleu


total = { k:{kk:np.average(vv) for kk,vv in v.items() } for k,v in eval_result.items() }

for k, v in eval_result.items():
    print(k)
    for kk, vv in v.items():
        print(f"{kk}: {np.average(vv)}")
    print()
