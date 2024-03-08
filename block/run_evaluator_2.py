"""
this script is for running the evaluator model on A1, B2

- Need to get output from A1, B2 evaluation_tf.csv s
"""

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import csv, json
#from constants import SPECIAL_TOKEN_SET
from model.constants import *


def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]


#eval_model_path = "./save/evaluator_ab_rand/230706/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch5-ws0-gas1-1gpu/"
#eval_model_path = "./save/evaluator/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_no_neg/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_no_random/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs128-epoch10-ws0-gas1-1gpu/"

#eval_model_path = "./save/evaluator/230705/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs64-epoch10-ws0-gas1-1gpu/"

#eval_model_path = "./save/evaluator_scorer/230707/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs64-epoch10-ws0-gas1-1gpu/"
# eval_model_path = "./save/evaluator_scorer/230707/roberta-base/evaluator-roberta-base-tf-lr1e-4-bs16-epoch10-ws0-gas1-1gpu/"


# device = torch.device("cuda")
# evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(device)
# # evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path, torch_dtype=torch.float16).to(device)
# eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
 
from nltk import sent_tokenize

p,n,r = [],[],[]
with open("./data/generated_negatives_train_with_none_with_random.json", "r") as fh:
    data = json.load(fh)
    #print(data)
    eval_data = []
    for dat in data:
        try:
            dialog = dat["dialog"]
        except:
            continue
        neg = dat["generated"]
        neg = sent_tokenize(dat["generated"])
        if len(neg) > 1:
            neg = neg[0]
            #neg = " ".join(neg[:-1])
        else:
            neg = " ".join(neg)
        pos = dat["pos_response"]
        rand = dat["random_response"]
        wf = dat["original_action"]
        
        p += [pos]
        n += [neg]
        r += [rand]
        # print(dialog)
        # print(neg)
        # input()
        eval_data += [ dialog.strip() + "\nAgent: "+rand +"\nWorkflow-Action: "+wf]

import numpy as np

lenp = [ len(x.split()) for x in p]
lenn = [ len(x.split()) for x in n]
lenr = [ len(x.split()) for x in r]

l = [lenp, lenn, lenr]

for ll in l:
    print(np.average(ll))
    print(np.std(ll))
exit()
print(eval_data[0])

scores, preds = [], []
for batch in chunk(eval_data, size=16):
    tokenized = eval_tok(batch, truncation=True, padding="longest", return_tensors="pt").to(device)

    output = evaluator(**tokenized)

    # p = output.logits.argmax(-1)
    # s = output.logits.softmax(-1)

    p = output.logits.sigmoid().flatten()
    s = output.logits.sigmoid().flatten()    
    #print(scores.shape)
    # print(output.logits.shape)
    # print(preds.shape)
    # input()
    preds += p.tolist()
    scores += s.tolist() #
    # scores += [ x[1] for x in s.tolist() ] 

import numpy as np

print(np.average(preds))
print(np.average(scores))