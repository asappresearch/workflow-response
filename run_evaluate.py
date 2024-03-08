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


from run_special_tokens_clm import process_with_tokens

# don't need now
# from trlx.model.nn.ilql_models import CausalLMWithValueHeads
# from trlx.data.configs import TRLConfig
# from trlx.utils.evaluate import prepare_model_input

# from dialogue_rl.utils.conversation import (
#     convo_to_context_response_pairs_abcd,
#     convo_to_context_response_pairs_multi_woz,
#     convo_to_context_response_pairs_taskmaster3,
# )
from model.constants import *

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

from itertools import chain, islice
def chunks(iterable, size=16):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

from functools import partial

def convo_to_context_response_pairs_workflow_response(
    sample: Dict,
    use_special_tokens: bool = False
):  
    #print(samples)
    #exit()
    context_response_pairs: List[Dict] = []
    #for idx, sample in enumerate(samples):
        #print(idx)
        #print(sample)
        #sender_type, utterance = row
        #if (sender_type == 'agent'):
        #    context = convo_to_text_abcd(utterance_rows[:idx])
    
    if "system: " in sample["target"]:
        context = sample["input"]+"\nsystem: "
        utterance = sample["target"][len("system: "):].strip()#.strip("system:").strip() 
        # utterance = sample["target"].strip("system:").strip()  # this was nuking "yes"
    elif "wf-action: " in sample["target"]:
        context = sample["input"]+"\nwf-action: "
        utterance = sample["target"][len("wf-action: "):].strip()#.strip("system:").strip() 
        # utterance = sample["target"].strip("system:").strip()  # this was nuking "yes"
    else:
        #print("context:", context)
        #print("target:", sample["target"])
        context = sample["input"]+"\n"
        utterance = sample["target"].strip()
    if False:
        print("="*30)
        print("context:",context)
        print("response:",utterance)

    dic = {'context': context, 'response': utterance}
    if use_special_tokens:
        text = dic["context"] + dic["response"]
        text = process_with_tokens(text)
        #print("text:", text)
        
        s = text.split("|>")
        s = [ x for x in s if x != ""]
        #print(s)
        a = s[-2]
        #print(a)
        if a.endswith("workflow"):

            try:
                # only considering workflow prediction here
                context = text.split(WORKFLOW)[0] + " " + WORKFLOW
                response = text.split(WORKFLOW)[1]#.split()[0]

            except:
                print("wot")
                exit(1)
        elif a.endswith("response"):
            try:
                # only considering workflow prediction here
                context = text.split(RESPONSE)[0] + " " + RESPONSE
                response = text.split(RESPONSE)[1]#.split()[0]
            except:
                print("wot")    
                exit(1)
        else:
            print("this should not happen gg")
            print(a)
            exit(1)        
        dic = {'context': context, 'response': response}
        print(dic)


    context_response_pairs.append(dic)
    
    new = []
    for dic in context_response_pairs:
        context = dic["context"]
        response = dic["response"]
        if response.strip() == "":
            print("="*30)
            print("Context: ", context)
            print("Empty reference: ", response)
        else:
            new +=  [dic]
    context_response_pairs = new
    
    df_context_response_pairs = pd.DataFrame(context_response_pairs)
    
    return df_context_response_pairs


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def load_tf(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_ilql(config_path):
    config = TRLConfig.load_yaml(config_path)       
    model = CausalLMWithValueHeads(
        config.model.model_path, ilql_config=config.method).eval().to(device)
    model_state_dict = torch.load(f"{config.train.checkpoint_dir}/pytorch_model.bin")
    model.load_state_dict(model_state_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)

    return model, tokenizer

def generate_dt(context, model, tokenizer, num_responses, context_len, response_len, scorer=False):
    context = context.replace(f"{REP_START}", f"{REWARD_ONE}{REP_START}")
    prompt = context.replace("> ", ">").replace(" <", "<")
    prompt = prompt + f"{REWARD_ONE}{REP_START}"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_attn_mask = inputs.attention_mask.to(device)
    input_ids = input_ids[:, -context_len:]
    input_attn_mask = input_attn_mask[:, -context_len:]
    
    #  Always add greedy output
    outputs = model.generate(
        input_ids,
        max_new_tokens=response_len,
        eos_token_id=tokenizer.encode(REP_END)[0],
        use_cache=True,
        attention_mask=input_attn_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    output_ids = outputs[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    if num_responses > 1:
        outputs = model.generate(
            input_ids,
            max_new_tokens=response_len,
            num_beams=num_responses-1,
            num_return_sequences=num_responses-1,
            eos_token_id=tokenizer.encode(REP_END)[0],
            use_cache=True,
            attention_mask=input_attn_mask,
            pad_token_id=tokenizer.eos_token_id
        )
        output_ids = outputs[:, input_ids.shape[-1]:]
        preds.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    # preds = [p.replace(REP_END, "") for p in preds]
    
    if scorer is True:
        response_score_list = []
        for pred in preds:
            response_ids = tokenizer(pred, return_tensors="pt").input_ids.to(device)
            context_response_ids = torch.cat((input_ids, response_ids), dim=1)
            logits = model(context_response_ids).logits
                    
            response_logits = logits[0, input_ids.shape[-1]-1:-1, :]
            response_scores = torch.index_select(response_logits, dim=1, index=response_ids.flatten())  # response_len x response_len
            response_scores = torch.diagonal(response_scores)  # response_len x 1
            
            score = torch.sum(response_scores).item()
            row_dict = {'score': score, 'response': pred}
            response_score_list.append(row_dict) 
        response_score_df = pd.DataFrame(response_score_list)
        response_score_df = response_score_df.sort_values(by = 'score', ascending=False)
        preds = response_score_df['response'].to_list()
    
    return preds


from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [198]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_tf(context, model, tokenizer, num_responses, context_len, response_len, scorer=False):
    prompt = context.replace("> ", ">").replace(" <", "<")
    #prompt = prompt + f"{REP_START}" # now the training involves this token


    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_attn_mask = inputs.attention_mask.to(device)
    input_ids = input_ids[:, -context_len:]
    input_attn_mask = input_attn_mask[:, -context_len:]

    outputs = model.generate(
        input_ids,
        max_new_tokens=response_len,
        eos_token_id=tokenizer.eos_token_id, #tokenizer.encode(REP_END)[0],
        use_cache=True,
        attention_mask=input_attn_mask,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]) # added
    )
    output_ids = outputs[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    if num_responses > 1:
        outputs = model.generate(
        input_ids,
        max_new_tokens=response_len,
        num_beams=num_responses-1,
        num_return_sequences=num_responses-1,
        eos_token_id=tokenizer.eos_token_id,#tokenizer.encode(REP_END)[0],
        use_cache=True,
        attention_mask=input_attn_mask,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]) # added
    )
        output_ids = outputs[:, input_ids.shape[-1]:]
        preds.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        # preds = [p.replace(REP_END, "") for p in preds]
    
    if scorer is True:
        response_score_list = []
        for pred in preds:
            response_ids = tokenizer(pred, return_tensors="pt").input_ids.to(device)
            context_response_ids = torch.cat((input_ids, response_ids), dim=1)
            logits = model(context_response_ids).logits
                    
            response_logits = logits[0, input_ids.shape[-1]-1:-1, :]
            response_scores = torch.index_select(response_logits, dim=1, index=response_ids.flatten())  # response_len x response_len
            response_scores = torch.diagonal(response_scores)  # response_len x 1
            
            score = torch.sum(response_scores).item()
            row_dict = {'score': score, 'response': pred}
            response_score_list.append(row_dict) 
        response_score_df = pd.DataFrame(response_score_list)
        response_score_df = response_score_df.sort_values(by = 'score', ascending=False)
        preds = response_score_df['response'].to_list()
            
    return preds
    
def generate_tf_batch(context, model, tokenizer, num_responses, context_len, response_len, scorer=False):
    bsz = len(context)
    prompt = [x.replace("> ", ">").replace(" <", "<") for x in context]
    #prompt = prompt + f"{REP_START}"

    """
    I added the following per https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2    
    """
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True)
    input_ids = inputs.input_ids.to(device)
    input_attn_mask = inputs.attention_mask.to(device)
    
    input_ids = input_ids[:, -context_len:]
    input_attn_mask = input_attn_mask[:, -context_len:]
    #input_ids = input_ids[:, -context_len:]
    #input_attn_mask = input_attn_mask[:, -context_len:]

    outputs = model.generate(
        input_ids,
        max_new_tokens=response_len,
        eos_token_id=tokenizer.eos_token_id, #tokenizer.encode(REP_END)[0],
        use_cache=True,
        attention_mask=input_attn_mask,
        pad_token_id=tokenizer.eos_token_id,
        #stopping_criteria=StoppingCriteriaList([StopOnTokens()]) # added
    )
    output_ids = outputs[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # print("preds")
    # print(preds)
    # print(len(preds))
    if num_responses > 1:
        outputs = model.generate(
        input_ids,
        max_new_tokens=response_len,
        num_beams=num_responses-1,
        num_return_sequences=num_responses-1,
        eos_token_id=tokenizer.eos_token_id, #tokenizer.encode(REP_END)[0],
        use_cache=True,
        attention_mask=input_attn_mask,
        pad_token_id=tokenizer.eos_token_id,
        #stopping_criteria=StoppingCriteriaList([StopOnTokens()]) # added
    )
        output_ids = outputs[:, input_ids.shape[-1]:]
        #preds.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        add_preds =tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print("add_preds")
        # print(add_preds)
        # print(len(add_preds))
        #preds.extend(add_preds)
        
        new_preds = []
        count = 0
        for chnk in chunk(add_preds, size=num_responses-1):
            new_preds += [ preds[count]]
            new_preds += list(chnk)
            count += 1
        preds = new_preds
        #print(preds)
        # preds = [p.replace(REP_END, "") for p in preds]
    
    # Note: post processing because with batch stopping criteria trickier just gen 
    preds = [ pred.split("\n")[0] for pred in preds]
    #print(preds)

    # TODO: when use_special_tokens, need to cut by RESPONSE_END or WORKFLOW_END

    if scorer is True:
        response_score_list = []
        for pred in preds:
            response_ids = tokenizer(pred, return_tensors="pt").input_ids.to(device)
            context_response_ids = torch.cat((input_ids, response_ids), dim=1)
            logits = model(context_response_ids).logits
                    
            response_logits = logits[0, input_ids.shape[-1]-1:-1, :]
            response_scores = torch.index_select(response_logits, dim=1, index=response_ids.flatten())  # response_len x response_len
            response_scores = torch.diagonal(response_scores)  # response_len x 1
            
            score = torch.sum(response_scores).item()
            row_dict = {'score': score, 'response': pred}
            response_score_list.append(row_dict) 
        response_score_df = pd.DataFrame(response_score_list)
        response_score_df = response_score_df.sort_values(by = 'score', ascending=False)
        preds = response_score_df['response'].to_list()
            
    return preds

def generate_ilql(context, model, tokenizer, num_responses, context_len, response_len):
    prompt = context.replace("> ", ">").replace(" <", "<")
    prompt = prompt + f"{REP_START}"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_attn_mask = inputs.attention_mask.to(device)
    input_ids = input_ids[:, -context_len:]
    input_attn_mask = input_attn_mask[:, -context_len:]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=input_attn_mask,
        beta=0.,
        temperature=1e-6,
        max_length=context_len+response_len,
        eos_token_id=tokenizer.encode(REP_END)[0],
        pad_token_id=tokenizer.eos_token_id
    )

    output_ids = outputs[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # preds = [p.replace(REP_END, "") for p in preds]
    
    return preds

def generate_ilql_scorer(context, model, tokenizer, num_responses, context_len, response_len):
    prompt = context.replace("> ", ">").replace(" <", "<")
    prompt = prompt + f"{REP_START}"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_attn_mask = inputs.attention_mask.to(device)
    input_ids = input_ids[:, -context_len:]
    input_attn_mask = input_attn_mask[:, -context_len:]

    response_score_list = []
    for k in range(num_responses):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=input_attn_mask,
            beta=1.,
            temperature=1,
            max_length=context_len+response_len,
            eos_token_id=tokenizer.encode(REP_END)[0],
            pad_token_id=tokenizer.eos_token_id
        )
        output_ids = outputs[:, input_ids.shape[-1]:]
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        context_w_repstart = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        context_response = [f"{context_w_repstart}{SPLIT_TOKEN}{preds}{REP_END}"]
        context_response_ids, context_response_mask, actions_ixs, states_ixs = prepare_model_input(tokenizer, samples=context_response, split_token=SPLIT_TOKEN, device=device)
        _, _, _, vs, _ = model(input_ids=context_response_ids, attention_mask=context_response_mask, actions_ixs=actions_ixs, states_ixs=states_ixs)
        V = vs[:, :-1].view(-1) # vs: 1 x K x 1
        V_terminal = V[-1].item()
        row_dict = {'score': V_terminal, 'response': preds}
        response_score_list.append(row_dict) 
    # preds = [p.replace(REP_END, "") for p in preds]
    response_score_df = pd.DataFrame(response_score_list)
    response_score_df = response_score_df.sort_values(by = 'score', ascending=False)
    
    return response_score_df['response'].to_list()   

def generate_ilql_scorer_on_tf(context, model, model_tf, tokenizer, tokenizer_tf, num_responses, context_len, response_len):
    preds_tf = generate_dt(context=context, model=model_tf, tokenizer=tokenizer_tf, num_responses=num_responses, context_len=context_len, response_len=response_len)    
    prompt = context.replace("> ", ">").replace(" <", "<")
    prompt = prompt + f"{REP_START}"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_ids = input_ids[:, -context_len:]
    
    response_score_list = []
    for pred in preds_tf:
        context_w_repstart = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        context_response = [f"{context_w_repstart}{SPLIT_TOKEN}{pred}{REP_END}"]
        context_response_ids, context_response_mask, actions_ixs, states_ixs = prepare_model_input(tokenizer, samples=context_response, split_token=SPLIT_TOKEN, device=device)
        logits, _, _, vs, _ = model(input_ids=context_response_ids, attention_mask=context_response_mask, actions_ixs=actions_ixs, states_ixs=states_ixs)
        V = vs[:, :-1].view(-1) # vs: 1 x K x 1
        V_terminal = V[-1].item()
        
        # logits_input = torch.diag(logits.index_select(dim=2, index=context_response_ids[0, :])[0, :])
        # logits_response = logits_input[-vs.shape[1]:]
        # logits_score = torch.sum(logits_response) 
        row_dict = {'score': V_terminal, 'response': pred}
        response_score_list.append(row_dict) 

    response_score_df = pd.DataFrame(response_score_list)
    response_score_df = response_score_df.sort_values(by = 'score', ascending=False)
    
    return response_score_df['response'].to_list()   

def evaluate_model(
    method: str = "tf",
    scorer: bool = False,
    dataset: str = "workflow-response",
    metrics:  List[str] = ["bert_score", "bleurt_score", "meteor", "bleu"],
    data_path: str = "data/b1.json",
    split: str = 'test',
    model_path: str = None,
    config_path: str = None,
    num_samples: int = 1000,
    num_responses: int = 1,
    context_len: int = 96,
    response_len: int = 32,
    bert_thresh: float = 0.6,
    save_path: Optional[str] = None,
    do_batch: bool = True,
    batch_size: int = 16,
    cascade_datapath: str = "./test_results/wf_prediction_epochs4/evaluation_tf.csv", #"./test_results/wf_prediction/evaluation_tf.csv",
    use_special_tokens: bool = False
):  
    #print("path:", path)
    for path in [data_path, model_path, config_path, save_path]:
        if path is None: continue    
        if not os.path.isabs(path): path = path #f"{BASE_PATH}/{path}"
            
    with open(data_path, 'r') as f:
        data = json.load(f)

    # print(dataset)
    # print("="*30)
    with Pool() as p:
        if dataset == "abcd":
            for path in [data_path, save_path, model_path]:    
                if path is None: continue
                if not os.path.isabs(path): path = f"{BASE_PATH}/{path}"

            with open(data_path, 'r') as f:
                data = json.load(f)
            data_split = "dev" if (split == "val") else split
            context_response_pairs_list = list(tqdm(p.imap(convo_to_context_response_pairs_abcd, [row['original'] for row in data[data_split]]), total=len(data[data_split])))
        elif dataset == "multi_woz":
            data = load_dataset("multi_woz_v22")
            data_split = "validation" if (split == "val") else split
            context_response_pairs_list = list(tqdm(p.imap(convo_to_context_response_pairs_multi_woz, data[data_split]), total=len(data[data_split])))
        elif dataset == "taskmaster3":
            data = load_dataset("taskmaster3")['train'] # all data are in the 'train' split
            # split by 80%, 10%, 10%
            indices = np.random.RandomState(42).permutation(len(data)).tolist()
            indices = {
                'train': indices[:int(len(data) * 0.8)],
                'val': indices[int(len(data) * 0.8):int(len(data) * 0.9)],
                'test': indices[int(len(data) * 0.9):],
            }
            data = {k: [data[i] for i in indices[k]] for k in indices}
            context_response_pairs_list = list(tqdm(p.imap(convo_to_context_response_pairs_taskmaster3, data[split]), total=len(data[split])))
        elif dataset == "workflow-response":
            data_split = "dev" if (split == "val") else split
            #print(data[data_split])
            func = partial(convo_to_context_response_pairs_workflow_response, use_special_tokens=use_special_tokens)
            context_response_pairs_list = list(tqdm(p.imap(func, data[data_split]), total=len(data[data_split])))
        elif dataset == "wf-cascade1":
            """
            This should be called first, then after this wf-cascade2
            This prepares the data with necessary gt response for the cascade2 utt prediction model
            """
            with open("data/utt_prediction.json", 'r') as f:
                data = json.load(f)
            data_split = "dev" if (split == "val") else split
            #print(data[data_split])
            func = partial(convo_to_context_response_pairs_workflow_response, use_special_tokens=use_special_tokens)
            context_response_pairs_list = list(tqdm(p.imap(func, data[data_split]), total=len(data[data_split])))
            context_response_pairs = pd.concat(context_response_pairs_list)
            new = []
            for idx, pair in context_response_pairs.iterrows():
                context = pair["context"]
                response = pair["response"]
                if "wf-action: " not in context:
                    print("error!")
                    exit()
                new_response = context.split("wf-action: ")[-1].strip()[:-len("system:")].strip()
                new_context = context.split("wf-action: ")[0].strip().strip("\n")+"\nwf-action: "
                #context = sample["input"]+"\nsystem: "
                #utterance = sample["target"][len("system: "):].strip()#.strip("system:").strip() 
                dic = {
                    "context": new_context,
                    "response": new_response,
                    "cont_response": response
                }
                new.append(dic)
                #print(dic) #
            context_response_pairs = pd.DataFrame(new)

        elif dataset == "wf-cascade2":
            pairs = []
            import csv
            with open(cascade_datapath, 'r') as data:
                for line in csv.DictReader(data):
                    context = line["context"]+line["response_1"].strip()+"\nsystem: "
                    response = line["cont_response"]
                    true_wf = line["true_response"]
                    dic = { "context": context, "response": response, "true_wf": true_wf }
                    #print(dic) #
                    pairs += [dic]

            context_response_pairs = pd.DataFrame(pairs)

        else:
            raise NotImplementedError(f"{dataset=}")
    
    if dataset != "wf-cascade1" and dataset != "wf-cascade2":
        context_response_pairs = pd.concat(context_response_pairs_list)

    if False:
        import random
        rint = random.randint(0, len(context_response_pairs_list)-1)
        print(context_response_pairs_list[rint])
        exit()

    if num_samples is not None and dataset != "wf-cascade2":
        # wf-cascade2 is excluded because it operates on output of wf-cascade1
        set_seed(42)
        context_response_pairs = context_response_pairs.sample(n=num_samples, random_state=1)
        
    print("loading model")
    if (method == 'tf'):
        model, tokenizer = load_tf(model_path)
        if not do_batch:
            def generate_fn(context): return generate_tf(
                context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len, scorer=scorer)
        else:
            def generate_fn(context): return generate_tf_batch(
                context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len, scorer=scorer)
       
    elif (method == 'tf_top'):
        model, tokenizer = load_tf(model_path)
        def generate_fn(context): return generate_tf(
            context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len, scorer=scorer)
    elif (method == 'tf_all'):
        model, tokenizer = load_tf(model_path)
        def generate_fn(context): return generate_tf(
            context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len, scorer=scorer)
    elif (method == 'dt'):
        model, tokenizer = load_tf(model_path)
        def generate_fn(context): return generate_dt(
            context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len, scorer=scorer)
    elif (method == 'ilql'):
        model, tokenizer = load_ilql(config_path)
        def generate_fn(context): return generate_ilql(
            context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len)
    elif (method == 'ilql_scorer'):
        model_tf, tokenizer_tf = load_tf(model_path)
        model, tokenizer = load_ilql(config_path)
        def generate_fn(context): return generate_ilql_scorer_on_tf(
            context=context, model=model, model_tf=model_tf, tokenizer=tokenizer, tokenizer_tf=tokenizer_tf, num_responses=num_responses, context_len=context_len, response_len=response_len)
    elif (method == 'tf_batch'):
        model, tokenizer = load_tf(model_path)
        def generate_fn(context): return generate_tf_batch(
            context=context, model=model, tokenizer=tokenizer, num_responses=num_responses, context_len=context_len, response_len=response_len, scorer=scorer)
   
    else:
        print(f"method {method} not found")
        return

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"# of param: {num_param / 10**6:.2f} M")

    all_pred_responses = []
    all_gold_responses = []

    if not do_batch:
        for idx, row in tqdm(context_response_pairs.iterrows(), total=context_response_pairs.shape[0]):
            preds = generate_fn(context=row['context'])
            all_gold_responses.append(row['response'])
            all_pred_responses.append(preds)
        
    else:
        # pre walk
        cr = []
        for idx, row in tqdm(context_response_pairs.iterrows(), total=context_response_pairs.shape[0]):
            context = row["context"]
            response = row["response"]
            cr += [ {"context": context, "response": response}]

        for idx, row in tqdm(enumerate(chunk(cr, size=batch_size)), total=len(cr)//batch_size + 1):
            #print(idx)
            #print(len(row))
            contexts = [ x["context"] for x in row]
            responses = [ x["response"] for x in row ]
            preds = generate_fn(context=contexts)
            all_gold_responses.extend(responses)
            all_pred_responses.extend(preds)

        all_pred_responses = list(chunk(all_pred_responses, size=num_responses))
        #preds_flat = all_pred_responses #[p for pred_responses in all_pred_responses for p in pred_responses]
        #gold_flat = [gold_response for gold_response in all_gold_responses for i in range(num_responses)]
 
    preds_flat = [p for pred_responses in all_pred_responses for p in pred_responses]
    gold_flat = [gold_response for gold_response in all_gold_responses for i in range(num_responses)]

    if use_special_tokens:
        p,g = [], []
        for pp,gg in zip(preds_flat, gold_flat):
            pp = pp.replace(WORKFLOW_END,"").strip()
            pp = pp.replace(SYSTEM_END,"").strip()

            gg = gg.replace(WORKFLOW_END,"").strip()
            gg = gg.replace(SYSTEM_END,"").strip()
            p.append(pp)
            g.append(gg)
        pred_flat, gold_flat = p, g    

    # print(all_pred_responses)
    # print("="*30)
    # for g, p in zip(gold_flat, preds_flat):
    #     print("-"*30)
    #     print(g)
    #     print(p)

    # This is not the way to handle it ==> must handle it in bleu computation    
    # p, g = [], []
    # for pred, gold in zip(preds_flat, gold_flat):
    #     if pred.strip() == "" or gold.strip() == "":
    #         pass
    #     else:
    #         p.append(pred)
    #         g.append(gold)

    # preds_flat, gold_flat = p, g

    all_metric_values: Dict[str, List] = {} # metric -> metric_values (list of lists, n_examples x n_responses)
    for metric in metrics:
        print(metric)
        if metric == 'bert_score':
            print("loading bert scorer")
            scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            P, R, F1 = scorer.score(preds_flat, gold_flat)
            bert_f1_flat = F1.tolist() # list of (n_examples * n_responses) elements
            bert_f1_values = [bert_f1_flat[i:i+num_responses] for i in range(0, len(bert_f1_flat), num_responses)]
            all_metric_values[metric] = bert_f1_values
        elif (metric == 'bleurt_score'):
            print("loading bleurt scorer")
            bleurt = evaluate.load("bleurt", module_type="metric")#evaluate.load("bleurt", "BLEURT-20")
            bleurt_flat = bleurt.compute(predictions=preds_flat, references=gold_flat)['scores']
            bleurt_values = [bleurt_flat[i:i+num_responses] for i in range(0, len(bleurt_flat), num_responses)]
            all_metric_values[metric] = bleurt_values
        elif (metric == 'perplexity'):
            perplexity = evaluate.load("perplexity", module_type="metric")
            perplexity_flat = perplexity.compute(predictions=preds_flat, model_id='gpt2')['perplexities']
            perplexity_values = [perplexity_flat[i:i+num_responses] for i in range(0, len(perplexity_flat), num_responses)]
            all_metric_values[metric] = perplexity_values
        elif (metric == 'exact_match'):
            em = [1.0 if  p.strip()== g.strip() else 0.0 for p,g in zip(preds_flat, gold_flat)]
            em = [em[i:i+num_responses] for i in range(0, len(em), num_responses)]
            all_metric_values[metric] = em
        else: # "meteor", "bleu"
            evaluate_metric = evaluate.load(metric)
            metric_values_flat = []
            for pred, gold in zip(preds_flat, gold_flat):
                #pred = "\n"
                results = evaluate_metric.compute(predictions=[pred], references=[gold])[metric] if pred.strip() != "" else 0.
                metric_values_flat.append(results)
            metric_values = [metric_values_flat[i:i+num_responses] for i in range(0, len(metric_values_flat), num_responses)]
            all_metric_values[metric] = metric_values
            
    evaluation_data = []
    for cidx, (_, row) in enumerate(context_response_pairs.iterrows()):
        #print(row)
        row_dict = {}
        row_dict['context'] = row['context']
        row_dict['true_response'] = row['response']
        if "cont_response" in row:
            row_dict["cont_response"] = row["cont_response"]
        if "true_wf" in row:
            row_dict["true_wf"] = row["true_wf"]

        for ridx in range(0, num_responses):
            #print(cidx, ridx, len(all_pred_responses), len(all_pred_responses[cidx]))
            #print(all_pred_responses[cidx])
            row_dict[f'response_{ridx+1}'] = all_pred_responses[cidx][ridx]
            for metric in metrics:
                row_dict[f'{metric}_{ridx+1}'] = all_metric_values[metric][cidx][ridx]
        evaluation_data.append(row_dict)
        
    # Create eval dataframe
    evaluation_df = pd.DataFrame(evaluation_data)

    # Compute topk over num_responses
    print("=" * 80)
    for topk in range(1, num_responses+1):
        print(f"{method}, top {topk}:")
        for metric in metrics:
            metric_column_list = [f'{metric}_{ridx+1}' for ridx in range(0, topk)]
            metric_topk = evaluation_df[metric_column_list].max(axis=1)
            print(f'{metric}: {metric_topk.mean()}')
            if metric=='bert_score':
                print(f'bert_click: {(metric_topk > bert_thresh).mean()}')
    print("=" * 80)

    # Save to csv
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        evaluation_df.to_csv(f"{save_path}/evaluation_{method}.csv")
        print(f"Saved evaluation results to {save_path}/evaluation_{method}.csv")

if __name__ == "__main__":
    fire.Fire(evaluate_model)