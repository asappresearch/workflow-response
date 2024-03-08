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
with open("./data/kb.json", "r") as fh:
    kb = json.load(fh)   

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

from itertools import chain, islice
def chunks(iterable, size=16):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

PRINT = False #  True
def convo_to_context_response_pairs_workflow_response(dataset_type: str, datum):
    """{'sample_id': 2, 'convo_id': 3695, 'turns': [{'speaker': 'user', 'text': 'hey ho!', 'turn_count': 1, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'good afternoon, how can i help you?', 'turn_count': 2, 'targets': ['timing', 'retrieve_utterance', None, [], 84], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "i've got a promo code and i want to know when they expire.", 'turn_count': 3, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "i'd like to use it to buy some hats for my cat.", 'turn_count': 4, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'sure!  let me check that.', 'turn_count': 5, 'targets': ['timing', 'retrieve_utterance', None, [], 16], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'one moment please', 'turn_count': 6, 'targets': ['timing', 'retrieve_utterance', None, [], 26], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "some people think it's funny to put hats on cats...i do not feel that way.", 'turn_count': 7, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'cats deserve to look good too', 'turn_count': 8, 'targets': ['timing', 'retrieve_utterance', None, [], 54], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': 'exactly!', 'turn_count': 9, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'ok, just to verify you already tried to use the code?', 'turn_count': 10, 'targets': ['timing', 'retrieve_utterance', None, [], 77], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': 'no, i just want to see how long they last for.', 'turn_count': 11, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'ok, sorry for the doubt and i will answer your question.', 'turn_count': 12, 'targets': ['timing', 'retrieve_utterance', None, [], 43], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'one moment please', 'turn_count': 13, 'targets': ['timing', 'retrieve_utterance', None, [], 65], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'action', 'text': 'searching the faq pages ...', 'turn_count': 14, 'targets': ['timing', 'take_action', 'search-faq', [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'action', 'text': 'system action: search timing', 'turn_count': 15, 'targets': ['timing', 'take_action', 'search-timing', [], -1], 'workflow_action': ['timing', 'take_action', 'search-timing', [], -1]}, {'speaker': 'action', 'text': 'faq answer related to timing (question4) was selected.', 'turn_count': 16, 'targets': ['timing', 'take_action', 'select-faq', ['timing_4'], -1], 'workflow_action': ['timing', 'take_action', 'select-faq', ['timing_4'], -1]}, {'speaker': 'system', 'text': 'ok, all promo codes expire after 7 days without fail.', 'turn_count': 17, 'targets': ['timing', 'retrieve_utterance', None, [], 9], 'workflow_action': None}, {'speaker': 'user', 'text': 'perfect. thanks', 'turn_count': 18, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'not problem! a pleasure to help you and your cat too', 'turn_count': 19, 'targets': ['timing', 'retrieve_utterance', None, [], 5], 'workflow_action': None}, {'speaker': 'user', 'text': "that's all, have a great day! don't forget to spay or neuter your pet!", 'turn_count': 20, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'have a nice day', 'turn_count': 21, 'targets': ['timing', 'retrieve_utterance', None, [], 58], 'workflow_action': None}, {'speaker': 'system', 'text': "i won't", 'turn_count': 22, 'targets': ['timing', 'retrieve_utterance', None, [], 1], 'workflow_action': None}]}
    """
    context_response_pairs: List[Dict] = []
    
    strings = []

    turns = datum["turns"]
    flow = turns[0]["targets"][0]
    if dataset_type == "kb":       
        kb_flow = kb[flow]  
        string = CONTEXT +WORKFLOW +", ".join([str(x) for x in kb_flow]) +    WORKFLOW_END
    else:
        string = CONTEXT
    
    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"]

        if speaker == "user":
            string += USER +  text + USER_END 
        elif speaker == "action":
            if dataset_type == "b1":
                pass
            else:
                button = turn["targets"][2] 
                slot =  turn["targets"][3]
                string += ACTION  +button +  " " + ", ".join(slot).strip() + ACTION_END 
        elif speaker == "system":
            if dataset_type == "b1" or dataset_type == "b2" or dataset_type == "kb":
                string += RESPONSE +  text + RESPONSE_END 
            elif "future" in dataset_type:
                workflow = turn["workflow_action"]
                if workflow != [None]:
                    workflow = [x[2] if x is not None else x for x in workflow ]
                string += WORKFLOW +", ".join([str(x) for x in workflow]) +    WORKFLOW_END +RESPONSE + text +  RESPONSE_END 
            else:
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
                string += WORKFLOW +  str(workflow) +   WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
            strings.append(string)
        else:
            print("impossib")
            exit()
        
    for string in strings:
        context = RESPONSE.join(string.split(RESPONSE)[:-1]).strip() +RESPONSE
        response = string.split(RESPONSE)[-1].strip()#.strip(RESPONSE_END).strip()
        response = response.replace(RESPONSE_END,"")
        dic = {"context": context, "response": response, "subflow":flow}
        if PRINT:
            print(dic)
        context_response_pairs.append(dic)
        if context.strip().endswith(USER_END):
            print(context)
            exit()
    
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

from functools import partial



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
    tokenizer.truncation_side = 'left'
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True)
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
        #stopping_criteria=StoppingCriteriaList([StopOnTokens()]) # added
    )
    output_ids = outputs[:, input_ids.shape[-1]:]
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
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
        add_preds =tokenizer.batch_decode(output_ids, skip_special_tokens=False)
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
    for stoken in SPECIAL_TOKEN_SET:
        preds = [ pred.split(stoken)[0] for pred in preds]
        #preds = [ pred.split(WORKFLOW_END)[0] for pred in preds]
        #print(preds)

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
    dataset_type = "b1",
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
    cascade_datapath: str = "./test_results/230809/dist_st/wf_prediction_epochs10/epoch10/evaluation_tf.csv", #"./test_results/230801/gpt2_st/wf_prediction_epochs10/epoch10/evaluation_tf.csv"
    #cascade_datapath: str = "./test_results/dist_st/wf_prediction_epochs10/epoch10/evaluation_tf.csv", #"./test_results/wf_prediction/evaluation_tf.csv",
    #use_special_tokens: bool = False
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
            func = partial(convo_to_context_response_pairs_workflow_response, dataset_type)
            context_response_pairs_list = list(tqdm(p.imap(func, data[data_split]), total=len(data[data_split])))

        elif dataset == "wf-cascade1":
            """
            This should be called first, then after this wf-cascade2
            This prepares the data with necessary gt response for the cascade2 utt prediction model
            """
            # with open("data/utt_prediction.json", 'r') as f:
            #     data = json.load(f)
            # data_split = "dev" if (split == "val") else split
            # #print(data[data_split])
            # func = partial(convo_to_context_response_pairs_workflow_response, dataset_type)
            # context_response_pairs_list = list(tqdm(p.imap(func, data[data_split]), total=len(data[data_split])))
            data_split = "dev" if (split == "val") else split
            #print(data[data_split])
            func = partial(convo_to_context_response_pairs_workflow_response, dataset_type)
            context_response_pairs_list = list(tqdm(p.imap(func, data[data_split]), total=len(data[data_split])))
            context_response_pairs = pd.concat(context_response_pairs_list)
            new = []
            for idx, pair in context_response_pairs.iterrows():
                context = pair["context"]
                response = pair["response"]
                if WORKFLOW not in context:
                    print("error!")
                    exit()
                #new_response = context.split(WORKFLOW)[-1].strip()[:-len(RESPONSE)].strip()
                #new_context = context.split(WORKFLOW)[0].strip()+" "+WORKFLOW
                new_context = WORKFLOW.join(context.split(WORKFLOW)[:-1]).strip() +WORKFLOW
                new_response = context.split(WORKFLOW)[-1].strip() 
                #response = string.split(RESPONSE)[-1].strip()#.strip(RESPONSE_END).strip()
                #response = response.replace(RESPONSE_END,"")
                #context = sample["input"]+"\nsystem: "
                #utterance = sample["target"][len("system: "):].strip()#.strip("system:").strip() 
                subflow = pair["subflow"]
                dic = {
                    "context": new_context,
                    "response": new_response,
                    "cont_response": response,
                    "subflow": subflow
                }
                new.append(dic)
                if PRINT:
                    print(dic) #
            context_response_pairs = pd.DataFrame(new)

        elif dataset == "wf-cascade2":
            pairs = []
            import csv
            with open(cascade_datapath, 'r') as data:
                for line in csv.DictReader(data):
                    context = line["context"]+ line["response_1"].strip() + WORKFLOW_END +RESPONSE
                    response = line["cont_response"]
                    true_wf = line["true_response"]
                    subflow = line["subflow"]
                    dic = { "context": context, "response": response, "true_wf": true_wf, "subflow": subflow }
                    if PRINT:
                        print(dic) #
                    pairs += [dic]

            context_response_pairs = pd.DataFrame(pairs)

        else:
            raise NotImplementedError(f"{dataset=}")
    
    if dataset != "wf-cascade1" and dataset != "wf-cascade2":
        context_response_pairs = pd.concat(context_response_pairs_list)
        print("Original data size:" ,len(context_response_pairs))
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
            if False:
            #if idx < 5:
                for c,r,p in zip(contexts, responses, preds):
                    print("===")
                    print(c)
                    print("GT:", r)
                    print("Model:", p)

        all_pred_responses = list(chunk(all_pred_responses, size=num_responses))
        #preds_flat = all_pred_responses #[p for pred_responses in all_pred_responses for p in pred_responses]
        #gold_flat = [gold_response for gold_response mux nin all_gold_responses for i in range(num_responses)]
 
    preds_flat = [p for pred_responses in all_pred_responses for p in pred_responses]
    gold_flat = [gold_response for gold_response in all_gold_responses for i in range(num_responses)]

    if True:
        p,g = [], []
        count = 0
        for pp,gg in zip(preds_flat, gold_flat):
            for stoken in SPECIAL_TOKEN_SET:
                pp = pp.replace(stoken, "").strip()
                gg = gg.replace(stoken, "").strip()
            # pp = pp.replace(WORKFLOW_END,"").strip()
            # pp = pp.replace(RESPONSE_END,"").strip()
            # pp = pp.replace(USER_END,"").strip()

            # gg = gg.replace(WORKFLOW_END,"").strip()
            # gg = gg.replace(RESPONSE_END,"").strip()
            # gg = gg.replace(USER_END,"").strip()

            
            p.append(pp)
            g.append(gg)
            # if count < 5:
            # #if True:
            #     print("===="*30)
            #     print("pred:", pp)
            #     print("gold:", gg)
            count += 1
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
            bleurt = evaluate.load("bleurt", "BLEURT-20") # evaluate.load("bleurt", module_type="metric")#evaluate.load("bleurt", "BLEURT-20")
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
        if False:
            if dataset_type == "wf":
                if not row["context"].strip().endswith(ACTION_END+WORKFLOW):
                    """Checking for hard examples where the model really has to figure out on its own"""
                    #print(row["context"])
                    continue
            else:
                # for oracle, we want to catch where
                # ACTION_END + WORKFLOW + <workflow> + WORKFLOW_END
                if not row["context"].strip().split(WORKFLOW)[-2].endswith(ACTION_END):
                    continue
            
        #print(row)
        row_dict = {}
        row_dict['context'] = row['context']
        row_dict['true_response'] = row['response']
        row_dict["subflow"] = row["subflow"]
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