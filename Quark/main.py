import os
import torch
import json
import time
import logging
import random
import argparse
import numpy as np
import itertools
from typing import List
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from Quark.arguments import get_args
from Quark.policy import Policy
from Quark.data_pool import DataPool
from Quark.reward import Reward, LLMReward, BlockReward
from Quark.utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

from functools import partial
from model.constants import *
import csv

from model.retrieve_openai import *


PRINT = False

def clean_check(subflow, workflow):

    guideline =  retrieve_guideline_text_action(subflow, workflow) 
    
    # print("="*30)
    # print(subflow, workflow)
    # print(guideline)
    # print() 
    return guideline != -1

def convo_to_context_response_pairs_workflow_response(dataset_type: str, oracle: bool, reward_mode: str, \
    limit_none: bool, no_none : bool, only_standard_data: bool, datum, ):
    """{'sample_id': 2, 'convo_id': 3695, 'turns': [{'speaker': 'user', 'text': 'hey ho!', 'turn_count': 1, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'good afternoon, how can i help you?', 'turn_count': 2, 'targets': ['timing', 'retrieve_utterance', None, [], 84], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "i've got a promo code and i want to know when they expire.", 'turn_count': 3, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "i'd like to use it to buy some hats for my cat.", 'turn_count': 4, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'sure!  let me check that.', 'turn_count': 5, 'targets': ['timing', 'retrieve_utterance', None, [], 16], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'one moment please', 'turn_count': 6, 'targets': ['timing', 'retrieve_utterance', None, [], 26], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "some people think it's funny to put hats on cats...i do not feel that way.", 'turn_count': 7, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'cats deserve to look good too', 'turn_count': 8, 'targets': ['timing', 'retrieve_utterance', None, [], 54], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': 'exactly!', 'turn_count': 9, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'ok, just to verify you already tried to use the code?', 'turn_count': 10, 'targets': ['timing', 'retrieve_utterance', None, [], 77], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': 'no, i just want to see how long they last for.', 'turn_count': 11, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'ok, sorry for the doubt and i will answer your question.', 'turn_count': 12, 'targets': ['timing', 'retrieve_utterance', None, [], 43], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'one moment please', 'turn_count': 13, 'targets': ['timing', 'retrieve_utterance', None, [], 65], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'action', 'text': 'searching the faq pages ...', 'turn_count': 14, 'targets': ['timing', 'take_action', 'search-faq', [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'action', 'text': 'system action: search timing', 'turn_count': 15, 'targets': ['timing', 'take_action', 'search-timing', [], -1], 'workflow_action': ['timing', 'take_action', 'search-timing', [], -1]}, {'speaker': 'action', 'text': 'faq answer related to timing (question4) was selected.', 'turn_count': 16, 'targets': ['timing', 'take_action', 'select-faq', ['timing_4'], -1], 'workflow_action': ['timing', 'take_action', 'select-faq', ['timing_4'], -1]}, {'speaker': 'system', 'text': 'ok, all promo codes expire after 7 days without fail.', 'turn_count': 17, 'targets': ['timing', 'retrieve_utterance', None, [], 9], 'workflow_action': None}, {'speaker': 'user', 'text': 'perfect. thanks', 'turn_count': 18, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'not problem! a pleasure to help you and your cat too', 'turn_count': 19, 'targets': ['timing', 'retrieve_utterance', None, [], 5], 'workflow_action': None}, {'speaker': 'user', 'text': "that's all, have a great day! don't forget to spay or neuter your pet!", 'turn_count': 20, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'have a nice day', 'turn_count': 21, 'targets': ['timing', 'retrieve_utterance', None, [], 58], 'workflow_action': None}, {'speaker': 'system', 'text': "i won't", 'turn_count': 22, 'targets': ['timing', 'retrieve_utterance', None, [], 1], 'workflow_action': None}]}
    """
    context_response_pairs = []
    
    strings = []
    wfs = []
    future_wfs = []

    turns = datum["turns"]
    try:
        flow = turns[0]["targets"][0]
    except:
        flow = None
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
                #workflow = "N/A"
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
            elif "future" in dataset_type:
                workflow = turn["workflow_action"]
                #print(workflow)
                #if workflow != [None]:
                workflow = [x[2] if x is not None else x for x in workflow ]
                #print(workflow)
                string += WORKFLOW +", ".join([str(x) for x in workflow]) +    WORKFLOW_END +RESPONSE + text +  RESPONSE_END 
            else:
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
                string += WORKFLOW +  str(workflow) +   WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
            strings.append(string)
            wfs.append(str(workflow))
            future_wfs.append(workflow)
        else:
            print("impossib")
            exit()
    

    if dataset_type == "future":
        wfs = future_wfs
    end_string = strings[-1]
    split = end_string.split(RESPONSE)
    assert len(split)-1 == len(wfs), f"{len(split)}-1 != {len(wfs)}"

    for i, s in enumerate(split[1:]):
        first = RESPONSE.join(end_string.split(RESPONSE)[:i+1]).strip() +RESPONSE
        second = RESPONSE.join(end_string.split(RESPONSE)[i+1:]).strip() 
        
        second =  RESPONSE_END.join(second.split(RESPONSE_END)[:-1]) 
        for stoken in [ACTION_END, CONTEXT]:
            second = second.split(stoken)[0] 

        if not reward_mode == "block":
            second = second.split(RESPONSE_END)[0]

        if not oracle and dataset_type != "b2":
            new_context = WORKFLOW.join(first.split(WORKFLOW)[:-1]).strip() +WORKFLOW
            new_response = second #context.split(WORKFLOW)[-1].strip() 

            first = new_context
            second = new_response

        dic = {"context": first, "gt_response": second, "subflow":flow, "workflow":wfs[i]}


        if reward_mode == "block" and False: # and False added for multiwoz
            if (dataset_type == "b2" or dataset_type == "kb") and not first.strip().endswith(ACTION_END+RESPONSE):
                #continue
                dic = "NB"
                context_response_pairs.append(dic)
                continue
            if (WORKFLOW not in first or not first.strip().split(WORKFLOW)[-2].endswith(ACTION_END)) and (dataset_type != "b2" and dataset_type != "kb") :
                #continue
                dic = "NB"
                context_response_pairs.append(dic)
                continue


        if only_standard_data:
            """
            kb.json check
            """
            if dataset_type == "future":
                clean = clean_check(flow, str(future_wfs[i][0]))
                # clean = future_wfs[i][0] in kb[flow]
            else:
                clean = clean_check(flow, wfs[i])
                # clean = wfs[i] in kb[flow]
            if not clean:
                #print(flow, wfs[i]) # remove
                #continue
                dic = "AnnoError"
                context_response_pairs.append(dic)
                continue

        # greeting (the block model is not trained on this one)
        # commented for multiwoz
        # if wfs[i] == "None" or wfs[i] == None:
        #     #continue
        #     dic = None
        
        if wfs[i] == "end-dialog" or wfs[i] == "None" or wfs[i] == None and False: # and False added for multiwoz
            if no_none == True:
                #continue
                dic = None
            if limit_none == True:
                if random.uniform(0.0,1.0) >= 0.2:
                    #continue
                    dic = None
        
        # for multiwoz test
        # if len(first.split()) > 50 or len(second.split()) > 50:
        #     dic = None

        context_response_pairs.append(dic)

    
    return context_response_pairs

class PromptDataset(Dataset):
    def __init__(self, dataset_type, path, split, data_len=None, oracle=True, reward_mode="single", limit_none=True, no_none=False, only_standard_data=False):
        #self.prompts = [json.loads(s.strip())["prompt"]["text"].strip() for s in open(path, 'r').readlines()]
        self.dataset_type = dataset_type
        self.reward_mode = reward_mode
        self.limit_none = limit_none
        self.no_none = no_none
        self.only_standard_data = only_standard_data
        with open(path, 'r') as f:
            all_data = json.load(f)
            data = all_data[split]
            
            func = partial(convo_to_context_response_pairs_workflow_response, self.dataset_type, \
                oracle, self.reward_mode, self.limit_none, self.no_none, self.only_standard_data)

            context_response_pairs_list = list(tqdm(map(func, data), total=len(data)))
            context_response_pairs_list = [ y for temp in context_response_pairs_list for y in temp ]
            print("Original data size:", )
            blocks = [ x for x in  context_response_pairs_list if x != "NB"]
            print("Blocks size:", len(blocks))
            anno_error = [ x for x in  context_response_pairs_list if x == "AnnoError"]
            print("Anno Error size:", len(anno_error))
            context_response_pairs_list = [ x for x in  context_response_pairs_list if x is not None and x != "NB" and x != "AnnoError"]
            print("Filtered data size:", len(context_response_pairs_list))

            if data_len == None:
                data_len = float("inf")
            L = min(data_len, len(context_response_pairs_list)) #len(context_response_pairs_list)
            self.prompts = [ x["context"] for x in context_response_pairs_list][:L] 
            self.workflows =  [ x["workflow"] for x in context_response_pairs_list][:L]
            self.gt_responses = [ x["gt_response"] for x in context_response_pairs_list][:L]
            self.subflows = [ x["subflow"] for x in context_response_pairs_list][:L]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx], "workflow": self.workflows[idx], "gt_response": self.gt_responses[idx], "subflow": self.subflows[idx]}


class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.max_length = 512
        self.return_subflow = False

    def __call__(self, sequences):
        #self.tokenizer = set_tokenizer(self.tokenizer, padding_side="left")

        prompts = [sequence['prompt'] for sequence in sequences]
        workflows = [sequence['workflow'] for sequence in sequences]
        gt_responses = [sequence["gt_response"] for sequence in sequences]
        subflows = [ sequence["subflow"] for sequence in sequences ]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        if self.return_subflow:
            return input_ids, attention_mask, workflows, gt_responses, subflows
        else:    
            return input_ids, attention_mask, workflows, gt_responses


class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool):
        self.queries, self.responses, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx]
                }


class SequenceCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.max_length = 512

    def __call__(self, sequences):
        #self.tokenizer = set_tokenizer(self.tokenizer, padding_side="right")

        queries = [sequence['query'] for sequence in sequences]
        #responses = [sequence['response']+RESPONSE_END for sequence in sequences] # trying to curb endless continuing
        responses = [sequence['response'] for sequence in sequences] # trying to curb endless continuing
        cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]

        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        query_input_ids = torch.cat([query_input_ids.new(cat_ids)[:, None], query_input_ids], dim=1)
        query_mask = torch.cat([query_mask.new([1] * len(query_mask))[:, None], query_mask], dim=1)

        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True, truncation=True, max_length=512)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask


class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult




class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: Policy,
                 eval_policy: Policy,
                 data_pool: DataPool,
                 score_model: Reward,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 oracle: bool,
                 reward_mode: bool,
                 interactive: bool):

        self.params = params
        self.policy = policy
        self.response_length = self.params.response_length
        self.ref_policy = ref_policy


        if eval_policy == None:
            self.eval_policy = self.ref_policy
        else:
            self.eval_policy = eval_policy

        self.data_pool = data_pool
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.writer = SummaryWriter(self.params.tensorboard_dir)

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.best_cat = self.tree_tokens[0]
        self.best_cat_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=self.policy.tokenizer)

        self.oracle = oracle

        self.special_token_set = SPECIAL_TOKEN_SET
        self.reward_mode = reward_mode
        if not self.oracle:
            self.special_token_set = [ x for x in SPECIAL_TOKEN_SET if x != WORKFLOW and x != WORKFLOW_END and x != RESPONSE]
        if self.reward_mode == "block":
            #self.special_token_set = [ACTION, CONTEXT]
            #self.special_token_set = [ACTION_END, CONTEXT]
            self.special_token_set = [CONTEXT_END]

        self.interactive = interactive
        self.MAX_INTERACTION = 3

        self.params.eval_interval = len(self.train_dataloader)
        self.params.sample_interval = len(self.train_dataloader)

        with open(self.params.save_dir+"/eval.csv", "a") as fh:
            #fh.write("step,prompt,response,workflow,reward\n")
            csvwriter = csv.writer(fh) 
            csvwriter.writerow(["step","prompt","response","workflow","reward", "perplexity", "gt_response"])

        with open(os.path.join(self.params.save_dir, 'args.json'), 'w') as f:
            json.dump(params.__dict__, f, indent=4)


    def add_control_code(self, input_ids, attention_mask):
        """
        Question: does this contradict / interfere with left padding?
                  I think it shouldn't, as long as it's not truncated
        """
        input_ids = torch.cat([input_ids.new([self.best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        return input_ids, attention_mask    

    def decode(self, query_input_ids, response_input_ids=None):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                 for p in query_input_ids]

        query = [ q.replace(self.policy.tokenizer.eos_token, "") for q in query]

        if response_input_ids is None:
            return query

        response = [self.policy.tokenizer.decode(r, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                    for r in response_input_ids]
        response = [ q.replace(self.policy.tokenizer.eos_token, "") for q in response]

        # Note: post processing because with batch stopping criteria trickier just gen 
        response = [ pred.split("\n")[0] for pred in response]
        #print(preds)

        # Mpte: when use_special_tokens, need to cut by RESPONSE_END or WORKFLOW_END
        for stoken in self.special_token_set:
            response = [ pred.split(stoken)[0] for pred in response]

        return query, response

    def sample(self, step, depleted=False):
        #print(step, self.params.sample_interval)
        if (step) % self.params.sample_interval != 0 and not depleted:
            #print(step, self.params.sample_interval)
            return
        log.info(f"[step {step}] Sampling ...")

        prompts, responses = [], []
        workflows = []
        gt_responses = []
        for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader),
                                       desc='Sampling from current policy')):
            input_ids, attention_mask, workflow, gt_response = batch
            #print("batch:", batch)
            if step == 0:
                rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
                prompt, response = rollouts['query/text'], rollouts['response/text']
                if self.interactive:
                    response = [ x.split(ACTION_END)[0] + ACTION_END if ACTION_END in x else x for x in response]
                """
                This part is to ensure workflow generation is trained too
                """
                np, nr = [], []
                for q, r, wf in zip(prompt, response, workflow):
                    # wf = q.split(WORKFLOW)[-1].strip(WORKFLOW_END).strip()
                    # q = WORKFLOW.join(q.split(WORKFLOW)[:-1]) + WORKFLOW
                    # r = wf + WORKFLOW_END + r
                    wf = q.split(WORKFLOW)[-1].split(WORKFLOW_END)[0]#.strip()
                    q = WORKFLOW.join(q.split(WORKFLOW)[:-1]) + WORKFLOW
                    r = wf + WORKFLOW_END + RESPONSE + r
                    np.append(q)
                    nr.append(r)
                prompt = np
                response = nr 
                ############
            else:
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
                if self.interactive:
                    rollouts = self.interactive_sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
                    response = rollouts['response/text']
                    prompt = self.decode(rollouts['query/input_ids'])

                else:
                    rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
                    response = rollouts['response/text']
                    prompt = self.decode(rollouts['query/input_ids'][:, 1:])

            prompts.extend(prompt)
            responses.extend(response)
            workflows.extend(workflow)
            gt_responses.extend(gt_response)

        scores = self.score_model.get_reward(prompts, responses, workflows, gt_responses, f'step{step}')
        self.data_pool.add(prompts=prompts, responses=responses, scores=scores)
        if False:
            for p, r, w, s, gt in zip(prompts, responses, workflows, scores, gt_responses):
                print("="*30)
                print("prompt:", p)
                print("response:", r)
                print("gt_response:", gt)
                print("workflow:", w)
                print("score:", s)
                #print()

        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, drop_last=True, collate_fn=self.seq_collator)
        self.sampler = iter(self.sample_dataloader)




    def interactive_sample(self, input_ids, attention_mask, top_p, max_len): #step, depleted=False):

        PRINT = False
        self.MAX_INTERACTION = 1 # added for multiwoz
        for i in range(self.MAX_INTERACTION):
            input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)

            #agent_query, agent_response = self.generate(self.policy, input_ids, attention_mask)
            agent_rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
            
            agent_response = agent_rollouts['response/text']

            agent_response = [ x.split(USER)[0] for x in agent_response ]
            
            temp = []
            for r in agent_response:
                if ACTION in r and ACTION_END in r.split(ACTION)[1]:
                    r = r.split(ACTION_END)[0] +ACTION_END
                else:
                    r = r.split(ACTION)[0]
                    r = RESPONSE_END.join(r.split(RESPONSE_END)[:-1]) + RESPONSE_END
                temp.append(r)
            agent_response = temp

            #agent_rollouts['response/text'] = agent_response

            agent_query = agent_rollouts['query/text']

            if i == 0:
                first_query = agent_query
            
            if i == self.MAX_INTERACTION -1:
                # no need to generate user response
                if False:
                    for a in agent_response:
                        print("="*30)
                        print("Iter:",i)
                        print(a)
                break

            user_query =  [ q + r  + USER  for q,r in zip(agent_query, agent_response) ]
            encodings_dict = self.ref_policy.tokenizer(user_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

            user_rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
            user_response = user_rollouts['response/text']
            #user_query = user_rollouts['query/text']
            #user_query, user_response = self.generate(self.ref_policy, input_ids, attention_mask)

            for stoken in [RESPONSE, WORKFLOW, ACTION, CONTEXT]:
                user_response = [ x.split(stoken)[0] for x in user_response ]
            #prompt = self.decode(rollouts['query/input_ids'][:, 1:])
            if PRINT:
                for a,u in zip(agent_response, user_response):
                    print("="*30)
                    print("Iter:",i)
                    print(a)
                    print(u)
                    print()

            agent_query = [ q + r + WORKFLOW for q,r in zip(user_query, user_response) ]

            encodings_dict = self.policy.tokenizer(agent_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

        # TODO: need to remake the roll out
        full_texts = [ q + r for q, r in zip(agent_query, agent_response)]

        query, response = [], []
        for i,f in enumerate(full_texts):
            assert(f.startswith(first_query[i])), f"something's wrong {f} // {first_query[i]}"
            q = first_query[i]
            r = f[len(first_query[i]):]
            if ACTION in r:
                if ACTION_END in r:
                    try:
                        r = r.split(ACTION)[0] + ACTION + r.split(ACTION)[1].split(ACTION_END)[-2] + ACTION_END
                    except:
                        print(r)
                        r = r.split(ACTION)[0] + ACTION 
                else:
                    r = r.split(ACTION)[0] + ACTION 
            """
            Note: workflow has to be reattached to the response to make the loss work
            """
            wf = q.split(WORKFLOW)[-1].split(WORKFLOW_END)[0]#.strip()
            q = WORKFLOW.join(q.split(WORKFLOW)[:-1]) + WORKFLOW
            r = wf + WORKFLOW_END + RESPONSE + r

            query.append(q)
            response.append(r)
            #############
            if PRINT:
                    print("="*30)
                    print("Query:", q)
                    print("Response:", r)
                    print()

        encodings_dict = self.policy.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        query_input_ids = encodings_dict['input_ids']
        query_attention_mask = encodings_dict['attention_mask']   

        encodings_dict = self.policy.tokenizer(response, return_tensors="pt", padding=True, truncation=True, max_length=512)
        response_input_ids = encodings_dict['input_ids']
        response_attention_mask = encodings_dict['attention_mask']   


        interactive_rollouts = {
            'query/input_ids': query_input_ids,
            'query/text': query,
            'query/mask': query_attention_mask,
            'response/input_ids': response_input_ids,
            'response/text': response,
            'response/mask': response_attention_mask.to(self.policy.device),
        }

        return interactive_rollouts



    def step(self, step_num):
        step_started_at = time.time()
        self.sample(step=step_num)

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration):
            """
            the dataloader has runout, so override and create the dataloader again
            """
            depleted = True
            self.sample(step=step_num, depleted=depleted)
            batch = next(self.sampler)
        except (AssertionError):
            """
            If StopIteration & only_top, we need to generate the 
            """
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)

        self.optimizer.zero_grad()
        ppo_loss, stats = self.loss(step_num, *batch)
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        for metric in ['kl', 'entropy']:
            self.writer.add_scalar(f'Objective/{metric}', stats[f'objective/{metric}'], step_num)
        for metric in ['lm', 'kl', 'entropy', 'total']:
            self.writer.add_scalar(f'Loss/{metric}', stats[f'loss/{metric}'], step_num)
        self.writer.add_scalar(f'Params/lr', self.optimizer.param_groups[0]['lr'], step_num)
        self.writer.add_scalar(f'Params/kl_coef', self.kl_ctl.value, step_num)
        self.writer.add_scalar(f'Params/entropy_coef', self.entropy_ctl.value, step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True)
        self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        log.info(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        self.save(step=step_num)
        self.eval(step=step_num)

    def loss(self, step, query_input_ids, query_mask, response_input_ids, response_mask):
        """
        TODO: cascade sucks possibly because of this step?
              i.e. i need to take the loss of (GT) workflow too
        """
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        logits = outputs['response/logits'][:, :, :-len(self.tree_tokens)] # taking case of embedding / added tree token
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            """
            This is right, because quantile is code is added in the sequence collator / datapool
            """
            ref_outputs = self.ref_policy.forward_pass(query_input_ids[:, 1:], query_mask[:, 1:],
                                                       response_input_ids, response_mask)                                                      
            ref_logprobs, ref_logits = ref_outputs['response/log_prob'], ref_outputs['response/logits']

        kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)

        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)
        
        #print(f"step {step}: {loss}")
        if torch.isnan(loss).any():
            print("-"*30)
            print(loss)
            exit()

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = self.decode(query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step)

        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step):
        if (step) % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save(self, step):
        if (step) % self.params.save_interval != 0 and (step) != self.params.total_steps -1:
            return
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        log.info(f"[step {step}] model checkpoint saved")

    def eval(self, step):
        """
        Eval needs to be rewritten to handle cascade & oracle
        # use query/text and response/text
        # rewrite this part
        """
        #if step % self.params.eval_interval != 0:
        if (step) % self.params.eval_interval != 0 and (step) != self.params.total_steps -1:
            return
        log.info(f"[step {step}] evaluating ...")

        """
        if oracle also ast performance
        """

        generations, perplexities, compliances = [], [], []
        workflows = []
        predicted_wfs = []
        for i, (input_ids, attention_mask, workflow, gt_response) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
                if self.interactive:
                    rollouts = self.interactive_sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
                    forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                    'query_mask': rollouts['query/mask'],
                                    'response_input_ids': rollouts['response/input_ids'],
                                    'response_mask': rollouts['response/mask']}
                else:
                    rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
                    forward_inputs = {'query_input_ids': rollouts['query/input_ids'][:, 1:],
                                    'query_mask': rollouts['query/mask'][:, 1:],
                                    'response_input_ids': rollouts['response/input_ids'],
                                    'response_mask': rollouts['response/mask']}
                ref_logprobs = self.eval_policy.forward_pass(**forward_inputs)['response/log_prob']
                
                queries = rollouts["query/text"]
                responses = rollouts["response/text"]
                # print("="*30)
                # print(queries)
                # print(responsess)
                
                if not self.oracle:
                    nq, nr = [], []
                    
                    for p,r in zip(queries, responses):
                        cat = p + r
                        p = RESPONSE.join(cat.split(RESPONSE)[:-1]).strip() +RESPONSE
                        p_wf = r.split(WORKFLOW_END)[0].strip() 
                        #for stoken in SPECIAL_TOKEN_SET:
                        #    p_wf = p_wf.replace(stoken,"").strip()
                        r = cat.split(RESPONSE)[-1].strip()#.strip(RESPONSE_END).strip() d
                        nq.append(p)
                        nr.append(r)
                         
                        predicted_wfs.append(p_wf)
                    #     print("="*30)
                    #     print(p)
                    #     print(r)
                    #     print(p_wf)
                    # exit()
                    queries = nq
                    responses = nr

                #print(ref_logprobs.shape, rollouts['response/mask'].shape)
                #print(ref_logprobs, rollouts['response/mask'])
                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                #print(perplexity)
                perplexity = perplexity.cpu().detach().numpy().tolist()
                perplexities.extend(perplexity)
                
                prompt = queries #self.decode(rollouts['query/input_ids'][:, 1:])
                response = responses #rollouts['response/text']
                score = self.score_model.get_reward(prompt, response, workflow, gt_response, f'step{step}_eval{i}', eval_mode=True)
                compliance = [x for x in score if x is not None]
                compliances.extend(compliance)

                generations.extend(response)
                workflows.extend(workflow)
                #generations.extend(rollouts['response/text'])

                with open(self.params.save_dir+"/eval.csv", "a") as fh:
                    csvwriter = csv.writer(fh) 
                    for p,r,w,s,pp, gt in zip(prompt, response, workflow, score, perplexity, gt_response):
                        #line = f"{epoch},{p},{r},{w},{s}\n"
                        #fh.write(line)
                        csvwriter.writerow([f'step{step}_eval{i}',p,r,w,s,pp, gt])

        #if final:
        ppl_score= np.mean(perplexities)
        print(f"  perplexity = {ppl_score:+.2f}")
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)
        compliance_score = np.mean(compliances)
        dist_1, dist_2, dist_3 = distinctness(generations)

        
        print(f"  compliance = {compliance_score:+.2f}")
        print(f'dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}')
        
        self.writer.add_scalar('Evaluation/compliance', compliance_score, step)
        self.writer.add_scalar('Evaluation/Dist-1', dist_1, step)
        self.writer.add_scalar('Evaluation/Dist-2', dist_2, step)
        self.writer.add_scalar('Evaluation/Dist-3', dist_3, step)

        if not self.oracle:
            # print(workflows)
            # print("-"*30)
            # print(predicted_wfs)
            # exit()
            wf_accuracy = np.average([x==y for x,y in zip(workflows, predicted_wfs)])    
            print(f"  workflow accuracy = {wf_accuracy:+.2f}")  
            self.writer.add_scalar('Evaluation/wf-accuracy', wf_accuracy, step)                  

if __name__ == "__main__":
    pass
    #main()

    # def generate(self, policy, input_ids, attention_mask):

    #     device = policy.device
    #     input_ids = input_ids.to(device)
    #     attention_mask = attention_mask.to(device)
    #     generated = policy.model.generate(input_ids = input_ids, attention_mask = attention_mask, \
    #     max_new_tokens=self.response_length, temperature=self.params.temperature,\
    #     top_p=self.params.top_p,  do_sample=True, eos_token_id=policy.end_of_response_id, pad_token_id = policy.tokenizer.eos_token_id)
        
    #     generated = generated[:, input_ids.shape[-1]:]
    #     responses  = policy.tokenizer.batch_decode(generated, skip_special_tokens=False)
    #     responses = [output.replace(self.policy.tokenizer.eos_token,"") for output in responses]
        
    #     queries = policy.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    #     queries = [output.replace(self.policy.tokenizer.eos_token,"") for output in queries]
        
    #     if policy.tree_tokens is not None:
    #         for tt in policy.tree_tokens:
    #             queries = [output.replace(tt,"") for output in queries]

    #     #responses = [ x.split(ACTION_END)[0] for x in responses ] #
    #     if False:
    #         print("="*30)
    #         for q,r in zip(queries, responses):
    #             print(q)
    #             print("+"*30)
    #             print(r)
    #             print()
    #             input()
    #     return queries, responses


# def set_tokenizer_for_policy(policy):
#     # need to use hf generate functionality
#     policy.tokenizer.padding_side = "left"
#     policy.tokenizer.truncation_side = 'left'
#     # Define PAD Token = EOS Token = 50256
#     policy.tokenizer.pad_token = policy.tokenizer.eos_token
#     policy.model.config.pad_token_id = policy.model.config.eos_token_id
    
#     # self.ref_policy.tokenizer.padding_side = "left"
#     # self.ref_policy.tokenizer.truncation_side = 'left'
#     # # Define PAD Token = EOS Token = 50256
#     # self.ref_policy.tokenizer.pad_token = self.ref_policy.tokenizer.eos_token
#     # self.ref_policy.model.config.pad_token_id = self.ref_policy.model.config.eos_token_id
#     return policy

# def set_tokenizer(tokenizer, padding_side="left"):
#     # need to use hf generate functionality
#     tokenizer.padding_side = padding_side
#     tokenizer.truncation_side = padding_side #'left'
#     # Define PAD Token = EOS Token = 50256
#     #tokenizer.pad_token = policy.tokenizer.eos_token
#     #policy.model.config.pad_token_id = policy.model.config.eos_token_id
    
#     # self.ref_policy.tokenizer.padding_side = "left"
#     # self.ref_policy.tokenizer.truncation_side = 'left'
#     # # Define PAD Token = EOS Token = 50256
#     # self.ref_policy.tokenizer.pad_token = self.ref_policy.tokenizer.eos_token
#     # self.ref_policy.model.config.pad_token_id = self.ref_policy.model.config.eos_token_id
#     return tokenizer

# def main():
#     args = get_args()

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     num_gpus = torch.cuda.device_count()
#     log.info(f'Detect {num_gpus} GPUS')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     time = datetime.now()
#     date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
#     args.save_dir = os.path.join(args.output_dir, date_time)
#     args.reward_dir = os.path.join(args.save_dir, 'reward')
#     args.model_dir = os.path.join(args.save_dir, 'model')
#     args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
#     for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
#         ensure_dir(d)
#     log.info(f'Write to output directory: {args.save_dir}')

#     with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
#         json.dump(args.__dict__, f, indent=2)

#     tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] + \
#                   [' _TREE_TOKEN_ZERO_COMMENTS']

#     log.info(f'Initializing models ...')
#     ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device)
#     policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device,
#                     reward_cond=True, tree_tokens=tree_tokens)
#     reward = Reward(save_path=args.reward_dir, batch_size=args.batch_size)
#     data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens)
#     log.info(f'Initialization done!')

#     prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
#     train_dataset = PromptDataset(dataset_type=args.dataset_type, path=args.dataset, split="train")
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator)
#     log.info(f'Load train set with {len(train_dataset)} examples')

#     val_dataset = PromptDataset(dataset_type=args.dataset_type, path=args.dataset, split="dev")
#     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
#     log.info(f'Load val set with {len(val_dataset)} examples')

#     # set up optimizer and scheduler
#     optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-5)
#     args.total_steps = ceil_div(args.total_episodes, args.batch_size)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

#     trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, data_pool=data_pool,
#                                score_model=reward, tree_tokens=tree_tokens,
#                                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
#                                optimizer=optimizer, scheduler=scheduler)

#     for step_num in range(args.total_steps):
#         try:
#             trainer.step(step_num)
#         except RuntimeError:
#             torch.cuda.empty_cache()
#             continue




"""

    def old_interactive_sample(self, input_ids, attention_mask, top_p, max_len): #step, depleted=False):

        PRINT = True
        for i in range(self.MAX_INTERACTION):
            input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
            agent_rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
            
            agent_response = agent_rollouts['response/text']
            agent_response = [ x.split(USER)[0] for x in agent_response ]
            #agent_response = [ x.split(ACTION_END)[0] +ACTION_END if ACTION_END in x else RESPONSE_END.join(x.split(RESPONSE_END)[:-1]) + RESPONSE_END for x in agent_response ]
            temp = []
            for r in agent_response:
                # if ACTION in r:
                #     if ACTION_END in r:
                #         r = r.split(ACTION_END)[0] +ACTION_END if ACTION_END
                #     else:
                #         r = RESPONSE_END.join(r.split(RESPONSE_END)[:-1]) + RESPONSE_END
                if ACTION in r and ACTION_END in r.split(ACTION)[1]:
                    r = r.split(ACTION_END)[0] +ACTION_END
                else:
                    r = r.split(ACTION)[0]
                    r = RESPONSE_END.join(r.split(RESPONSE_END)[:-1]) + RESPONSE_END
                temp.append(r)
            agent_response = temp

            agent_rollouts['response/text'] = agent_response

            agent_query = agent_rollouts['query/text']

            if i == 0:
                first_query = agent_query

            
            #prompt = self.decode(rollouts['query/input_ids'][:, 1:])
            
            if i == self.MAX_INTERACTION -1:
                # no need to generate user response
                if PRINT:
                    for a in agent_response:
                        print("="*30)
                        print("Iter:",i)
                        print(a)
                break

            user_query =  [ q + r  + USER  for q,r in zip(agent_query, agent_response) ]
            encodings_dict = self.ref_policy.tokenizer(user_query, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

            user_rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
            user_response = user_rollouts['response/text']
            user_query = user_rollouts['query/text']

            for stoken in [RESPONSE, WORKFLOW, ACTION, CONTEXT]:
                user_response = [ x.split(stoken)[0] for x in user_response ]
            #prompt = self.decode(rollouts['query/input_ids'][:, 1:])
            if PRINT:
                for a,u in zip(agent_response, user_response):
                    print("="*30)
                    print("Iter:",i)
                    print(a)
                    print(u)
                    print()
            #print("Agent query:", agent_query)
            #print("Agent response:", agent_response)
            
            #print("User query:", user_query)
            #print("User response:", user_response)       

            agent_query = [ q + r + WORKFLOW for q,r in zip(user_query, user_response) ]

            encodings_dict = self.policy.tokenizer(agent_query, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

        # TODO: need to remake the roll out
        full_texts = [ q + r for q, r in zip(agent_rollouts['query/text'], agent_rollouts['response/text'])]

        # new_full = []
        # for f in full_texts:
        #     new_f = ACTION.join(f.split(ACTION)[:-1])+ ACTION + f.split(ACTION_END)[-1].split(ACTION)[-1] + ACTION_END
        #     new_full += [new_f]
        # full_texts = new_full

        query, response = [], []
        for i,f in enumerate(full_texts):
            assert(f.startswith(first_query[i])), f"something's wrong {f} // {first_query[i]}"
            q = first_query[i]
            r = f[len(first_query[i]):]
            if ACTION in r:
                #r = r.split(ACTION)[0] + ACTION + r.split(ACTION)[1].split(ACTION_END)[-1] + ACTION_END
                #r = ACTION.join(r.split(ACTION)[:0])+ ACTION + r.split(ACTION_END)[-1].split(ACTION)[-1] + ACTION_END # action end
                if ACTION_END in r:
                    try:
                        r = r.split(ACTION)[0] + ACTION + r.split(ACTION)[1].split(ACTION_END)[-2] + ACTION_END
                    except:
                        # must be the case where ACTION_END is before ACTION...?
                        print(r)
                        # offending example
                        # thank you.<|endofresponse|><|action|>verify-identity joseph banter, 
                        #i was able to verify your purchase.<|endofresponse|><|action|>validate-purchase jbanter1, jbanter1@email.com, 0494833246<|endofaction|>
                        #<|endofresponse|><|user|>ok thank you<|endofuser|><|endofuser|><|workflow|>None<|endofworkflow|>
                        #<|response|>your order has been verified.<|endofresponse|><|user|>thank you<|endofuser|><|endofuser|>
                        #<|workflow|>None<|endofworkflow|><|response|>you are welcome. is there anything else that i can help you with?
                        r = r.split(ACTION)[0] + ACTION 
                else:
                    r = r.split(ACTION)[0] + ACTION 
            query.append(q)
            response.append(r)
            if PRINT:
                    print("="*30)
                    print("Query:", q)
                    print("Response:", r)
                    print()

        encodings_dict = self.policy.tokenizer(query, return_tensors="pt", padding=True)
        query_input_ids = encodings_dict['input_ids']
        query_attention_mask = encodings_dict['attention_mask']   

        encodings_dict = self.policy.tokenizer(response, return_tensors="pt", padding=True)
        response_input_ids = encodings_dict['input_ids']
        response_attention_mask = encodings_dict['attention_mask']   


        interactive_rollouts = {
            'query/input_ids': query_input_ids,
            'query/text': query,
            'query/mask': query_attention_mask,
            'response/input_ids': response_input_ids,
            'response/text': response,
            'response/mask': response_attention_mask.to(self.policy.device),
            #'response/log_prob': agent_rollouts['response/log_prob'],
        }

        return interactive_rollouts


    def old_eval(self, step):

        #if step % self.params.eval_interval != 0:
        if (step) % self.params.eval_interval != 0 and (step) != self.params.total_steps -1:
            return
        log.info(f"[step {step}] evaluating ...")

        generations, perplexities, compliances = [], [], []
        for i, (input_ids, attention_mask, workflow, gt_response) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, max_len = self.response_length)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'][:, 1:],
                                  'query_mask': rollouts['query/mask'][:, 1:],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask']}
                #ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
                ref_logprobs = self.eval_policy.forward_pass(**forward_inputs)['response/log_prob']
                # queries = rollouts["query/text"]
                # responses = rollouts["response/text"]
                # texts = [ q+r for q,r in zip(queries, responses)]
                # print(texts)
                # for stoken in SPECIAL_TOKEN_SET:
                #     texts = [ x.replace(stoken, "") for x in texts]
                # forward_inputs = self.eval_policy.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
                # print(forward_inputs)
                # ref_logprobs = self.eval_policy.model(**forward_inputs, return_dict=True, output_attentions=False, output_hidden_states=False).logits
                # print(ref_logprobs)

                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                prompt = self.decode(rollouts['query/input_ids'][:, 1:])
                response = rollouts['response/text']
                score = self.score_model.get_reward(prompt, response, workflow, gt_response, f'step{step}_eval{i}')
                compliance = [x for x in score if x is not None]
                compliances.extend(compliance)

                generations.extend(rollouts['response/text'])

        ppl_score, compliance_score = np.mean(perplexities), np.mean(compliances)
        dist_1, dist_2, dist_3 = distinctness(generations)
        print(f"  perplexity = {ppl_score:+.2f}")
        print(f"  compliance = {compliance_score:+.2f}")
        print(f'dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}')
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)
        self.writer.add_scalar('Evaluation/compliance', compliance_score, step)
        self.writer.add_scalar('Evaluation/Dist-1', dist_1, step)
        self.writer.add_scalar('Evaluation/Dist-2', dist_2, step)
        self.writer.add_scalar('Evaluation/Dist-3', dist_3, step)



"""