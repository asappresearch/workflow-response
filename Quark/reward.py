import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from typing import Optional, List, Iterable, Dict, Any

from Quark.policy import Policy
from Quark.utils.utils import batchify, load_jsonl
#from Quark.utils.perspective_api import PerspectiveWorker, make_generations_col

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from model.constants import *
import json, csv

from Quark.utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness


from bert_score import BERTScorer
#from call_openai import batch_evaluate_reward
# scorer = BERTScorer(lang="en", rescale_with_baseline=True)
# P, R, F1 = scorer.score(preds_flat, gold_flat)
# bert_f1_flat = F1.tolist() # list of (n_examples * n_responses) elements

def chunk(l, size=16):
      
    # looping till length l
    for i in range(0, len(l), size): 
        yield l[i:i + size]

class Reward:
    def __init__(self, reward_model_path: str,  batch_size: int, save_dir: str, bert_filter: bool = False):
        #self.path = save_path
        #self.rate_limit = rate_limit
        self.batch_size = batch_size
        # init the model here

        eval_model_path = reward_model_path 
        self.device = torch.device("cuda")
        self.evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(self.device)
        # evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path, torch_dtype=torch.float16).to(device)
        self.eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
        
        self.save_dir = save_dir


        #self.oracle = oracle

        self.bert_filter = bert_filter

        if self.bert_filter:
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def get_reward(self, prompts: List[str], responses: List[str], workflows: List[str], gt_responses: List[str], epoch: str) -> List[float]:
        # perspective_file = Path(self.path) / f'perspective_{epoch}.json'
        # perspective = PerspectiveWorker(
        #     out_file=perspective_file,
        #     total=len(prompts),
        #     rate_limit=self.rate_limit
        # )
        assert len(prompts) == len(responses), f'prompts({len(prompts)}) and responses({len(responses)}) mismatch'
        assert len(prompts) == len(workflows), f'prompts({len(prompts)}) and workflows({len(workflows)}) mismatch'
        assert len(prompts) == len(gt_responses), f'prompts({len(prompts)}) and gt_responses({len(gt_responses)}) mismatch'

        inputs = []
        for p, r, wf in zip(prompts, responses, workflows):
            # if not self.oracle:
            #     # TODO
            #     cat = p + r
            #     p = RESPONSE.join(cat.split(RESPONSE)[:-1]).strip() +RESPONSE
            #     r = cat.split(RESPONSE)[-1].strip()#.strip(RESPONSE_END).strip()

            context = p.replace(USER, "\nClient: ")
            context = context.replace(RESPONSE, "\nAgent: ")
            context = context.replace(WORKFLOW, "\nNext Action: ")
            context = context.replace(ACTION, "\nAction: ")

            for stoken in SPECIAL_TOKEN_SET:
                context = context.replace(stoken, "")
            
            context = context.strip()
            # 
            context = "\nNext Action: ".join(context.split("\nNext Action: ")[:-1]).strip()
            model_input = context + "\nAgent: " + r + "\nWorkflow Action: " + wf
            inputs.append(model_input)

        rewards = []
        for batch in chunk(inputs, size=self.batch_size):
            tokenized = self.eval_tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)

            output = self.evaluator(**tokenized)

            scores = output.logits.sigmoid().flatten()
            rewards += scores.tolist()
        # for i, r in enumerate(responses):
        #     perspective(f'generation-{i}', r)

        # #perspective.stop()
        # assert os.path.exists(perspective_file), 'missing perspective file'
        # data = pd.DataFrame.from_dict({'prompt': prompts})
        # results = collate(data, responses, load_jsonl(perspective_file), os.path.join(self.path, f'reward_{epoch}.json'))
        # rewards = [toxicity_to_reward(y['toxicity']) for x in results for y in x]

        if self.bert_filter and "eval" not in epoch:
            #for r, gt_response, response in zip(rewards, gt_responses, responses):
            P, R, F1 = self.bert_scorer.score(responses, gt_responses)
            bert_f1_flat = F1.tolist() # list of (n_examples * n_responses) elements
            new_rewards = [ float(x>=0.6)*r for x,r in zip(bert_f1_flat, rewards)]
            rewards = new_rewards

        assert len(prompts) == len(rewards), f'prompts({len(prompts)}) and rewards({len(rewards)}) mismatch'

        return rewards


class BlockReward(Reward):
    def __init__(self, reward_model_path: str,  batch_size: int, save_dir: str, \
    bert_filter: bool = False, action_end_enforce = True, repetition_penalty = False, context_reward=True):
        #self.path = save_path
        #self.rate_limit = rate_limit
        self.batch_size = batch_size
        # init the model here

        eval_model_path = reward_model_path 
        self.device = torch.device("cuda")
        self.evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path).to(self.device)
        # evaluator = AutoModelForSequenceClassification.from_pretrained(eval_model_path, torch_dtype=torch.float16).to(device)
        self.eval_tok = AutoTokenizer.from_pretrained(eval_model_path)
        
        self.save_dir = save_dir

        self.action_end_enforce = action_end_enforce
        self.repetition_penalty = repetition_penalty
        self.context_reward = context_reward

    def get_reward(self, prompts: List[str], responses: List[str], workflows: List[str], gt_responses: List[str], epoch: str, eval_mode=False) -> List[float]:
        """
        TODO: consider checking for termination with ACTION_END
              should i increase gen length then? let's do without for now
        """
        
        pos_responses = []
        action_ended = []
        repetition_penalty = []
        contexts = []
        for p,r,w in zip(prompts, responses, workflows):
            # print("-"*30)
            # print(p)
            # print("+"*30)
            # print(r)
            # print("="*30)
            # print(w)
            # print()
            """
            This part is for ensuring that workflow generation for cascade model is trained too
            """
            r = WORKFLOW_END.join(r.split(WORKFLOW_END)[1:]) # need to remove added workflow in the beginning
            #############

            if ACTION in r and r.split(ACTION)[-1].endswith(ACTION_END):
                action_ended.append(1.0)
            else:
                if w == "None" or w == None:
                    action_ended.append(1.0)
                else:
                    action_ended.append(0.0)

            target = r.replace(USER, "\nClient: ")
            target = target.replace(RESPONSE, "\nAgent: ")
            target = target.replace(WORKFLOW, "\nNext Action: ")
            target = target.replace(ACTION, "\nAction: ")

            for stoken in SPECIAL_TOKEN_SET:
                target = target.replace(stoken, "")
            
            target = target.strip()

            if self.context_reward:
                p = p + w
                #if p.endswith(WORKFLOW):
                #    p = p[:-len(WORKFLOW)]#p.strip(WORKFLOW)
                context = p.replace(USER, "\nClient: ")
                context = context.replace(RESPONSE, "\nAgent: ")
                context = context.replace(WORKFLOW, "\nNext Action: ")
                context = context.replace(ACTION, "\nAction: ")

                for stoken in SPECIAL_TOKEN_SET:
                    context = context.replace(stoken, "")
                
                context = context.strip()
                contexts.append(context)
            # 
            #context = "\nNext Action: ".join(context.split("\nNext Action: ")[:-1]).strip()

            if self.context_reward:
                r = target
            else:
                r = "Agent: "+target
            new = []
            rsplit = r.split("\n")
            temp = []
            for i, rs in enumerate(rsplit):
                #if i == len(rsplit) -1:
                if rs.startswith("Action:"):
                    pass
                elif rs.startswith("Next Action:"):
                    pass
                else:
                    temp.append(rs)
            r = "\n".join(temp)
            pos_responses.append(r)


            if self.repetition_penalty:
                generations = []
                for s in r.split("\n"):
                    if s.startswith("Agent:"):
                        generations.append(s)
                dist_1, dist_2, dist_3 = distinctness(generations)
                repetition_penalty.append(dist_3)

        if self.context_reward:
            inputs = [f"{context}\n{pr}\nWorkflow Action: {pos_act}" for context, pr, pos_act in zip(contexts, pos_responses, workflows)]
            # print("="*30)
            # print(inputs)
        else:
            inputs = [f"{pr}\nWorkflow Action: {pos_act}" for pr, pos_act in zip(pos_responses, workflows)]
        
        rewards = []
        for batch in chunk(inputs, size=self.batch_size):
            tokenized = self.eval_tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)

            output = self.evaluator(**tokenized)

            scores = output.logits.sigmoid().flatten()
            reward = scores.tolist()
            
            rewards += reward

        if not eval_mode:
            if self.action_end_enforce:
                rewards = [ r*a for r,a in zip(rewards, action_ended)]
            if self.repetition_penalty:
                rewards = [ r*p for r,p in zip(rewards, repetition_penalty)]

        return rewards



class LLMReward(Reward):
    def get_reward(self, prompts: List[str], responses: List[str], epoch: str) -> List[float]:
        return batch_evaluate_reward(prompts, responses) #[np.random.normal() for x in prompts]


class DummyReward(Reward):
    def get_reward(self, prompts: List[str], responses: List[str], epoch: str) -> List[float]:
        return [np.random.normal() for x in prompts]



def collate(dataset: Optional[pd.DataFrame],
            generations: List[str],
            responses: Iterable[Dict[str, Any]],
            output_file: str = ''):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    if output_file:
        dataset.to_json(output_file, orient='records', lines=True)
    return generations_col
