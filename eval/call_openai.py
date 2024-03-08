import os
import random
from tqdm import tqdm
import fire
import openai

from model.retrieve_openai import *

from model.dispatch_openai_requests import *
import time
import pprint

from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

import numpy as np

cost_dict = {
    'gpt-3.5-turbo-0301': (0.0015, 0.002),
    'gpt-3.5-turbo-0613': (0.0015, 0.002),
    'gpt-3.5-turbo-16k-0613': (0.0015, 0.002),
    'gpt-4-0314': (0.03, 0.06),
    'gpt-4-0613': (0.03, 0.06),
    #'text-davinci-003': (0.0)
}

def get_cost(output):
    if output is None:
        return 0.
    input_rate, output_rate = cost_dict[output['model']]
    return input_rate * int(output['usage']['prompt_tokens']) / 1000 + output_rate * int(output['usage']['completion_tokens']) / 1000



def call_api(prompt):
    openai.organization = "org-v6pbgo91xzgComEoVQwjoU0w"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    #print(openai.api_key)

    #models = openai.Model.list()
    #print("Available models:", models)

    # gpt_prompt = "I have two string lists in Python.\n \
    #  ['recover_username', 'recover_password', 'reset_2fa', 'status_service_added', 'status_service_removed', 'status_shipping_question', 'status_credit_missing', 'manage_change_address', 'manage_change_name', 'manage_change_phone', 'manage_payment_method', 'status_mystery_fee', 'status_delivery_time', 'status_payment_method', 'status_quantity', 'manage_upgrade', 'manage_downgrade', 'manage_create', 'manage_cancel', 'refund_initiate', 'refund_update', 'refund_status', 'return_stain', 'return_color', 'return_size', 'bad_price_competitor', 'bad_price_yesterday', 'out_of_stock_general', 'out_of_stock_one_item', 'promo_code_invalid', 'promo_code_out_of_date', 'mistimed_billing_already_returned', 'mistimed_billing_never_bought', 'status', 'manage', 'missing', 'cost', 'boots', 'shirt', 'jeans', 'jacket', 'pricing', 'membership', 'timing', 'policy', 'status_active', 'status_due_amount', 'status_due_date', 'manage_pay_bill', 'manage_extension', 'manage_dispute_bill', 'credit_card', 'shopping_cart', 'search_results', 'slow_speed'] \
    #  \n \
    # ['Initiate Refund', 'Update Refund', 'Refund Status', 'Return Due to Stain', 'Return Due to Color', 'Return Due to Size', 'Status Mystery Fee', 'Status Delivery Time', 'Status Payment Method', 'Status Quantity', 'Manage Upgrade', 'Manage Downgrade', 'Manage Create', 'Manage Cancel', 'Recover Username', 'Recover Password', 'Reset Two-Factor Auth', 'Invalid Credit Card', 'Cart Not Updating', 'Search Not Working', 'Website Too Slow', 'Status Service Added', 'Status Service Removed', 'Status Shipping Question', 'Status Credit Missing', 'Manage Change Address', 'Manage Change Name', 'Manage Change Phone', 'Manage Payment Method', 'Bad Price Competitor', 'Bad Price Yesterday', 'Out-of-Stock General', 'Out-of-Stock One Item', 'Promo Code Invalid', 'Promo Code Out of Date', 'Mistimed Billing Already Returned', 'Mistimed Billing Never Bought', 'Shipping Status', 'Manage Shipping', 'Missing Item', 'Shipping Cost', 'Status Active', 'Status Due Amount', 'Status Due Date', 'Manage Pay Bill', 'Manage Extension', 'Manage Dispute Bill', 'Boots FAQ', 'Shirt FAQ', 'Jeans FAQ', 'Jacket FAQ', 'Pricing FAQ', 'Membership FAQ', 'Timing FAQ', 'Policy FAQ']\n \
    # Those two lists have same length and have 1-1 correspondence. Now return me a dict in k:v form where the corresponding items are matched. Don't repeat the input lists in your output."
    
    
    gpt_prompt = prompt #input ("What prompt do you want to use?: ") 
    #print(gpt_prompt)

    model="gpt-3.5-turbo"
    #and
    message=[{"role": "user", "content": gpt_prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages = message,
        temperature=0.2,
        max_tokens=100,
        frequency_penalty=0.0
    )

    #completion = openai.Completion.create(model="text-davinci-003", prompt="Hello world")
    #print(response)
    return response


def chunkify(l, n=20):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
  

def generate(result_path = "./test_results/openai_baseline.json"):
    BSZ = 2
    results = []
    with open("./data/b2.json", "r") as fh:
        data = json.load(fh)
    #data = data["train"] + data["dev"]

    test = data["test"]
    #random.shuffle(data)

    #test = test[:4]
    messages = []
    targets = []
    for d in tqdm(test):
        cid = d["convo_id"]
        subflow = cid2subflow[cid]
        dialog = d["input"]
        target = d["target"]
        
        target = target.replace("system:","").strip()
        
        #print(d)
        prompt = construct_baseline_prompt(subflow, dialog, cid)
        messages.append([prompt, target])
        targets.append(target)

    
    for chunk in tqdm(chunkify(messages, n=BSZ), total=(len(messages) // BSZ) +1 ):

        try:
            result = asyncio.run(
                dispatch_openai_requests(
                    messages_list=[
                        [x ] for x in chunk
                    ],
                    #messages_list=chunk,
                    model="text-davinci-003",
                    temperature=0.3,
                    max_tokens=30,
                    top_p=1.0,
                )
            )
            #print(len(result))
            #exit()
            texts = [ ]
            results += result

            print("=" * 30)
            for c, r in zip(chunk, result):
                print("Prompt:", c[0])
                print("Target:", c[1])
                print("Generated:", r.choices[0].text)
                print(r)
                print()
            input()
        except:
            break


    print(f"{len(results)} predicted in total.")

    save = []
    for prompt, target, response in zip(messages, targets, results):
        dic = {"target": target, "response": response, "prompt": prompt}
        save.append(dic)
    
    print(save)
    with open(result_path, "w") as fh:
        json.dump(save, fh, indent=4)

        
def evaluate(test_path = "test_path/openai_baseline.json", result_path = "./test_results/openai_eval_result.json"):
    BSZ = 50
    results = []
    with open("./data/b2.json", "r") as fh:
        data = json.load(fh)
    #data = data["train"] + data["dev"]

    dataset = data["test"]
    random.shuffle(dataset)

    #test = test[:4]
    messages = []
    targets = []
    for d in tqdm(dataset):
        cid = d["convo_id"]
        subflow = cid2subflow[cid]


        neg_pool = []
        for k,v in subflow2systemutts.items():
            if k != subflow:
                neg_pool += [x[1] for x in v]

        neg_response = random.choice(neg_pool)
        dialog = d["input"]
        target = d["target"]

        
        target = target.replace("system:","").strip()
        
        #print(d)
        # construct_evaluate_prompt(subflow: str = None, dialog: str="", convo_id=None, response=""):
        pos_prompt = construct_evaluate_prompt(subflow, dialog, cid, response=target)
        neg_prompt = construct_evaluate_prompt(subflow, dialog, cid, response=neg_response)

        #messages.append(prompt)
        #targets.append(target)

    
        result = asyncio.run(
            dispatch_openai_requests(
                # messages_list=[
                #     [{"role": "user", "content":x} ] for x in chunk
                # ],
                messages_list=[pos_prompt, neg_prompt],
                model="text-davinci-003",
                temperature=0.3,
                max_tokens=40,
                top_p=1.0,
            )
        )
        
        print("="*30)
        print("Positive prompt:", pos_prompt)
        print("negative response:", neg_response)

        judgements = [x.choices[0].text for x in result]
        print(judgements)
        input()


def neg_gen(LEN=200, datasplit="train"):
    #BSZ = 50
    #pp = pprint.PrettyPrinter(indent=4)

    LEN = 200
    datasplit = "dev"
    dataset = get_neg_gen_data(min_len=0, datasplit=datasplit, data_balance=True)

    random.shuffle(dataset)

    dataset = dataset[:LEN]

    colleced_data = []

    #test = test[:4]
    messages = []
    targets = []
    for d in tqdm(dataset):
        cid = d["convo_id"]
        sid = d["sample_id"]
        subflow = cid2subflow[cid]

        neg_flow_pool = [ x for x in cid2subflow.values() if x != subflow ]

        neg_flow = random.choice(neg_flow_pool)
        with open("./data/kb.json", "r") as fh:
            kb = json.load(fh)


        
        pos_actions = kb[subflow]
        neg_next_action = pos_actions[0]
        while neg_next_action in pos_actions:
            neg_next_action = random.choice(kb[neg_flow])
        


        neg_pool = []
        for k,v in subflow2systemutts.items():
            if k != subflow:
                neg_pool += [x[1] for x in v]

        neg_response = random.choice(neg_pool)
        dialog = d["input"]
        target = d["target"]
        pos_next_action = d["workflow"]

        
        target = target.replace("system:","").strip()

        neg_gen_prompt = construct_neg_gen_prompt(neg_flow, dialog, cid, neg_next_action)
        try:    
            result = asyncio.run(
                dispatch_openai_requests(
                    # messages_list=[
                    #     [{"role": "user", "content":x} ] for x in chunk
                    # ],
                    messages_list=[neg_gen_prompt],
                    model="text-davinci-003",
                    temperature=0.3,
                    max_tokens=40,
                    top_p=1.0,
                )
            )
        except Exception as e:
            print("Error e: ",e)
            break

        gens = [x.choices[0].text.strip() for x in result]

        dic = {
            "prompt": neg_gen_prompt,
            "conv_id": cid,
            "sample_id": sid,
            "dialog": dialog,
            "neg_response": gens[0],
            "pos_response": target,
            "kb_actions": pos_actions,
            "original_flow": subflow,
            "negative_flow": neg_flow,
            "original_action": pos_next_action,
            "negative_action": neg_next_action,
        }
        colleced_data.append(dic)
    
    print("Created data len:", len(colleced_data))
    with open(f"./data/generated_negatives_{datasplit}.json", "w") as fh:
        json.dump(colleced_data, fh, indent=4)

rate_limit = AsyncLimiter(50, 60)
@retry(wait=wait_fixed(2), stop=stop_after_attempt(6))
async def task(msg, MAX_GEN_LEN=40):
    
    async with rate_limit:
        prompt = msg["prompt"]
        result = await openai.ChatCompletion.acreate(
            #model="gpt-3.5-turbo", 
            model="gpt-3.5-turbo-16k-0613",
            max_tokens=MAX_GEN_LEN,
            #temperature=0.3,
            #top_p=1.0,
            messages=[
                {"role": "user", "content": prompt}
            ])
        # result = await openai.Completion.acreate(
        #     model="text-davinci-003", 
        #     prompt= prompt,
        #     max_tokens=40,
        #     #temperature=0.3,
        #     #top_p=1.0,
        # )
        # print("result:", result)
        # exit()
        #gens = [x.choices[0].text.strip() for x in result]
        gen = result.choices[0].message.content.strip() #text.strip()
        msg.update({"generated":gen})
        #return await asyncio.gather(*msg) #msg
        return msg, result

async def batch_neg_gen(BSZ=50, LEN=200, datasplit="train"):
    dataset = get_neg_gen_data(min_len=0, datasplit=datasplit, data_balance=True, LEN=LEN)
    print("Dataset size:", len(dataset))
    with open(f"./data/wf2utts_{datasplit}.json", "r") as fh:
        wf2utts = json.load(fh)

    random.shuffle(dataset)

    dataset = dataset#[:]

    colleced_data = []
    
    messages = []
    
    for d in tqdm(dataset):
        cid = d["convo_id"]
        sid = d["sample_id"]
        pos_next_action = str(d["workflow"])
        subflow = cid2subflow[cid]

        with open("./data/kb.json", "r") as fh:
            kb = json.load(fh)

        pos_actions = kb[subflow]

        neg_flow_pool = [ x for x in cid2subflow.values() if x != subflow ]


        neg_flow = random.choice(neg_flow_pool)
        neg_actions = kb[neg_flow]

        while set(neg_actions).intersection(set(pos_actions)) == set(neg_actions) or set(neg_actions).intersection(set(pos_actions)) == set(pos_actions):
            neg_flow = random.choice(neg_flow_pool)
            neg_actions = kb[neg_flow]
  
        
        neg_next_action = random.choice(kb[neg_flow]) #+[str(None)]) #spos_actions[0]
        # while neg_next_action in pos_actions:
        #     neg_next_action = random.choice(kb[neg_flow]+[None])
        while neg_next_action == pos_next_action:
            neg_next_action = random.choice(kb[neg_flow]) #+[str(None)])

        neg_pool = []
        # for k,v in subflow2systemutts.items():
        #     if k != subflow:
        #         neg_pool += [x[1] for x in v]
        for k,v in wf2utts.items():
            if k != pos_next_action:
                neg_pool += [k]
        random_action = random.choice(neg_pool)
        random_response = random.choice(wf2utts[random_action])
        dialog = d["input"]
        target = d["target"]
        

        
        target = target.replace("system:","").strip()

        #neg_gen_prompt = construct_neg_gen_prompt(neg_flow, dialog, cid, neg_next_action)
        neg_gen_prompt = new_construct_neg_gen_prompt(neg_flow, dialog, cid, neg_next_action)

        dic = {
            "prompt": neg_gen_prompt,
            "conv_id": cid,
            "sample_id": sid,
            "dialog": dialog,
            #"neg_response": gens[0],
            "pos_response": target,
            "kb_actions": pos_actions,
            "original_flow": subflow,
            "negative_flow": neg_flow,
            "original_action": pos_next_action,
            "negative_action": neg_next_action,
            "random_response": random_response,
            "random_action": random_action
        }
        messages.append(dic)
        

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    print("Created data len:", len(collected_data))
    collected_data = [{"total cost": total_cost}] + collected_data

    save_path = f"./data/generated_negatives_{datasplit}_{LEN}_with_none_with_random.json"
    print("saved to", save_path)
    with open(save_path, "w") as fh:
        json.dump(collected_data, fh, indent=4)
    output_path = f"./data/output_generated_negatives_{datasplit}_{LEN}_with_none_with_random.json"
    with open(output_path, "w") as fh:
        json.dump(outputs, fh, indent=4)


async def batch_evaluate(BSZ=50, LEN=200, datasplit="train", eval_target="pos"):
    #test_path = "test_path/openai_baseline.json", result_path = "./test_results/openai_eval_result.json"

    #dataset = get_neg_gen_data(min_len=0, datasplit=datasplit)
    dataset = get_block_eval_data()
    # with open(f"./data/wf2utts_{datasplit}.json", "r") as fh:
    #     wf2utts = json.load(fh)

    random.shuffle(dataset)

    dataset = [ x for x in dataset if x["workflow"] !="None"]
    dataset = dataset[:LEN]

    colleced_data = []
    
    messages = []

    if eval_target == "rand":
        with open(f"./data/wf2utts_{datasplit}.json", "r") as fh:
            wf2utts = json.load(fh)

    for d in tqdm(dataset):
        cid = d["convo_id"]
        sid = d["sample_id"]
        pos_next_action = str(d["workflow"])
        subflow = cid2subflow[cid]

        with open("./data/kb.json", "r") as fh:
            kb = json.load(fh)

        pos_actions = kb[subflow]

        dialog = d["input"]
        target = d["target"]
        
        target = target.replace("system:","").strip()

        if eval_target == "rand":
            # neg_flow_pool = [ x for x in cid2subflow.values() if x != subflow ]


            # neg_flow = random.choice(neg_flow_pool)
            # neg_actions = kb[neg_flow]

            # while set(neg_actions).intersection(set(pos_actions)) == set(neg_actions) or set(neg_actions).intersection(set(pos_actions)) == set(pos_actions):
            #     neg_flow = random.choice(neg_flow_pool)
            #     neg_actions = kb[neg_flow]
    
            # neg_next_action = random.choice(kb[neg_flow]+[str(None)]) #spos_actions[0]
            # # while neg_next_action in pos_actions:
            # #     neg_next_action = random.choice(kb[neg_flow]+[None])
            # while neg_next_action == pos_next_action:
            #     neg_next_action = random.choice(kb[neg_flow]+[str(None)])

            # neg_pool = []
            # # for k,v in subflow2systemutts.items():
            # #     if k != subflow:
            # #         neg_pool += [x[1] for x in v]
            # for k,v in wf2utts.items():
            #     if k != pos_next_action:
            #         neg_pool += [k]
            # random_action = random.choice(neg_pool)
            # random_response = random.choice(wf2utts[random_action])
            while True: 
                cand = random.choice(dataset) 
                # print(cand["workflow"], pos_next_action)
                # print(subflow, cid2subflow[cand["convo_id"]])
                if cand["workflow"] == pos_next_action or subflow == cid2subflow[cand["convo_id"]]:
                    pass
                else:
                    random_response = cand["target"]
                    random_action = str(cand["workflow"])
                    random_response = random_response.replace("system:","").strip()
                    break
            evaluate_prompt = new_construct_evaluate_prompt(random_response, subflow, dialog, cid, pos_next_action)
        else: 
            evaluate_prompt = new_construct_evaluate_prompt(target, subflow, dialog, cid, pos_next_action)
            random_action = None
        dic = {
            "prompt": evaluate_prompt,
            "conv_id": cid,
            "sample_id": sid,
            "dialog": dialog,
            #"neg_response": gens[0],
            "pos_response": target,
            "kb_actions": pos_actions,
            "original_flow": subflow,
            # "negative_flow": neg_flow,
            "original_action": pos_next_action,
            # "negative_action": neg_next_action,
            # "random_response": random_response,
            "random_action": random_action
        }
        messages.append(dic)
        # print(dic)
        # exit()
        
    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    """
    TODO: compute results
    """
    if eval_target == "pos":
        pool = ["1"] #  $, "0"]
        #pool = ["1"]
    else:
        pool = ["0" ]#, "0"]
    #got_right = [ x["generated"].split("\n")[0].strip() in pool  for x in collected_data]
    got_right = [ x["generated"].split("Score:")[1].strip() in pool if "Score:" in x["generated"] else -1 for x in collected_data ]
    #print(got_right)
    print("Accuracy: ", np.average(got_right))

    print("Evaluated data len:", len(collected_data))
    collected_data = [{"eval_mode": eval_target, "accuracy":np.average(got_right), "total cost": total_cost}] + collected_data
    
    save_path = f"./data/evaluated_{eval_target}_{datasplit}_{LEN}.json"
    print("saved to", save_path)
    with open(save_path, "w") as fh:
        json.dump(collected_data, fh, indent=4)
    output_path = f"./data/output_evaluated_{eval_target}_{datasplit}_{LEN}.json"
    with open(output_path, "w") as fh:
        json.dump(outputs, fh, indent=4)

async def batch_evaluate_reward(prompts, responses, workflows, subflows):

    messages = []
    for p, r, w, s in tqdm(zip(prompts, responses, workflows, subflows)):
        evaluate_prompt = new_construct_evaluate_prompt(r, s, p, None, w)
    
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    print([x["generated"] for x in collected_data])
    predictions = [ x["generated"].split("Score:")[1].strip()  if "Score:" in x["generated"] else None for x in collected_data ]
    print(predictions)
    # 
    new = []
    for p in predictions:
        try:
            n = float(p)
            new.append(n)
        except:
            new.append(0.0)
    predictions = new
    return predictions

async def batch_evaluate_fluency_coherence(prompts, responses, workflows, subflows):

    messages = []
    for p, r, w, s in tqdm(zip(prompts, responses, workflows, subflows)):
        evaluate_prompt = llm_fluency_evaluate_prompt(r, s, p, None, w)
    
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    #print([x["generated"] for x in collected_data])
    predictions = [ x["generated"].strip() for x in collected_data ]

    # 
    new = []
    for p in predictions:
        try:
            n = float(p)
            new.append(n)
        except:
            new.append(0.0)
    predictions = new
    return predictions


async def defunct_batch_evaluate_compliance(prompts, responses, workflows, subflows):

    messages = []
    for p, r, w, s in tqdm(zip(prompts, responses, workflows, subflows)):
        evaluate_prompt = compliance_evaluate_prompt(r, s, p, None, w)
    
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    #print([x["generated"] for x in collected_data])
    predictions = [ x["generated"].strip() for x in collected_data ]

    # 
    new = []
    for p in predictions:
        try:
            n = float(p)
            new.append(n)
        except:
            new.append(0.0)
    predictions = new
    return predictions


async def batch_evaluate_compliance(guidelines, responses, workflows, subflows, inputs=None):
    if inputs is None:
        inputs = [ None for s in responses]

    if guidelines is None:
        guidelines = [ retrieve_guideline_text_action(s, w) for w,s in zip(workflows, subflows)]

    messages = []

    for g, r, w, s,i in tqdm(zip(guidelines, responses, workflows, subflows, inputs)):
        #evaluate_prompt = compliance_evaluate_prompt(r, s, p, None, w)

        if i is not None:
            evaluate_prompt = f"Read the provide guideline and assess the extent to which the agent's behavior \
in the input interaction aligns with the specified workflow action, \
considering the name and a concise description of the workflow provided. \
1 = Compliant\n0 = Non-compliant\n\nSubflow: {s}\nWorkflow: {w}\nDescription: {g}\n\n\
Dialogue History:\n{i}\n\nInput Interaction:\n{r}\n\nAnswer:"
        
        else:
            evaluate_prompt = f"Read the provide guideline and assess the extent to which the agent's behavior \
in the input interaction aligns with the specified workflow action, \
considering the name and a concise description of the workflow provided. \
1 = Compliant\n0 = Non-compliant\n\nSubflow: {s}\nWorkflow: {w}\nDescription: {g}\n\nInput Interaction:\n{r}\n\nAnswer:"

#         if i is not None:
#             evaluate_prompt = f"Assess the degree to which the agent's behavior aligns with the specified workflow action, \
# taking into account the action's name and policy guideline. \
# If the agent has already completed certain steps or the entire policy guideline behavior in the dialogue history, \
# they should not be penalized for not repeating those corresponding steps.\n\n\
# 1 = Compliant: The agent successfully executes all the steps outlined in the policy guideline.\n\
# 0 = Partially Compliant: The agent only partially achieves the steps described in the behavior or makes errors while doing so.\n\
# -1 = Non-compliant: The agent fails to execute any of the steps mentioned in the policy guideline.\n\n\
# Subflow: {s}\nWorkflow: {w}\Policy Guideline: {g}\n\n\
# Dialogue History:\n{i}\n\nInput Interaction:\n{r}\n\nAnswer:"
        
#         else:
#             evaluate_prompt = f"Assess the degree to which the agent's behavior aligns with the specified workflow action, \
# taking into account the action's name and policy guideline.\n\n\
# 1 = Compliant: The agent successfully executes all the steps outlined in the policy guideline.\n\
# 0 = Partially Compliant: The agent only partially achieves the steps described in the behavior or makes errors while doing so.\n\
# -1 = Non-compliant: The agent fails to execute any of the steps mentioned in the policy guideline.\n\n\
# Subflow: {s}\nWorkflow: {w}\Policy Guideline: {g}\n\n\
# Input Interaction:\n{r}\n\nAnswer:"
        # print(evaluate_prompt)
        # exit()
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    #print([x["generated"] for x in collected_data])
    predictions = [ x["generated"].strip() for x in collected_data ]

    # 
    new = []
    for p in predictions:
        try:
            n = float(p)
            new.append(n)
        except:
            new.append(0.0)
    predictions = new
    return predictions    

async def batch_evaluate_fluency(guidelines, responses, workflows, subflows, inputs=None):
    if inputs is None:
        inputs = [ None for s in responses]

    if guidelines is None:
        guidelines = [ retrieve_guideline_text_action(s, w) for w,s in zip(workflows, subflows)]

    messages = []

    for g, r, w, s,i in tqdm(zip(guidelines, responses, workflows, subflows, inputs)):
        #evaluate_prompt = compliance_evaluate_prompt(r, s, p, None, w)

        if inputs is None:
            evaluate_prompt = f"Please rate the fluency of the agent's linguistic behavior on a binary scale (1 = very fluent, 0 = not fluent). \
In this evaluation, please do not consider repetitive agents as fluent. Additionally, do not penalize the agent for disfluent client behavior in the evaluation.\n\n\
Input Interaction:\n{r}\n\nAnswer:"
        else:
            evaluate_prompt = f"Please rate the fluency of the agent's linguistic behavior on a binary scale (1 = very fluent, 0 = not fluent). \
In this evaluation, please do not consider repetitive agents as fluent. Additionally, do not penalize the agent for disfluent client behavior in the evaluation.\n\n\
Coherent Example:\n\
Agent: thanks for your information.\n\
Agent: the system said that your shipping address is the same as the one you stated above. the email was incorrect. you can ignore it.\n\
Client: thank you that's all i needed to know\n\
Agent: great, is there anything else that i can help you with?\n\
Client: no, that is all.\n\
Agent: have a nice day!\n\n\
Incoherent Example:\n\
Agent: how much was the service?\n\
Client: it was $40.\n\
Agent: how much was the extra price?\n\
Client: i was charged $40\n\
Agent: how much was the price you were charged?\n\n\
Input Interaction:\n{r}\n\nAnswer:"
            

        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    #print([x["generated"] for x in collected_data])
    predictions = [ x["generated"].strip() for x in collected_data ]

    # 
    new = []
    for p in predictions:
        try:
            n = float(p)
            new.append(n)
        except:
            new.append(0.0)
    predictions = new
    return predictions   

async def batch_llm_generate(input_contexts, guidelines, workflows, subflows, examples):

    if guidelines is None:
        guidelines = [ retrieve_guideline_text_action(s, w) for w,s in zip(workflows, subflows)]

    messages = []

    #examples = None

    for g, w, s, i in tqdm(zip(guidelines, workflows, subflows, input_contexts)):
        #evaluate_prompt = compliance_evaluate_prompt(r, s, p, None, w)
        #chosen_examples = random.sample(e, 2)
        filtered_examples = [ x[-1] for x in examples if x[0] == s and x[1] == w]
        try:
            chosen_examples = random.sample(filtered_examples, min(2, len(filtered_examples)))
        except Exception as e:
            print(e)
            # print(len(filtered_examples))
            # print(s,w)
            # filtered_examples = ["", ""]
            exit()
        example_str= ""
        for ex in chosen_examples:
            example_str += "Example:\n"+ex+"\n\n"
        generate_prompt = f"You are a cusotmer agent helping a customer with a issue. Read the dialogue context, provided \
policy guideline, and generate an agent utterance to help the customer in a way that is compliant to the guideline. \
The generated agent turn should be at most 2 utterances, and should be similar in length to the agent utterances shown in the examples \
that demonsrtate compliant agent behavior.\n\
Custome Situation: {s}\n\
Policy Action Name: {w}\n\
Policy Name Guideline: {g}\n\n\
{example_str}\
Dialog Context: {i}\n\n\
Agent: "  
#         generate_prompt = f"You are a cusotmer agent helping a customer with a issue. Read the dialogue context, provided \
# policy guideline, and generate an agent utterance to help the customer in a way that is compliant to the guideline. \
# The generated agent turn should be at most 2 utterances, and should be similar in length to the agent utterances shown in the examples \
# that demonsrtate compliant agent behavior.\n\
# Custome Situation: {s}\n\
# Policy Action Name: {w}\n\
# Policy Name Guideline: {g}\n\n\
# Example 1:\n{chosen_examples[0]}\n\n\
# Example 2:\n{chosen_examples[1]}\n\n\
# Dialog Context: {i}\n\n\
# Agent: "       
        #print("="*30)
        #print(generate_prompt)

        dic = {
            "prompt": generate_prompt,
        }
        messages.append(dic)

    tasks = []

    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))

    results = await asyncio.gather(*tasks)
    outputs =  [ x[1] for x in results]
    
    total_cost = sum([get_cost(output) for output in outputs])
    print(f'Total cost: {total_cost:.4f}')

    collected_data = [ x[0] for x in results] 
    #print([x["generated"] for x in collected_data])
    predictions = [ x["generated"].strip() for x in collected_data ]

    #print(predictions)
    return predictions   

def controller(do="neg_gen", **kwargs):

    if do == "neg_gen":
        asyncio.run(batch_neg_gen(**kwargs))
    elif do == "evaluate":
        asyncio.run(batch_evaluate(**kwargs))        
    elif do == "generate":
        asyncio.run(batch_generate(**kwargs))  

if __name__ == "__main__":
    #fire.Fire(generate)
    #fire.Fire(evaluate)    
    #fire.Fire(neg_gen)
    fire.Fire(controller)
    #fire.Fire(batch_neg_gen)

    # for datasplit in ["train", "dev"]:
        
    #     with open(f"./data/wf2utts_{datasplit}.json", "r") as fh:
    #         wf2utts = json.load(fh)



    #     with open(f"./data/generated_negatives_{datasplit}_with_none.json", "r") as fh:
    #         data = json.load(fh)
    #         new_data = []
    #         for d in data:
    #             pos_act = d["original_action"]

    #             neg_pool = []
    #             for k,v in wf2utts.items():
    #                 if k != pos_act:
    #                     neg_pool += [k]
    #             random_action = random.choice(neg_pool)
    #             random_response = random.choice(wf2utts[random_action])
    #             #random_action = random.choice(neg_pool)
    #             #print("neg_response:", neg_response)
    #             #input()
    #             d.update({"random_response": random_response, "random_action": random_action })
    #             new_data.append(d)

    #     with open(f"./data/generated_negatives_{datasplit}_with_none_with_random.json", "w") as fh:
    #         json.dump(new_data, fh, indent=4)