import json
import fire

import random
from tqdm import tqdm

from prompt_formatter import *


kb2guideline = {'recover_username': 'Recover Username', 'recover_password': 'Recover Password', 'reset_2fa': 'Reset Two-Factor Auth',\
 'status_service_added': 'Status Service Added', 'status_service_removed': 'Status Service Removed', 'status_shipping_question': 'Status Shipping Question', \
 'status_credit_missing': 'Status Credit Missing', 'manage_change_address': 'Manage Change Address', 'manage_change_name': 'Manage Change Name', \
 'manage_change_phone': 'Manage Change Phone', 'manage_payment_method': 'Manage Payment Method', 'status_mystery_fee': 'Status Mystery Fee', \
 'status_delivery_time': 'Status Delivery Time', 'status_payment_method': 'Status Payment Method', 'status_quantity': 'Status Quantity',\
  'manage_upgrade': 'Manage Upgrade', 'manage_downgrade': 'Manage Downgrade', 'manage_create': 'Manage Create', 'manage_cancel': 'Manage Cancel', \
  'refund_initiate': 'Initiate Refund', 'refund_update': 'Update Refund', 'refund_status': 'Refund Status', 'return_stain': 'Return Due to Stain',\
   'return_color': 'Return Due to Color', 'return_size': 'Return Due to Size', 'bad_price_competitor': 'Bad Price Competitor', \
   'bad_price_yesterday': 'Bad Price Yesterday', 'out_of_stock_general': 'Out-of-Stock General', 'out_of_stock_one_item': 'Out-of-Stock One Item',\
    'promo_code_invalid': 'Promo Code Invalid', 'promo_code_out_of_date': 'Promo Code Out of Date', 'mistimed_billing_already_returned': 'Mistimed Billing Already Returned', \
    'mistimed_billing_never_bought': 'Mistimed Billing Never Bought', 'status': 'Shipping Status', 'manage': 'Manage Shipping', 'missing': 'Missing Item', \
    'cost': 'Shipping Cost', 'boots': 'Boots FAQ', 'shirt': 'Shirt FAQ', 'jeans': 'Jeans FAQ', 'jacket': 'Jacket FAQ', 'pricing': 'Pricing FAQ', 'membership': 'Membership FAQ',\
     'timing': 'Timing FAQ', 'policy': 'Policy FAQ', 'status_active': 'Status Active', 'status_due_amount': 'Status Due Amount', 'status_due_date': 'Status Due Date', \
     'manage_pay_bill': 'Manage Pay Bill', 'manage_extension': 'Manage Extension', 'manage_dispute_bill': 'Manage Dispute Bill', 'credit_card': 'Invalid Credit Card',\
      'shopping_cart': 'Cart Not Updating', 'search_results': 'Search Not Working', 'slow_speed': 'Website Too Slow'}
guideline2kb = {v:k for k,v in kb2guideline.items() }

wf_name_mapping = {'pull-up-account': 'Pull up Account', 'enter-details': 'Enter Details',\
 'verify-identity': 'Verify Identity', 'make-password': 'Make Password', 'search-timing': 'Timing',\
  'search-policy': 'Policy', 'validate-purchase': 'Validate Purchase', 'search-faq': 'Search FAQ',\
   'search-membership': 'Membership', 'search-boots': 'Boots', 'try-again': 'Try Again', \
   'ask-the-oracle': 'Ask the Oracle', 'update-order': 'Update Order', 'promo-code': 'Promo Code',\
    'update-account': 'Update Account', 'membership': 'Membership Privileges', 'make-purchase': 'Make Purchase', \
    'offer-refund': 'Offer Refund', 'notify-team': 'Notify Internal Team', 'record-reason': 'Record Reason', \
    'search-jeans': 'Jeans', 'shipping-status': 'Shipping Status', 'search-shirt': 'Shirt', 'instructions': 'Instructions', \
'search-jacket': 'Jacket', 'log-out-in': 'Log Out/In', 'select-faq': 'Select Answer', 'subscription-status': 'Subscription Status',\
 'send-link': 'Send Link', 'search-pricing': 'Pricing', "None": "None"}
reverse_wf_name_mapping = {v:k for k,v in wf_name_mapping.items()}

def read_kb(path="./data/kb.json"):
    with open(path, "r") as fh:
        workflows = json.load(fh)

    keys = workflows.keys()

    return workflows, keys

def read_guideline(path="./data/guidelines.json"):
    with open(path, "r") as fh:
        guideline =json.load(fh)

    subflows = [ x["subflows"] for x in guideline.values()]
    sf = {}
    for l in subflows:
        for k,v in l.items():
            sf[guideline2kb[k]] = v

    subflows = sf
    return sf #ssguideline


def read_examples(path="./data/abcd_v1.1.json"):
    with open(path, "r") as fh:
        data =json.load(fh)

    full = data["train"] + data["dev"] + data["test"]
    data = data["train"] + data["dev"]

    cid2subflow = {}    
    for convo in full:
        convo_id = convo["convo_id"]#
        subflow = convo["delexed"][0]["targets"][0] #["subflow"]
        cid2subflow[convo_id] = subflow 

    #print(data[0])

    subflow2data = {}
    subflow2systemutts = {}
    

    smap = { "agent": "system", "customer":"user", "action":"action" }
    for convo in data:
        convo_id = convo["convo_id"]#
        subflow = convo["delexed"][0]["targets"][0] #["subflow"]

        cid2subflow[convo_id] = subflow

        #if subflow[-2] == "_":
        #    subflow = subflow[:-2]

        #if subflow == "boots_how":
        ##    print(convo)
        #    exit()
        orig = convo["original"]
        
        delexed = convo["delexed"]

        convo_str = ""
        for turn in delexed:
            #print(turn)
            #input()
            
            speaker = smap[turn["speaker"]]
            targets = turn["targets"]
            if speaker == "action":
                convo_str += "action: " + targets[2] + " " + str(targets[3]) +"\n"
            else:
                convo_str += f"{speaker}: " + turn["text"] +"\n"  

            if speaker == "system":
                if subflow in subflow2systemutts:
                    subflow2systemutts[subflow].append([convo_id, turn["text"]])
                else:
                    subflow2systemutts[subflow] = [[convo_id, turn["text"]]]
        # convo_str = ""
        # for speaker, utt in orig:
        #     if speaker == "agent":
        #         convo_str += "system: " + utt +"\n"
        #     elif speaker =="customer":
        #         convo_str += "user: " + utt + "\n"
        #     elif speaker == "action":
        #         convo_str += "action:" + utt.replace("System Action:", "").strip() + "\n"
        #     else:
        #         print(speaker)
        #         print("error")
        #         exit(1)

        convo_str = convo_str.strip()
        if subflow in subflow2data:
            subflow2data[subflow].append([convo_id, convo_str])
        else:
            subflow2data[subflow] = [[convo_id, convo_str]]

    return subflow2data, cid2subflow, subflow2systemutts
subflow2data, cid2subflow, subflow2systemutts = read_examples()
# print(len(subflow2data))
# print(subflow2data.keys())
# exit()

# print(subflow2systemutts["jeans"])
# exit()



def get_wf_to_utts(splits=["test"]):
    with open("./data/wc_seed.json", "r") as fh:
        data = json.load(fh)

    dat = []
    for s in splits:
        dat += data[s]

    data = dat

    wf2utts = {}
    for d in data:
        turns = d["turns"]
        for turn in turns:
            speaker = turn["speaker"]
            if speaker == "system":

                wf = turn["workflow_action"]
                if wf == None:
                    wf = str(wf)
                else:
                    wf = wf[2]
                text = turn["text"]

                if wf not in wf2utts:
                    wf2utts[wf] = [ text]
                else:
                    wf2utts[wf] += [text]

    for k,v in wf2utts.items():
        wf2utts[k] = list(set(v))
        #print(d)
        #input()

    s = "-".join(splits)
    with open(f"./data/wf2utts_{s}.json", "w") as fh:
        json.dump(wf2utts, fh, indent=4)
    return wf2utts

# get_wf_to_utts(["train"])
# get_wf_to_utts(["test"])
# get_wf_to_utts(["dev"])
def retrieve_guideline_text(subflow: str = None):
    workflows, keys = read_kb()

    # format_in_md

    guideline = read_guideline()
    sf_guideline = guideline[subflow]

    s = format_in_md(sf_guideline)

    return s

guideline = read_guideline()
kb, keys = read_kb()
def alt_retrieve_guideline_text_action(subflow: str = None, action: str = None):

    if action == "end-dialog":
        return "Transition smoothly to conclude the assistance provided to the customer, expressing gratitude and bidding farewell."
    #workflows, keys = read_kb()

    #print(kb)
    #input()
    # format_in_md
    sf_guideline = guideline[subflow]
    action = wf_name_mapping[action]
    #print(subflow)
    #print(sf_guideline["actions"], action)
    #input()
    try:
        dic = [ x for x in sf_guideline["actions"] if x["button"] == action ][0]

        text = dic['text']
        subtext = '\n'.join(dic['subtext'])

        formatted_string = f"{text}\n\n{subtext}"
        return formatted_string
        #return action_guideline
        #print(action_guideline)
        #s = format_in_md(action_guideline)

        #return s
    except:
        return -1

#workflows, keys = read_kb()
def retrieve_guideline_text_action(subflow: str = None, action: str = None):

    if action == "end-dialog":
        return "Transition smoothly to conclude the assistance provided to the customer, expressing gratitude and bidding farewell."
    

    # format_in_md

    #print(subflow)
    #print(sf_guideline["actions"], action)
    #input()
    #try:
    if action in kb[subflow]:
        sf_guideline = guideline[subflow]
        action = wf_name_mapping[action]
        try:
            dic = [ x for x in sf_guideline["actions"] if x["button"] == action ][0]

            text = dic['text']
            subtext = '\n'.join(dic['subtext'])

            formatted_string = f"{text}\n\n{subtext}"
            return formatted_string
        except Exception as e:
            #print(e)
            return -1
        #return action_guideline
        #print(action_guideline)
        #s = format_in_md(action_guideline)

        #return s
    else:
        return -1

l = []
for action, v in wf_name_mapping.items():
    for subflow, vv in kb2guideline.items():
        t= retrieve_guideline_text_action(subflow, action) 
        if t != -1:
            l.append([subflow, action, t])

with open("guidelines.txt", "w") as fh:
    json.dump(l, fh, indent=4)


def retrieve_examples(subflow: str = None, num: int =2, pos:bool = True, convo_id=None):
    #workflows, keys = read_kb()

    if pos:
        pool = subflow2data[subflow]
    else:
        pool = []
        for k,v in subflow2data.items():
            if k != subflow:
                pool += list(v)

    if convo_id != None:
        pool = [ x[1] for x in pool if x[0] != convo_id ]
    

    chosen = random.sample(pool, num)


    return chosen


def old_construct_evaluate_prompt(subflow: str = None, dialog: str="", convo_id=None, response=""):
    guideline_str = retrieve_guideline_text(subflow)
    examples = retrieve_examples(subflow, convo_id = convo_id)

    example_str = ""
    for i, e in enumerate(examples):
        #print("Example: ", e)
        example_str += f"Example {i+1}\n{e}\n\n" #format_in_md(e)
        #print("ex_str:", example_str)
    prompt = f"You are evaluating a customer service agent who is helping a user with a service issue. \
Read the provided guideline and examples for the corresponding workflow and determine if the agent is responding in a manner that achieves the specified workflow step \
(next action) in helping the customer.\n\n\
Your evaluation should be 0 for negative, 1 for positive.\n\n\
Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\n{example_str}Dialog:\n{dialog}\n\nAgent Response: {response}\n\nEvaluation: "

    return prompt

def construct_baseline_prompt(subflow: str = None, dialog: str="", convo_id=None):
    
    guideline_str = retrieve_guideline_text(subflow)
    examples = retrieve_examples(subflow, convo_id = convo_id)

    example_str = ""
    for i, e in enumerate(examples):
        #print("Example: ", e)
        example_str += f"Example {i+1}\n{e}\n\n" #format_in_md(e)
        #print("ex_str:", example_str)
    prompt = f"You are a helping a user with a customer service issue. \
Read the provided guideline and examples for the corresponding workflow and predict the next utterance the system should say to the customer in the next step in the input dialog. Only generate the next system utterance, not actions. \n\n\
Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\n{example_str}Dialog:\n{dialog}\n\nsystem: "

    return prompt

from model.constants import *
with open("./data/kb.json", "r") as fh:
    kb = json.load(fh)   
def convo_to_context_response_pairs_workflow_response(dataset_type: str, datum, include_wf=False):
    """{'sample_id': 2, 'convo_id': 3695, 'turns': [{'speaker': 'user', 'text': 'hey ho!', 'turn_count': 1, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'good afternoon, how can i help you?', 'turn_count': 2, 'targets': ['timing', 'retrieve_utterance', None, [], 84], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "i've got a promo code and i want to know when they expire.", 'turn_count': 3, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "i'd like to use it to buy some hats for my cat.", 'turn_count': 4, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'sure!  let me check that.', 'turn_count': 5, 'targets': ['timing', 'retrieve_utterance', None, [], 16], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'one moment please', 'turn_count': 6, 'targets': ['timing', 'retrieve_utterance', None, [], 26], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': "some people think it's funny to put hats on cats...i do not feel that way.", 'turn_count': 7, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'cats deserve to look good too', 'turn_count': 8, 'targets': ['timing', 'retrieve_utterance', None, [], 54], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': 'exactly!', 'turn_count': 9, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'ok, just to verify you already tried to use the code?', 'turn_count': 10, 'targets': ['timing', 'retrieve_utterance', None, [], 77], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'user', 'text': 'no, i just want to see how long they last for.', 'turn_count': 11, 'targets': ['timing', None, None, [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'ok, sorry for the doubt and i will answer your question.', 'turn_count': 12, 'targets': ['timing', 'retrieve_utterance', None, [], 43], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'system', 'text': 'one moment please', 'turn_count': 13, 'targets': ['timing', 'retrieve_utterance', None, [], 65], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'action', 'text': 'searching the faq pages ...', 'turn_count': 14, 'targets': ['timing', 'take_action', 'search-faq', [], -1], 'workflow_action': ['timing', 'take_action', 'search-faq', [], -1]}, {'speaker': 'action', 'text': 'system action: search timing', 'turn_count': 15, 'targets': ['timing', 'take_action', 'search-timing', [], -1], 'workflow_action': ['timing', 'take_action', 'search-timing', [], -1]}, {'speaker': 'action', 'text': 'faq answer related to timing (question4) was selected.', 'turn_count': 16, 'targets': ['timing', 'take_action', 'select-faq', ['timing_4'], -1], 'workflow_action': ['timing', 'take_action', 'select-faq', ['timing_4'], -1]}, {'speaker': 'system', 'text': 'ok, all promo codes expire after 7 days without fail.', 'turn_count': 17, 'targets': ['timing', 'retrieve_utterance', None, [], 9], 'workflow_action': None}, {'speaker': 'user', 'text': 'perfect. thanks', 'turn_count': 18, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'not problem! a pleasure to help you and your cat too', 'turn_count': 19, 'targets': ['timing', 'retrieve_utterance', None, [], 5], 'workflow_action': None}, {'speaker': 'user', 'text': "that's all, have a great day! don't forget to spay or neuter your pet!", 'turn_count': 20, 'targets': ['timing', None, None, [], -1], 'workflow_action': None}, {'speaker': 'system', 'text': 'have a nice day', 'turn_count': 21, 'targets': ['timing', 'retrieve_utterance', None, [], 58], 'workflow_action': None}, {'speaker': 'system', 'text': "i won't", 'turn_count': 22, 'targets': ['timing', 'retrieve_utterance', None, [], 1], 'workflow_action': None}]}
    """
    context_response_pairs: List[Dict] = []
    
    strings = []
    wfs = []

    turns = datum["turns"]
    flow = turns[0]["targets"][0]
    kb_flow = kb[flow]  
    if dataset_type == "kb":       
        #kb_flow = kb[flow]  
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
            if dataset_type == "b1" or dataset_type == "b2" or dataset_type == "kb" or not include_wf:
                string += RESPONSE +  text + RESPONSE_END 
                #workflow = "Oracle"
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
            elif "future" in dataset_type:
                workflow = turn["workflow_action"]
                if workflow != [None]:
                    workflow = [x[2] if x is not None else x for x in workflow ]
                string += WORKFLOW +", ".join([str(x) for x in workflow]) +    WORKFLOW_END +RESPONSE + text +  RESPONSE_END 
                #wfs.append(workflow)
            else:
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
                string += WORKFLOW +  str(workflow) +   WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
            wfs.append(str(workflow))
            strings.append(string)
        else:
            print("impossib")
            exit()

    end_string = strings[-1]

    split = end_string.split(RESPONSE)
    assert len(split)-1 == len(wfs), f"{len(split)}-1 != {len(wfs)}"

    for i, s in enumerate(split[1:]):
        first = RESPONSE.join(end_string.split(RESPONSE)[:i+1]).strip() +RESPONSE
        second = RESPONSE.join(end_string.split(RESPONSE)[i+1:]).strip() #+RESPONSE #string.split(RESPONSE)[i:].strip()
        #print("="*30)
        #print("first:", first)
        #print("second:", second)
        
        second =  RESPONSE_END.join(second.split(RESPONSE_END)[:-1]) 
        for stoken in [ACTION_END, CONTEXT]:
            second = second.split(stoken)[0] 
        # also get only up until the last response (no user or other stuff)
        #print("actionless second:", second)
        # print(second)
        second = second.strip().split(ACTION)[0]
        # print(second)
        # input()
        dic = {"input": first, "target": second, "subflow":flow, "workflow":wfs[i], "convo_id": datum["convo_id"], "sample_id":datum["sample_id"], "kb":kb_flow}
        if False:
            print(dic)
        
        if dataset_type == "b2" and not first.strip().endswith(ACTION_END+RESPONSE):
            continue
        if (WORKFLOW not in first or not first.strip().split(WORKFLOW)[-2].endswith(ACTION_END)) and include_wf:
            #print("saas")
            continue
        #print(dic)
        context_response_pairs.append(dic)
        # print("="*30)
        # print("first:", first)
        # print("second:", second)
        #input()
    #exit()   
    # for string in strings:
    #     context = RESPONSE.join(string.split(RESPONSE)[:-1]).strip() +RESPONSE
    #     response = string.split(RESPONSE)[-1].strip()#.strip(RESPONSE_END).strip()
    #     response = response.replace(RESPONSE_END,"")
    #     dic = {"context": context, "response": response, "subflow":flow}
    #     if PRINT:
    #         print(dic)
    #     context_response_pairs.append(dic)
    #     if context.strip().endswith(USER_END):
    #         print(context)
    #         exit()
    
    new = []
    for dic in context_response_pairs:
        context = dic["input"]
        response = dic["target"]
        if response.strip() == "":
            print("="*30)
            print("Context: ", context)
            print("Empty reference: ", response)
        else:
            new +=  [dic]
    context_response_pairs = new
    
    #df_context_response_pairs = pd.DataFrame(context_response_pairs)
    
    return context_response_pairs #df_context_response_pairs

def get_block_eval_data(get_positive_examples =  False, min_len = 5, datasplit="train", include_wf=False):
    with open("./data/wc_seed_one_convo.json", "r") as fh:
        data = json.load(fh)


    split = data[datasplit]
    #random.shuffle(dataset)

    split = [ convo_to_context_response_pairs_workflow_response("oracle", x, include_wf) for x in split ]
    split = [ x  for temp in split for x in temp]
    new = []
    print("Raw data len:", len(split))
    for d in split:
        inp = d["input"]
        tgt = d["target"]
        workflow = d["workflow"]
        kb = d["kb"]
        if workflow not in kb:
            continue
        #print(d)
        #input()
        data = [ ]
        for i in [inp, tgt]:
            i = i.replace(USER, "\nClient: ")
            i = i.replace(RESPONSE, "\nAgent: ")
            i = i.replace(WORKFLOW, "\nNext Action: ")
            i = i.replace(ACTION, "\nAction: ")

            for stoken in SPECIAL_TOKEN_SET:
                i = i.replace(stoken, "")
            data.append(i)
        d["input"] = data[0]
        d["target"] = data[1]
        #print(d)
        #input()
        new.append(d)
        #print(k, true_response)
    print("Filtered data len:", len(new))
    return new

    res = []
    for d in tqdm(split):
        turns = d["turns"]
        subflow = turns[0]["targets"][0]
        string = "" #CONTEXT
        
        for i, turn in enumerate(turns):
            speaker = turn["speaker"]
            text = turn["text"]

            if speaker == "user":
                string += "Client: " + text +"\n" #+ USER_END 
            elif speaker == "action":
                button = turn["targets"][2] 
                slot =  turn["targets"][3]
                string += "Action: "  +button +  " " + ", ".join(slot).strip() +"\n" #+ ACTION_END 
            elif speaker == "system":
                #string += "Agent: " #+  text #+ RESPONSE_END 
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
                else:
                    workflow = str(workflow)
                #string +
                #string += WORKFLOW +  str(workflow) +   WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                if not get_positive_examples:
                    dic = { "input": string, "workflow":workflow, "convo_id":d["convo_id"], "target":text, "subflow":subflow, "sample_id":d["sample_id"] }
                    #print(dic)
                    #input()
                    if i >= min_len:
                        res.append(dic)
                    #string += text + "\n"
                    if include_wf:
                        string += "Next Action: " +  str(workflow) + "\n" + "Agent: " + text +"\n" #  WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                    else:
                        string += "Agent: " + text +"\n" #  WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                else:
                    #string += "Next Action: " +  str(workflow) + "\n" + "Agent: " + text +"\n" #  WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                    dic = { "input": string, "workflow":workflow, "convo_id":d["convo_id"], "target":text, "subflow":subflow, "sample_id":d["sample_id"] }
                    #print(dic)
                    #input()
                    if i == len(turns) -1:
                        res.append(dic)
                        
            else:
                print("impossib")
                exit()
    return res

def get_neg_gen_data(get_positive_examples =  False, min_len = 5, datasplit="train", data_balance=False, LEN=10, none_percent=0.03):
    with open("./data/wc_seed_one_convo.json", "r") as fh:
        data = json.load(fh)



    split = data[datasplit]
    #random.shuffle(dataset)

    res = []
    for d in tqdm(split):
        turns = d["turns"]
        subflow = turns[0]["targets"][0]
        string = "" #CONTEXT
        
        for i, turn in enumerate(turns):
            speaker = turn["speaker"]
            text = turn["text"]

            if speaker == "user":
                string += "Client: " + text +"\n" #+ USER_END 
            elif speaker == "action":
                button = turn["targets"][2] 
                slot =  turn["targets"][3]
                string += "Action: "  +button +  " " + ", ".join(slot).strip() +"\n" #+ ACTION_END 
            elif speaker == "system":
                #string += "Agent: " #+  text #+ RESPONSE_END 
                workflow = turn["workflow_action"]
                if workflow != None:
                    workflow = workflow[2]
                else:
                    workflow = str(workflow)
                #string +
                #string += WORKFLOW +  str(workflow) +   WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                if not get_positive_examples:
                    dic = { "input": string, "workflow":workflow, "convo_id":d["convo_id"], "target":text, "subflow":subflow, "sample_id":d["sample_id"] }
                    #print(dic)
                    #input()
                    if i >= min_len:
                        res.append(dic)
                    #string += text + "\n"
                    string += "Next Action: " +  str(workflow) + "\n" + "Agent: " + text +"\n" #  WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                else:
                    string += "Next Action: " +  str(workflow) + "\n" + "Agent: " + text +"\n" #  WORKFLOW_END + RESPONSE +  text +  RESPONSE_END 
                    dic = { "input": string, "workflow":workflow, "convo_id":d["convo_id"], "target":text, "subflow":subflow, "sample_id":d["sample_id"] }
                    #print(dic)
                    #input()
                    if i == len(turns) -1:
                        res.append(dic)
                        
            else:
                print("impossib")
                exit()

    if data_balance:
        wf = [ v["workflow"] for v in res]
        sf = [ v["subflow"] for v in res]
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

        from collections import OrderedDict
        sf2idx = OrderedDict(sorted(sf2idx.items(), key=lambda i: -len(i[1])))
        wf2idx = OrderedDict(sorted(wf2idx.items(), key=lambda i: len(i[1])))

        # print(sf2idx)
        # print(wf2idx)
        # exit()
        chosen = []
        while len(chosen) < int(LEN*(1-none_percent)):
            for k,v in wf2idx.items():
                if k == "None":
                    continue
                pool = list(set(v) - set(chosen))
                if len(pool) == 0:
                    continue
                else:
                    chosen.append(random.choice(pool))
                if len(chosen) >= LEN:
                    break

        chosen += random.sample(wf2idx["None"],int(LEN*none_percent))

        # for k,v in sf2idx.items():
        #     chosen.append(random.choice(v))
        # for k,v in sf2idx.items():
        #     if len(chosen) >= 100:
        #         break
        #     chosen.append(random.choice(list(set(v) - set(chosen))))
        #print(chosen)
        # print(wf2idx["None"])
        # print(len(chosen))
        from collections import Counter
        # print(Counter(wf))
        # print(Counter(sf))
        # print(len(Counter(wf)))
        # print(len(Counter(sf)))

        res = [ x for i,x in enumerate(res) if i in chosen]
        print(Counter([x["workflow"] for x in res]))
        print(Counter([x["subflow"] for x in res]))
    else:
        #random.shuffle(res)
        #res = res[:LEN]
        res = res
    return res


pos_examples = get_neg_gen_data(True)
def construct_neg_gen_prompt(subflow: str = None, dialog: str="", convo_id=None, next_action=""):
    
    guideline_str = retrieve_guideline_text(subflow)
    #examples = retrieve_examples(subflow, convo_id = convo_id)

    with open(f"./data/wf2utts_train.json", "r") as fh:
        wf2utts = json.load(fh)    

    examples = [ x["input"] for x in pos_examples if x["convo_id"] !=convo_id and x["subflow"] == subflow]    
    examples = random.sample(examples, 2)
    # examples = wf2utts[next_action]
    # examples = random.sample(examples, 10)
    #print(next_action, examples)
    #input()

    example_str = ""
    for i, e in enumerate(examples):
        #print("Example: ", e)
        example_str += f"Example {i+1}\n{e}\n\n" #format_in_md(e)
        #print("ex_str:", example_str)
    prompt = f"You are a helping a user with a customer service issue. \
Read the provided guideline for agents and examples for the corresponding workflow and predict the next utterance the system should say to the customer in the next action. The generated system utterance must \
help the agent achieve the next action. The style and length of the generation should match the agent in the examples. The generate rsesponse \
should be 10 ~ 15 words in length, and should explicitly express the next action.\n\n\
Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\n{example_str}Dialog:\n{dialog}\n\nNext Action:{next_action}\n\nAgent: "

    return prompt

def construct_neg_gen_prompt(subflow: str = None, dialog: str="", convo_id=None, next_action=""):
    
    guideline_str = retrieve_guideline_text(subflow)
    #examples = retrieve_examples(subflow, convo_id = convo_id)

    with open(f"./data/wf2utts_train.json", "r") as fh:
        wf2utts = json.load(fh)    

    examples = [ x["input"] for x in pos_examples if x["convo_id"] !=convo_id and x["subflow"] == subflow]    
    examples = random.sample(examples, 2)
    # examples = wf2utts[next_action]
    # examples = random.sample(examples, 10)
    #print(next_action, examples)
    #input()

    example_str = ""
    for i, e in enumerate(examples):
        #print("Example: ", e)
        example_str += f"Example {i+1}\n{e}\n\n" #format_in_md(e)
        #print("ex_str:", example_str)
    prompt = f"You are a helping a user with a customer service issue. \
Read the provided guideline for agents and examples for the corresponding workflow and predict the next utterance the system should say to the customer in the next action. The generated system utterance must \
help the agent achieve the next action. The style and length of the generation should match the agent in the examples. The generate rsesponse \
should be 10 ~ 15 words in length, and should explicitly express the next action.\n\n\
Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\n{example_str}Dialog:\n{dialog}\n\nNext Action:{next_action}\n\nAgent: "

    return prompt

def construct_evaluate_prompt(target: str = None, subflow: str = None, dialog: str="", convo_id=None, next_action=""):
    
    guideline_str = retrieve_guideline_text(subflow)
    #examples = retrieve_examples(subflow, convo_id = convo_id)

    

    examples = [ x["input"] for x in pos_examples if x["convo_id"] !=convo_id and x["subflow"] == subflow and next_action in x["input"]]    

    #examples = random.sample(examples, 2)

    example_str = ""
    for i, e in enumerate(examples):
        #print("Example: ", e)
        example_str += f"Example {i+1}\n{e}\n\n" #format_in_md(e)
        #print("ex_str:", example_str)
    dialog = dialog[:-len("Agent: ")]
    prompt = f"The following is a situation where a customer agent is helping a user with a customer service issue. \
Read the provided guideline for the corresponding workflow and determine if the agent is responding in a manner that achieves the specified workflow step \
(Target Action) in helping the customer given the context. \
Your response should be 1 for compliant, 0 for non-compliant. \
You should model compliance after the guideline, and check if the agent is working to fulfill the target action.\
If the target interaction doesn't involve the agent working toward the target action as specified in the guideline, \
it should be non-compliant (0). Only evaluate what the agent is doing in the target interaction, not the context. \
First provide your reasoning by explaining how the next action matches the target agent behaviour. \
Your response should be in Reasoning: [] Score: [] format.\n\
Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\nContext:\n{dialog}\n\Target Action: {wf_name_mapping[next_action]}\n\nTarget Interaction: Agent: {target}\n\nAnswer: "


#     prompt = f"The following is a situation where a customer agent is helping a user with a customer service issue. \
# Read the provided examples and determine if, in the target interaction the agent is responding in a manner that achieves the specified workflow action \
# (Next Action) in helping the customer given the context. \
# `None` next action specifies greeting or session ending only. \
# Your response should be 1 for workflow compliant, 0 for non-compliant. Also explain your reasoning. \
# You should model compliance after the agent behaviour in the provided examples.\n\n\
# {example_str}Context:\n{dialog}\n\nNext Action: {next_action}\n\nTarget: Agent: {target}\n\nScore: "



# Additionally, the system utterance can be marked as 0 in the following cases: While the utterance is not directly about the next action, \
# (1) it is responding to the client or following the dialog context in a plausible way, or (2) gathering information regarding \
# the fulfillment of the next action, or (3) the next action is None and the utterance follows context, including greeting and sending-off.\n\
# Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\n{example_str}Dialog:\n{dialog}\n\n[NEXT WORKFLOW]: {next_action}\n\n[SYSTEM]: {target}\n\nScore: "
#  Also explain the reasoning for the score.\n\n\

    return prompt


def llm_fluency_evaluate_prompt(target: str = None, subflow: str = None, dialog: str="", convo_id=None, next_action=""):
    
    prompt = f"After reading the following exchange between a customer agent and a user, judge \
if the agent is communicating in a fluent and coherent way. Output 1 for yes, 0 for no.\
\nInput: {target}\nScore: "

#     prompt = f"The following is a situation where a customer agent is helping a user with a customer service issue. \
# Read the provided examples and determine if, in the target interaction the agent is responding in a manner that achieves the specified workflow action \
# (Next Action) in helping the customer given the context. \
# `None` next action specifies greeting or session ending only. \
# Your response should be 1 for workflow compliant, 0 for non-compliant. Also explain your reasoning. \
# You should model compliance after the agent behaviour in the provided examples.\n\n\
# {example_str}Context:\n{dialog}\n\nNext Action: {next_action}\n\nTarget: Agent: {target}\n\nScore: "



# Additionally, the system utterance can be marked as 0 in the following cases: While the utterance is not directly about the next action, \
# (1) it is responding to the client or following the dialog context in a plausible way, or (2) gathering information regarding \
# the fulfillment of the next action, or (3) the next action is None and the utterance follows context, including greeting and sending-off.\n\
# Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\n{example_str}Dialog:\n{dialog}\n\n[NEXT WORKFLOW]: {next_action}\n\n[SYSTEM]: {target}\n\nScore: "
#  Also explain the reasoning for the score.\n\n\

    return prompt


def llm_compliance_evaluate_prompt(target: str = None, subflow: str = None, dialog: str="", convo_id=None, next_action=""):
    

    guideline_str = retrieve_guideline_text(subflow)
    #examples = retrieve_examples(subflow, convo_id = convo_id)

    

    examples = [ x["input"] for x in pos_examples if x["convo_id"] !=convo_id and x["subflow"] == subflow and next_action in x["input"]]    

    #examples = random.sample(examples, 2)

    example_str = ""
    for i, e in enumerate(examples):
        #print("Example: ", e)
        example_str += f"Example {i+1}\n{e}\n\n" #format_in_md(e)
        #print("ex_str:", example_str)
    dialog = dialog[:-len("Agent: ")]
    prompt = f"The following is a situation where a customer agent is helping a user with a customer service issue. \
Read the provided guideline for the corresponding workflow and determine if the agent is responding in a manner that achieves the specified workflow step \
(Target Action) in helping the customer given the context. \
Your response should be 1 for compliant, 0 for non-compliant. \
You should model compliance after the guideline, and check if the agent is working to fulfill the target action.\
If the target interaction doesn't involve the agent working toward the target action as specified in the guideline, \
it should be non-compliant (0). Only evaluate what the agent is doing in the target interaction, not the context. \
First provide your reasoning by explaining how the next action matches the target agent behaviour. \
Your response should be in Reasoning: [] Score: [] format.\n\
Workflow: {subflow}\n\nGuideline:\n{guideline_str}\n\nContext:\n{dialog}\n\Target Action: {wf_name_mapping[next_action]}\n\nTarget Interaction: Agent: {target}\n\nAnswer: "


    return prompt

if __name__ == "__main__":
    #fire.Fire(construct_evaluate_prompt)
    pass
    #workflows, keys = read_kb()


    #fire.Fire(test)
    #fire.Fire(retrieve_guideline_text)
    #a = fire.Fire(retrieve_examples)
    #print(a)