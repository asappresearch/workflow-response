import json
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
import copy

id2speaker = {0:"user", 1:"system"}
def prepare_seed_multiwoz(data, split_name=None):
    if split_name == "validation":
        split_name = "val"
    with open(f"./data/{split_name}_dials.json", "r") as fh:
        id2dialog = json.load(fh)
    

    new = []
    sid = 0
    for d in tqdm(data):
        
        d_id = d["dialogue_id"]
        if d_id not in id2dialog:
            continue
        user_utts = id2dialog[d_id]["usr"]
        sys_utts = id2dialog[d_id]["sys"]
        convo = { "convo_id":d_id, "sample_id":sid, "turns":[] }
        #print(d.keys())
        #input()
        turns = d["turns"]
        speaker = turns["speaker"]
        frames = turns["frames"]
        utterances = turns["utterance"]
        dialog_acts = turns["dialogue_acts"]
        # print("="*30)
        # print(turns)
        # print(frames)
        # input()
        assert len(speaker) == len(frames)
        assert len(speaker) == len(utterances)
        assert len(speaker) == len(dialog_acts)
        """
        delexicalizing
        """
        user_idx, sys_idx = 0, 0
        for i, s,f,u,d in zip(range(len(speaker)), speaker, frames, utterances, dialog_acts):
            # if s == 0:
            #     text = user_utts[user_idx]
            #     user_idx += 1
            # else:
            #     text = sys_utts[sys_idx]
            #     sys_idx += 1
            text = u.strip().lower()#text.strip()
            turn = { "speaker": id2speaker[s], "text":text, "dialog_act": d["dialog_act"], "frame": f}
            convo["turns"].append(turn)

        new.append(convo)
        sid += 1


    return new 

def format_intent_info(data):
    formatted_string = ""
    
    for item in data:
        active_intent = item['active_intent']
        requested_slots = ','.join(item['requested_slots']).strip()
        
        
        slots_values = dict(zip(item['slots_values']['slots_values_name'], [" ".join(i) for i in item['slots_values']['slots_values_list']]))

        formatted_string += f"Active Intent: {active_intent} "
        if requested_slots == "":
            formatted_string += f"Requested Slots: "
        else:
            formatted_string += f"Requested Slots: {requested_slots} "
        for key, value in slots_values.items():
            formatted_string += f"{key}: {value} "#.strip()#+" "
        #formatted_string += " "
    return formatted_string.strip()

def format_intent_info_only_updated(new_list, old_list):
    result = []

    # Create a dictionary for quick lookup of old items by active_intent
    old_items_by_intent = {item['active_intent']: item for item in old_list}

    for new_item in new_list:
        active_intent = new_item['active_intent']

        # Check if the active_intent exists in the old_list
        if active_intent in old_items_by_intent:
            old_item = old_items_by_intent[active_intent]

            # Compare slots_values_name and slots_values_list
            new_slot_values = new_item['slots_values']
            old_slot_values = old_item['slots_values']

            # Find the difference in slots_values_name
            new_name_set = set(new_slot_values['slots_values_name'])
            old_name_set = set(old_slot_values['slots_values_name'])
            new_names = list(new_name_set - old_name_set)

            # Find the differences in slots_values_list
            diff_list_values = {}
            for name in new_name_set:
                new_list_values = new_slot_values['slots_values_list'][new_slot_values['slots_values_name'].index(name)]
                try:
                    old_list_values = old_slot_values['slots_values_list'][old_slot_values['slots_values_name'].index(name)]
                except:
                    old_list_values = []
                if new_list_values != old_list_values:
                    diff_list_values[name] = new_list_values

            if new_names or diff_list_values:
                # Add the differences to the result
                new_item['slots_values']['slots_values_name'] = new_names
                new_item['slots_values']['slots_values_list'] = diff_list_values
                result.append(new_item)
        else:
            # Active intent not found in old_list, add it to the result
            result.append(new_item)
    
    formatted_string = ""
    print(new_list)
    print("+"*30)
    print(old_list)
    print("-"*30)
    print(result)
    input()

    for item in result:
        #continue
        active_intent = item['active_intent']
        requested_slots = ','.join(item['requested_slots']).strip()
        
        
        slots_values = dict(zip(item['slots_values']['slots_values_name'], [i[0] for i in item['slots_values']['slots_values_list']]))

        formatted_string += f"Active Intent: {active_intent} "
        if requested_slots == "":
            formatted_string += f"Requested Slots: "
        else:
            formatted_string += f"Requested Slots: {requested_slots} "
        for key, value in slots_values.items():
            formatted_string += f"{key}: {value} "#.strip()#+" "
        #formatted_string += " "
    return formatted_string.strip()


def prepare_with_da(seed):
    """
    need to have
    convo
        convo_id
        sample_id
        turns
            "text"
            "speaker"
            "workflow_action"
                workflow[2]
                [None]
    speaker
    """
    new = []
    da_set = []
    for convo in tqdm(seed):
        new_convo = { "convo_id": convo["convo_id"], "sample_id": convo["sample_id"], "turns":[]}
        turns = convo["turns"]
        new_turns = []

        curr_state = turns[-1]["dialog_act"]
        #print(curr_state)
        act = curr_state["act_type"]
        
        #active_intents = [ x["active_intent"] for x in curr_state]
        # if len(act) == 0:
        #     wf = None
        # else:
        #     wf = [None, None, " ".join(act)]
        #wf = act
        wf = [None, None, " ".join(act)]

        for turn in turns[::-1]:
            new_turn = turn.copy()
            #print(new_turn["frame"])
            state = new_turn["frame"]["state"]
            #print(state)
            # if state == []:
            #     action_string = "None"
            # else:
            #     action_string = ""
            #     for s in state:
            action_string = format_intent_info(state)
            # print(action_string)
            # input()
            new_turn["targets"] = [ "N/A", None, action_string ,[] ]
            del new_turn["frame"]
            act = turn["dialog_act"]
            # print(turn)
            # print(state)
            # input()
            act = act["act_type"]
            act = [ x.split("-")[1] for x in act ] # simple da
            da_set.extend(act)
            wf = [None, None, " ".join(act)]
            # active_intents = [ x["active_intent"] for x in state]
            # if len(active_intents) == 0:
            #     #wf = None
            #     pass
            # else:
            #     #wf = " ".join(active_intents)
            #     wf = [None, None, " ".join(active_intents)]
            new_turn["workflow_action"] = wf
            
            
            if new_turn["speaker"] == "user":
                """
                Adding Oracle Belief State
                """
                action_turn = new_turn.copy()
                action_turn["speaker"] = "action"
                action_turn["text"] = action_string
                #print(action_turn["targets"])
                #print(action_turn["text"])
                new_turns.append(action_turn)
            new_turns.append(new_turn)
            #elif new_turn["speaker"] == "user":
            #    past_action_string = action_string
            # print(turn)
            # print(active_intents)
            # print(wf)
            # input()
            #if len(state) > 1:
            #    print(state)
            #input()
        new_convo["turns"] = new_turns[::-1]
        new.append(new_convo)
    
    print("DA set:", set(da_set))
        
    return new

def prepare_with_intent(seed):
    """
    need to have
    convo
        convo_id
        sample_id
        turns
            "text"
            "speaker"
            "workflow_action"
                workflow[2]
                [None]
    speaker
    """
    new = []
    for convo in tqdm(seed):
        new_convo = { "convo_id": convo["convo_id"], "sample_id": convo["sample_id"], "turns":[]}
        turns = convo["turns"]
        new_turns = []

        # curr_state = turns[-1]["frame"]["state"]
        # active_intents = [ x["active_intent"] for x in curr_state]
        # if len(active_intents) == 0:
        #     wf = None
        # else:
        #     wf = [None, None, " ".join(active_intents)]

        # old_active_intents = active_intents
        # for turn in turns[::-1]:
        #     new_turn = turn.copy()
        #     new_turn["targets"] = [ "N/A"]
        #     del new_turn["frame"]
        #     del new_turn["dialog_act"]
        #     state = turn["frame"]["state"]
        #     #
        #     action_string = format_intent_info(state)
    
        #     active_intents = [ x["active_intent"] for x in state]
            

        #     if len(active_intents) == 0:
        #         #wf = None
        #         pass
        #     else:
        #         #wf = " ".join(active_intents)
        #         wf = [None, None, " ".join(active_intents)]
        #     new_turn["workflow_action"] = wf

        #     if new_turn["speaker"] == "user":
        #         """
        #         Adding Oracle Belief State
        #         """
        #         action_turn = new_turn.copy()
        #         action_turn["speaker"] = "action"
        #         action_turn["text"] = action_string
        #         #print(action_turn["targets"])
        #         #print(action_turn["text"])
        #         action_turn["targets"] = [ None, None, action_string, []]
        #         new_turns.append(action_turn)
            
        #     # print("="*30)
        #     # print(old_active_intents)
        #     # print(active_intents)
        #     # print(action_string)
        #     # print(new_turn)
        #     # input()

        #     new_turns.append(new_turn)
        #     old_active_intents = active_intents

        for idx, turn in enumerate(turns):
            new_turn = turn.copy()
            del new_turn["frame"]
            #del new_turn["dialog_act"]

            state = turn["frame"]["state"]
            # if turn["speaker"] == "user":
            #     action_string = format_intent_info(state)
            #     # if idx == 0:
            #     #     action_string = format_intent_info(state)
            #     # else:
            #     #     action_string = format_intent_info_only_updated(state, turns[idx-2]["frame"]["state"])
            #     # print("="*30)
            #     # print(idx)
            #     # print(action_string)
            #     # input()

            active_intents = [ x["active_intent"] for x in state]

            act = turn["dialog_act"]
            # print(turn)
            # print(state)
            # input()
            act = act["act_type"]
            #act = [ x.split("-")[1] for x in act ] # simple da
            
            #input()
            action_string = " ".join(act)
            # print("act:", act)
            # print(action_string)
            # input()

            if len(active_intents) == 0:
                #wf = None
                #pass
                wf = None
            else:
                #wf = " ".join(active_intents)
                wf = [None, None, " ".join(active_intents)]
            
            if new_turn["speaker"] == "user":
                prev_wf = wf


            if new_turn["speaker"] == "system":
                wf = prev_wf
                # if prev_wf is not None:
                #     wf = prev_wf
                # else:
                #     print("bug")
                #     exit()
            new_turn["workflow_action"] = wf
            new_turns.append(new_turn)

            if new_turn["speaker"] == "system":
                """
                Adding Oracle Belief State
                """
                action_turn = new_turn.copy()
                action_turn["speaker"] = "action"
                action_turn["text"] = action_string
                #print(action_turn["targets"])
                #print(action_turn["text"])
                action_turn["targets"] = [ None, None, action_string, []]
                #new_turns.append(action_turn) 

            if new_turn["speaker"] == "system":
                if action_turn != None:
                    new_turns.append(action_turn)
                else:
                    print("nooooo!!")
                    exit()
                action_turn = None
        new_convo["turns"] = new_turns#[::-1]
        new.append(new_convo)
        
        
    return new


# def prepare_with_intent(seed):
#     """
#     need to have
#     convo
#         convo_id
#         sample_id
#         turns
#             "text"
#             "speaker"
#             "workflow_action"
#                 workflow[2]
#                 [None]
#     speaker
#     """
#     new = []
#     for convo in tqdm(seed):
#         new_convo = { "convo_id": convo["convo_id"], "sample_id": convo["sample_id"], "turns":[]}
#         turns = convo["turns"]
#         new_turns = []

#         curr_state = turns[-1]["frame"]["state"]
#         active_intents = [ x["active_intent"] for x in curr_state]
#         if len(active_intents) == 0:
#             wf = None
#         else:
#             wf = [None, None, " ".join(active_intents)]

#         for turn in turns[::-1]:
#             new_turn = turn.copy()
#             new_turn["targets"] = [ "N/A"]
#             del new_turn["frame"]
#             del new_turn["dialog_act"]
#             state = turn["frame"]["state"]
#             active_intents = [ x["active_intent"] for x in state]
#             if len(active_intents) == 0:
#                 #wf = None
#                 pass
#             else:
#                 #wf = " ".join(active_intents)
#                 wf = [None, None, " ".join(active_intents)]
#             new_turn["workflow_action"] = wf
#             new_turns.append(new_turn)
#             # print(turn)
#             # print(active_intents)
#             # print(wf)
#             # input()
#             #if len(state) > 1:
#             #    print(state)
#             #input()
#         new_convo["turns"] = new_turns[::-1]
#         new.append(new_convo)
        
        
#     return new


def create_data():
    multiwoz = { k:prepare_seed_multiwoz(v,k) for k,v in read_multiwoz().items() }

    with open("./data/bs_multiwoz_intent.json", "w") as fh:
        dic = { "train": prepare_with_intent(multiwoz["train"]) , "dev": prepare_with_intent(multiwoz["validation"]), "test": prepare_with_intent(multiwoz["test"])}

        json.dump(dic, fh, indent=4)

    # with open("./data/simple_da_bs_multiwoz_da.json", "w") as fh:
    #     dic = { "train": prepare_with_da(multiwoz["train"]) , "dev": prepare_with_da(multiwoz["validation"]), "test": prepare_with_da(multiwoz["test"])}

    #     json.dump(dic, fh, indent=4)


def read_multiwoz():
    dataset = load_dataset("multi_woz_v22")
    #print(dataset)
    # "train", "validation", "test"
    return dataset

from collections import Counter
def default_filter(dialog):
    """
    This filter only accepts dialogues with 1 service (e.g. "hotel", "train") requested by the user
    """
    turns = dialog["turns"]
    frames = [ x["frame"] for x in turns]
    #services = [ " ".join(x["dialog_act"]["act_type"]) for x in turns] 
    services = [ x["dialog_act"]["act_type"] for x in turns] 
    services = [ x for temp in services for x in temp ] 
    services = [ x.split("-")[1] for x in services if x[0].isupper() ]

    services = set(services)
    print(services)
    # input()
    if len(services) == 1:
        #if services[0] != " ":
        #    if services[0].
        services = [ x["dialog_act"]["act_type"] for x in turns] 
        services = [ x for temp in services for x in temp ]     
        #print(services)
        return True
    return False

def filter_regular_convos(split, condition = default_filter):
    """
    TODO: Identify and filter out convos that have "regular" workflows
    1. extract workflows
    2. cluster workflows
    3. filter out outliers
    """
    filtered = []
    for d in tqdm(split):
        turns = d["turns"]
        frames = [ x["frame"] for x in turns]
        #services = [ " ".join(x["service"]) for x in frames]
        #ss.extend(services)
        if condition(d):
            filtered.append(d)
        else:
            continue
        #print()
        #print('='*30)
        #print(set(services))
        #print(d)
        #input()
    # print(Counter(ss))
    # print(set(ss))
    # input()
    print("filtered:", len(filtered))
    #input()
    return filtered



def create_filtered_data():
    multiwoz = { k:filter_regular_convos(prepare_seed_multiwoz(v,k)) for k,v in read_multiwoz().items() }

    with open("./data/filtered_multiwoz_intent.json", "w") as fh:
        dic = { "train": prepare_with_intent(multiwoz["train"]) , "dev": prepare_with_intent(multiwoz["validation"]), "test": prepare_with_intent(multiwoz["test"])}

        json.dump(dic, fh, indent=4)

    # with open("./data/filtered_multiwoz_da.json", "w") as fh:
    #     dic = { "train": prepare_with_da(multiwoz["train"]) , "dev": prepare_with_da(multiwoz["validation"]), "test": prepare_with_da(multiwoz["test"])}

    #     json.dump(dic, fh, indent=4)


if __name__ == "__main__":
    if False:
        data = read_multiwoz()
        test = data["test"]
        test = prepare_seed_multiwoz(test)
        dat = prepare_with_intent(test)
        # dat = prepare_with_da(test)
        #print(dat)
        for conv in dat:
            print("="*30)
            for t in conv["turns"]:
                print(t)
                input() 

    create_data()
    #create_filtered_data()