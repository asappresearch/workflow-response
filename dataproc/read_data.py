import json
#import ujson as json

def read_seed_data(fp = "./data/wc_seed.json"):
    with open(fp, "r") as fh:
        data = json.load(fh)


    # new = {}
    # for split, convos in data.items():
    #     for i, convo in enumerate(convos):
    #         turns = convo["turns"]
    #         for ii, t in enumerate(turns):
    #             if ii > 0:
    #                 if t["speaker"] == "action" and turns[ii-1]["speaker"] == "system":
    #                     print("yes")
    #                     exit()
    #                 #else:
    #                 #    print(t["speaker"])

    # exit()
    return data

def read_seed_future_actions_data(fp = "./data/wc_seed_future_actions.json"):
    with open(fp, "r") as fh:
        data = json.load(fh)

    return data

def universal_check(convo):
    """
    input: convo (dict object containing turns list (which is a list of turn dics) )
    output: T / F : T if passes check for dialog response training
    """
    turns = convo["turns"]
    if len(turns) == 1:
        return False
    context = turns[:-1]
    target = turns[-1]
    if target["speaker"] != "system":
        return False

    return True

def prep_b1_data():
    """
    <Finetuning without workflow>
    Just utterances ==> target utterance
    !!!: No actions
    """
    raw = read_seed_data()
    new = {}
    for split, convos in raw.items():
        s_count = 0
        new[split] = []
        for i, convo in enumerate(convos):
            if not universal_check(convo):
                continue

            turns = convo["turns"]
            context = turns[:-1]
            # filter out actions
            context = [ x for x in context if x["speaker"] !="action"]
            target = turns[-1]

            inp = ""
            for t in context:
                inp += t["speaker"]+": "
                inp += t["text"]+"\n"
            # last new line in inp is unnecessary
            inp = inp.strip("\n")

            tgt = target["speaker"]+": "+target["text"]

            datum = { "sample_id": s_count,
                "convo_id": convo["convo_id"],
                "input": inp,
                "target": tgt
            }

            s_count += 1
            new[split] += [datum]

    with open("./data/b1.json", "w") as fh:
        json.dump(new,fh)
    return new

def prep_b2_data():
    """
    <Finetuning with workflow>
    utterances + Previous acitons ==> target utterance
    """
    raw = read_seed_data()
    new = {}
    for split, convos in raw.items():
        s_count = 0
        new[split] = []
        for i, convo in enumerate(convos):
            if not universal_check(convo):
                continue

            turns = convo["turns"]
            context = turns[:-1]
            target = turns[-1]

            inp = ""
            for t in context:
                if t["speaker"] == "action":
                    action_str = t["speaker"]+": "
                    act = t["targets"][2]
                    svals = t["targets"][3]
                    if svals == []:
                        svals == None
                    action_str += act + " " + str(svals) +"\n"
                    inp += action_str
                else:
                    inp += t["speaker"]+": "
                    inp += t["text"]+"\n"
            # last new line in inp is unnecessary
            inp = inp.strip("\n")

            tgt = target["speaker"]+": "+target["text"]

            datum = { "sample_id": s_count,
                "convo_id": convo["convo_id"],
                "input": inp,
                "target": tgt
            }

            s_count += 1
            new[split] += [datum]

    with open("./data/b2.json", "w") as fh:
        json.dump(new,fh)
    return new



def prep_workflow_prediction_data():
    """
    Workflow Prediction
    utterances + Previous acitons ==> target workflow-action (plan)
    """
    raw = read_seed_data()
    new = {}
    for split, convos in raw.items():
        s_count = 0
        new[split] = []
        for i, convo in enumerate(convos):
            if not universal_check(convo):
                """
                TODO: Think and decide if universal check still is needed here
                i.e. do we need to also predict action when next utt is not systems?

                """
                continue

            turns = convo["turns"]
            context = turns[:-1]
            target = turns[-1]

            inp = ""
            for t in context:
                if t["speaker"] == "action":
                    action_str = t["speaker"]+": "
                    act = t["targets"][2]
                    svals = t["targets"][3]
                    if svals == []:
                        svals == None
                    action_str += act + " " + str(svals) +"\n"
                    inp += action_str
                else:
                    inp += t["speaker"]+": "
                    inp += t["text"]+"\n"
            # last new line in inp is unnecessary
            inp = inp.strip("\n")

            target_action = target["workflow_action"]

            if target_action == None:
                t_act = None
                t_svals = None
            else:
                t_act = target_action[2]
                t_svals = target_action[3]
                if t_svals == []:
                    t_svals == None
            # For now, we are not interested in predicting slotvalues 
            # (Since predicted workflow is only a plan, need more information)
            tgt = "wf-action: "+str(t_act) #+ " " + str(t_svals) 

            datum = { "sample_id": s_count,
                "convo_id": convo["convo_id"],
                "input": inp,
                "target": tgt
            }

            s_count += 1
            new[split] += [datum]

    with open("./data/wf_prediction.json", "w") as fh:
        json.dump(new,fh)
    return new


def prep_utterance_from_workflow_data():
    """
    utterances + Previous acitons + Predicted workflow prediction ==> target utterance
    """
    raw = read_seed_data()
    new = {}
    for split, convos in raw.items():
        s_count = 0
        new[split] = []
        for i, convo in enumerate(convos):
            if not universal_check(convo):
                """
                TODO: Think and decide if universal check still is needed here
                i.e. do we need to also predict action when next utt is not systems?

                """
                continue

            turns = convo["turns"]
            context = turns[:-1]
            target = turns[-1]

            inp = ""
            for t in context:
                if t["speaker"] == "action":
                    action_str = t["speaker"]+": "
                    act = t["targets"][2]
                    svals = t["targets"][3]
                    if svals == []:
                        svals == None
                    action_str += act + " " + str(svals) +"\n"
                    inp += action_str
                else:
                    inp += t["speaker"]+": "
                    inp += t["text"]+"\n"


            target_action = target["workflow_action"]

            if target_action == None:
                t_act = None
                t_svals = None
            else:
                t_act = target_action[2]
                t_svals = target_action[3]
                if t_svals == []:
                    t_svals == None
            # For now, we are not interested in predicting slotvalues 
            # (Since predicted workflow is only a plan, need more information)
            #tgt = str(t_act) #+ " " + str(t_svals) 

    
            inp += "wf-action: "+str(t_act) 

            tgt = target["speaker"]+": "+target["text"] 
            
            datum = { "sample_id": s_count,
                "convo_id": convo["convo_id"],
                "input": inp,
                "target": tgt
            }

            s_count += 1
            new[split] += [datum]

    with open("./data/utt_prediction.json", "w") as fh:
        json.dump(new,fh)
    return new



def prep_utterance_from_future_workflow_data():
    """
    utterances + Previous acitons + Predicted workflow prediction ==> target utterance
    """
    raw = read_seed_future_actions_data()
    new = {}
    for split, convos in raw.items():
        s_count = 0
        new[split] = []
        for i, convo in enumerate(convos):
            if not universal_check(convo):
                """
                TODO: Think and decide if universal check still is needed here
                i.e. do we need to also predict action when next utt is not systems?

                """
                continue

            turns = convo["turns"]
            context = turns[:-1]
            target = turns[-1]

            inp = ""
            for t in context:
                if t["speaker"] == "action":
                    action_str = t["speaker"]+": "
                    act = t["targets"][2]
                    svals = t["targets"][3]
                    if svals == []:
                        svals == None
                    action_str += act + " " + str(svals) +"\n"
                    inp += action_str
                else:
                    inp += t["speaker"]+": "
                    inp += t["text"]+"\n"


            target_action = target["workflow_action"]

            if target_action == None:
                t_act = None
                t_svals = None
            else:
                t_act = [ x[2] if x is not None else None for x in target_action ] #target_action[2]
                t_svals = [x[3] if x is not None else None for x in target_action ] #target_action[3]
                if t_svals == []:
                    t_svals == None
            # For now, we are not interested in predicting slotvalues 
            # (Since predicted workflow is only a plan, need more information)
            #tgt = str(t_act) #+ " " + str(t_svals) 

    
            inp += "wf-action: "+str(t_act) 

            tgt = target["speaker"]+": "+target["text"] 
            
            datum = { "sample_id": s_count,
                "convo_id": convo["convo_id"],
                "input": inp,
                "target": tgt
            }

            s_count += 1
            new[split] += [datum]

    with open("./data/utt_prediction_future_actions.json", "w") as fh:
        json.dump(new,fh)
    return new



def prep_utterance_from_oracle_kb_sequence():
    """
    utterances + Previous acitons + Predicted workflow prediction ==> target utterance
    """
    with open("./data/kb.json", "r") as fh:
        kb = json.load(fh)   

    raw = read_seed_data()
    new = {}
    for split, convos in raw.items():
        s_count = 0
        new[split] = []
        for i, convo in enumerate(convos):
            if not universal_check(convo):
                """
                TODO: Think and decide if universal check still is needed here
                i.e. do we need to also predict action when next utt is not systems?

                """
                continue

            turns = convo["turns"]
            context = turns[:-1]
            target = turns[-1]

            inp = ""
            for t in context:
                flow = t["targets"][0]
                if t["speaker"] == "action":
                    action_str = t["speaker"]+": "
                    act = t["targets"][2]
                    svals = t["targets"][3]
                    if svals == []:
                        svals == None
                    action_str += act + " " + str(svals) +"\n"
                    inp += action_str
                else:
                    inp += t["speaker"]+": "
                    inp += t["text"]+"\n"


            target_action = target["workflow_action"]

            if target_action == None:
                t_act = None
                t_svals = None
            else:
                t_act = target_action[2]
                t_svals = target_action[3]
                if t_svals == []:
                    t_svals == None
            # For now, we are not interested in predicting slotvalues 
            # (Since predicted workflow is only a plan, need more information)
            #tgt = str(t_act) #+ " " + str(t_svals) 

    
            #inp += "wf-action: "+str(t_act) 
            inp += "wf-action: "+str(kb[flow])

            tgt = target["speaker"]+": "+target["text"] 
            
            datum = { "sample_id": s_count,
                "convo_id": convo["convo_id"],
                "input": inp,
                "target": tgt,
                #"kb_sequence": kb[flow]
            }
            # print(datum)
            # input()

            s_count += 1
            new[split] += [datum]

    with open("./data/kb_utt_prediction.json", "w") as fh:
        json.dump(new,fh)
    return new



def prep_utterane_reward_data():
    """
    TODO: do when baseline and cascading model tested
    """
    return


def print_data_stats(data):
    for k,v in data.items():
        print("="*30)
        print(k)
        print("size:", len(v))
        none_actions = [ x for x in v if x["input"].endswith("None")]
        print("predictted None action size:", len(none_actions))

if __name__ == "__main__":
    #print(read_seed_data()["train"][0])
    #data = prep_workflow_prediction_data()
    #prep_b1_data()
    #prep_b2_data()
    #prep_workflow_prediction_data()
    prep_utterance_from_workflow_data()
    #prep_utterance_from_future_workflow_data()
    #prep_utterance_from_oracle_kb_sequence()
    exit()
    data = prep_utterance_from_workflow_data()
    print_data_stats(data)
    

    import random
    rsplit = random.choice(["train", "test","dev"])
    randint = random.randint(0,len(data[rsplit])-1)
    print(randint)
    print(len(data[rsplit]))
    print(data[rsplit][randint])
