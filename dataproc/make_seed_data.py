
import json

def make_seed_data(fp = "./data/abcd_v1.1.json"):

    with open(fp, "r") as fh:
        data = json.load(fh)

    #print(data.keys())

    smap = { "agent": "system", "customer":"user", "action":"action" }

    new_data = {}
    #scount = 0
    for k,v in data.items():
        print(k)
        new_data[k] = []
        convos = v
        scount = 0
        for convo in convos:
            delexed = convo["delexed"]
            
            turns = []
            #curr_action = None
            if delexed[-1]["speaker"] == "action":
                curr_action = delexed[-1]["targets"]
            else:
                curr_action = None

            for i, turn in enumerate(delexed[::-1]):
                # mapping it to diff name
                turn["speaker"] = smap[turn["speaker"]]
                dic = { }
                dic.update(turn)
                del dic["candidates"]
                # conceptually, if the action is said (as utterance) it is done
                # so next action should be predicted from that action-utterance
                # Actually .. this has to do with what the workflow-action label is supposed to be
                # (1) Future (after this turn)'s action then it should be here
                # (2) Current (this turn)'s action then it should be down there (2)
                #dic["workflow_action"] = curr_action

                if turn["speaker"] == "action":
                    curr_action = turn["targets"]
                dic["workflow_action"] = curr_action # (2)
                turns.append(dic)
            turns = turns[::-1]

            # this loop is to implement following logic
            # if client / user hasn't said anything, then 
            # the workflow-action must be none
            user_encountered = False
            for i, turn in enumerate(turns):
                if turn["speaker"] == "user":
                    turn["workflow_action"] = None
                    break
                else:
                    if turn["speaker"] == "system":
                        turn["workflow_action"] = None

            accumulated = []
            for i, turn in enumerate(turns):
                accumulated.append(turn)
                dic = {
                        "sample_id": scount,
                        "convo_id": convo["convo_id"],
                        "turns": accumulated.copy()
                        }
                new_data[k].append(dic)
                scount += 1
            #break
    with open("./data/wc_seed.json", "w") as fh:
        json.dump(new_data, fh, indent=4)


def make_seed_data_one_convo(fp = "./data/abcd_v1.1.json"):

    with open(fp, "r") as fh:
        data = json.load(fh)

    #print(data.keys())

    smap = { "agent": "system", "customer":"user", "action":"action" }

    new_data = {}
    #scount = 0
    for k,v in data.items():
        print(k)
        new_data[k] = []
        convos = v
        scount = 0
        for convo in convos:
            delexed = convo["delexed"]
            original = convo["original"]

            assert(len(delexed) == len(original))
            for d,o in zip(delexed, original):
                assert d["speaker"] == o[0]
                d["text"] = o[1].lower()

            # TODO: finishing up conversation
            delexed.append({"speaker":"action", "targets": [None, None, "end-dialog", [], -1], "candidates": None, "text":"dialogue ended."})
            
            turns = []
            #curr_action = None
            if delexed[-1]["speaker"] == "action":
                curr_action = delexed[-1]["targets"]
            else:
                curr_action = None

            for i, turn in enumerate(delexed[::-1]):
                # mapping it to diff name
                turn["speaker"] = smap[turn["speaker"]]
                dic = { }
                dic.update(turn)
                del dic["candidates"]
                # conceptually, if the action is said (as utterance) it is done
                # so next action should be predicted from that action-utterance
                # Actually .. this has to do with what the workflow-action label is supposed to be
                # (1) Future (after this turn)'s action then it should be here
                # (2) Current (this turn)'s action then it should be down there (2)
                #dic["workflow_action"] = curr_action

                if turn["speaker"] == "action":
                    curr_action = turn["targets"]
                dic["workflow_action"] = curr_action # (2)
                turns.append(dic)
            turns = turns[::-1]

            # this loop is to implement following logic
            # if client / user hasn't said anything, then 
            # the workflow-action must be none
            user_encountered = False
            for i, turn in enumerate(turns):
                if turn["speaker"] == "user":
                    turn["workflow_action"] = None
                    break
                else:
                    if turn["speaker"] == "system":
                        turn["workflow_action"] = None

            accumulated = []
            for i, turn in enumerate(turns):
                accumulated.append(turn)
            dic = {
                    "sample_id": scount,
                    "convo_id": convo["convo_id"],
                    "turns": accumulated.copy()
                    }
            new_data[k].append(dic)
            scount += 1
            #break
    with open("./data/wc_seed_one_convo.json", "w") as fh:
        json.dump(new_data, fh, indent=4)



def make_seed_data_future_actions(fp = "./data/abcd_v1.1.json"):

    with open(fp, "r") as fh:
        data = json.load(fh)

    #print(data.keys())

    smap = { "agent": "system", "customer":"user", "action":"action" }

    new_data = {}
    #scount = 0
    for k,v in data.items():
        print(k)
        new_data[k] = []
        convos = v
        scount = 0
        for convo in convos:
            delexed = convo["delexed"]

            original = convo["original"]

            assert(len(delexed) == len(original))
            for d,o in zip(delexed, original):
                assert d["speaker"] == o[0]
                d["text"] = o[1].lower()

            # TODO: finishing up conversation
            delexed.append({"speaker":"action", "targets": [None, None, "end-dialog", [], -1], "candidates": None, "text":"dialogue ended."})
            

            turns = []
            #curr_action = None
            if delexed[-1]["speaker"] == "action":
                curr_actions = [delexed[-1]["targets"]]
            else:
                curr_actions = [None]

            for i, turn in enumerate(delexed[::-1]):
                # mapping it to diff name
                turn["speaker"] = smap[turn["speaker"]]
                dic = { }
                dic.update(turn)
                del dic["candidates"]
                # conceptually, if the action is said (as utterance) it is done
                # so next action should be predicted from that action-utterance
                # Actually .. this has to do with what the workflow-action label is supposed to be
                # (1) Future (after this turn)'s action then it should be here
                # (2) Current (this turn)'s action then it should be down there (2)
                #dic["workflow_action"] = curr_action

                if turn["speaker"] == "action":
                    curr_actions = [turn["targets"] ] + curr_actions
                dic["workflow_action"] = curr_actions # (2)
                turns.append(dic)
            turns = turns[::-1]

            # this loop is to implement following logic
            # if client / user hasn't said anything, then 
            # the workflow-action must be none
            user_encountered = False
            for i, turn in enumerate(turns):
                if turn["speaker"] == "user":
                    turn["workflow_action"] =[ None]
                    break
                else:
                    if turn["speaker"] == "system":
                        turn["workflow_action"] = [None]

            accumulated = []
            for i, turn in enumerate(turns):
                accumulated.append(turn)
                dic = {
                        "sample_id": scount,
                        "convo_id": convo["convo_id"],
                        "turns": accumulated.copy()
                        }
                new_data[k].append(dic)
                scount += 1
            #break
    with open("./data/wc_seed_future_actions.json", "w") as fh:
        json.dump(new_data, fh, indent=4)


def make_seed_data_future_actions_one_convo(fp = "./data/abcd_v1.1.json"):

    with open(fp, "r") as fh:
        data = json.load(fh)

    #print(data.keys())

    smap = { "agent": "system", "customer":"user", "action":"action" }

    new_data = {}
    #scount = 0
    for k,v in data.items():
        print(k)
        new_data[k] = []
        convos = v
        scount = 0
        for convo in convos:
            delexed = convo["delexed"]
            original = convo["original"]
            
            turns = []
            #curr_action = None
            if delexed[-1]["speaker"] == "action":
                curr_actions = [delexed[-1]["targets"]]
            else:
                curr_actions = [None]

            assert(len(delexed) == len(original))
            for d,o in zip(delexed, original):
                assert d["speaker"] == o[0]
                d["text"] = o[1].lower()

            # TODO: finishing up conversation
            delexed.append({"speaker":"action", "targets": [None, None, "end-dialog", [], -1], "candidates": None, "text":"dialogue ended."})
            

            for i, turn in enumerate(delexed[::-1]):
                # mapping it to diff name
                turn["speaker"] = smap[turn["speaker"]]
                dic = { }
                dic.update(turn)
                del dic["candidates"]
                # conceptually, if the action is said (as utterance) it is done
                # so next action should be predicted from that action-utterance
                # Actually .. this has to do with what the workflow-action label is supposed to be
                # (1) Future (after this turn)'s action then it should be here
                # (2) Current (this turn)'s action then it should be down there (2)
                #dic["workflow_action"] = curr_action

                if turn["speaker"] == "action":
                    curr_actions = [turn["targets"] ] + curr_actions
                dic["workflow_action"] = curr_actions # (2)
                turns.append(dic)
            turns = turns[::-1]

            # this loop is to implement following logic
            # if client / user hasn't said anything, then 
            # the workflow-action must be none
            user_encountered = False
            for i, turn in enumerate(turns):
                if turn["speaker"] == "user":
                    turn["workflow_action"] =[ None]
                    break
                else:
                    if turn["speaker"] == "system":
                        turn["workflow_action"] = [None]

            accumulated = []
            for i, turn in enumerate(turns):
                accumulated.append(turn)
            dic = {
                    "sample_id": scount,
                    "convo_id": convo["convo_id"],
                    "turns": accumulated.copy()
                    }
            new_data[k].append(dic)
            scount += 1
            #break
    with open("./data/wc_seed_future_actions_one_convo.json", "w") as fh:
        json.dump(new_data, fh, indent=4)




if __name__ == "__main__":
    make_seed_data_one_convo()
    make_seed_data_future_actions_one_convo()
