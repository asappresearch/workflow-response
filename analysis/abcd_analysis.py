import json
from model.retrieve_openai import *
from collections import Counter
path = "./data/wc_seed_one_convo.json"


kb, keys = read_kb()
with open(path, 'r') as f:
    all_data = json.load(f)
    #data = all_data[split]
    splits = all_data.keys()

    for split in splits:
        print("="*30)
        print(split)

        offender_wf = []
        offender_sw = []
        data = all_data[split]
        counter = 0
        total = len(data)
        total_turns= 0
        counter_turn = 0
        counter_block = 0
        total_blocks = 0
        for convo in data:
            #print(convo)
            #input()
            subflow = None
            for i,turn in enumerate(convo["turns"]):
                
                #if turn["speaker"] != "system":
                    #print(turn)
                if turn["speaker"] != "system":
                    continue
                total_turns += 1
                if i == 0 or convo["turns"][i-1]["speaker"] != "action":
                    continue
                total_blocks += 1
                #print(turn)
                targets = turn["workflow_action"]
                #if targets == None:
                #    continue
                try:
                    subflow = targets[0]
                    workflow = targets[2]
                except Exception as e:
                    #print(e)
                    subflow = None
                    workflow = None
                    #exit()
                #print(subflow, workflow)

                if workflow != None and workflow != "None" and subflow != None and subflow != "None":
                    result1 = retrieve_guideline_text_action(subflow, workflow)
                    if result1 != -1:
                        result1 = 1
                    if workflow not in kb[subflow]:
                        result2 = -1
                    else:
                        result2 = 1

                    ## Option: Get guideline wrong
                    # Manage Pay Bill - subscription status
                    # refund status - update order

                    # if result1 == -1 and result2 == 1:
                    #     result = -1
                    # else:
                    #     result =1

                    ## Option 2: Probably annotator mistakes
                    assert result1 == result2
                    result = result2    #eesult2

                    if result == -1:
                        offender_wf.append(workflow)
                        offender_sw.append(subflow+"-"+workflow)
                    else:
                        counter += 1
                        counter_turn += 1
                        counter_block += 1

                        
        #print(f"Left convos: {counter} / Total: {total}")
        print(f"Total system turns: {total_turns}")
        print(f"Left blocks: {counter_block} / Total blocks: {total_blocks}")
        #print(Counter(offender_wf))
        print(len(offender_wf))
        #print(Counter(offender_sw))
        #input()

            