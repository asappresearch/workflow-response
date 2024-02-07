import fire
import json
import random
from Levenshtein import distance, ratio
import numpy as np

def levenshtein_dist_for_lists(source, target):
    unique_elements = sorted(set(source + target)) 
    char_list = [chr(i) for i in range(len(unique_elements))]
    if len(unique_elements) > len(char_list):
        raise Exception("too many elements")
    else:
        unique_element_map = {ele:char_list[i]  for i, ele in enumerate(unique_elements)}
    source_str = ''.join([unique_element_map[ele] for ele in source])
    target_str = ''.join([unique_element_map[ele] for ele in target])
    #transform_list = Levenshtein.editops(source_str, target_str)
    #return distance(source_str, target_str)
    # ratio: (lensum - ldist) / lensum
    return ratio(source_str, target_str)
    

def jaccard_dist(source, target):
    source = set(source)
    target = set(target)

    sim = len(source.intersection(target)) / len(source.union(target))
    return sim #1.- sim

def metric(list_a, list_b):
    #print(list_a, list_b)

    #return levenshtein_dist_for_lists(list_a, list_b)
    return jaccard_dist(list_a, list_b)

def test():
    with open("./data/kb.json", "r") as fh:
        workflows = json.load(fh)

    random_flow = random.choice(list(workflows.keys()))
    print("Random flow:", random_flow)
    chosen_flow = workflows[random_flow]
    print(chosen_flow)

    all_flows = []
    for v in workflows.values():
        all_flows +=  list(v)
    all_flows = list(set(all_flows))

    gen_flow = []
    gen_len = random.randint(1, 6)
    for i in range(gen_len):
        rf = random.choice(all_flows)
        gen_flow += [ rf]
    prev = object()
    gen_flow = [prev:=v for v in gen_flow if prev!=v]

    print("Gen flow:", gen_flow)

    res1 = metric(chosen_flow, gen_flow)
    res2 = metric(gen_flow, chosen_flow)

    print(res1, res2)
    assert res1 == res2, "not symmetric"
    return


def compare_real_vs_kb_flow(SPLITS = ["train",  "dev", "test"]):
    with open("./data/kb.json", "r") as fh:
        kb_flows = json.load(fh) 

    with open("./data/abcd_v1.1.json", "r") as fh:
        abcd_data = json.load(fh)
        abcd = [ abcd_data[split] for split in SPLITS ]   
        new = []
        for s in abcd:
            new += s
        abcd = new
        
        real_flows =[]
        for convo in abcd:
            delexed = convo["delexed"]
            actions = [ ]
            for d in delexed:
                target = d["targets"]
                subflow = target[0]
                act_type= target[1]
                action = target[2]
                if act_type == "take_action":
                    actions += [action]
            prev = object()
            actions = [prev:=v for v in actions if prev!=v]

            real_flows.append((subflow, actions))

    print("kb_flows size:", len(kb_flows))
    print("real_flows size:", len(real_flows))

    rint = random.randint(0, len(real_flows)-1)
    print(real_flows[rint])

    results = []
    count = 0.0
    for name, flow in real_flows:
        try:
            kb = kb_flows[name]
        except:
            print("je bee")
            exit(0)
        
        real = flow

        lev = levenshtein_dist_for_lists(real, kb)
        jac = jaccard_dist(real, kb)

        # print("lev:", lev)
        # print("jac:", jac)
        results.append([lev, jac])
        #if lev == 1 or jac == 1:
        #    print("subflow:", name)
        #    print("kb:", kb)
        #    print("real:", real)

        if kb == real:
            count += 1

    print("lev:", np.mean([x[0] for x in results]))
    print("jac:", np.mean([x[1] for x in results]))
    print(f"kb==real: {int(count)}/{len(real_flows)}==", count/len(real_flows))

if __name__ == "__main__":
    #fire.Fire(test)

    fire.Fire(compare_real_vs_kb_flow)