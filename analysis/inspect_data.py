import json

fp = "./data/rg_v1.1.json"

with open(fp, "r") as fh:
    data = json.load(fh)

for k,v in data.items():
    print("split:", k)

    t_a = [ a for a in v if a["target"].startswith("action") ]
    t_u = [ a for a in v if a["target"].startswith("user") ]
    t_s = [ a for a in v if a["target"].startswith("system") ]

    print("total len:", len(v))
    print("target action len:", len(t_a))
    print("target user len:", len(t_u))
    print("target system len:", len(t_s))
    print()


    


