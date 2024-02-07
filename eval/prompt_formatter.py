import json

import fire
import markdown
"""
For prompting with LLMs
Ethan's tip is to format in markdown syntax
"""
from typing import List, Dict

def dic2str(dic, lev=1):
    #sset = []
    s = ""
    for k,v in dic.items():
        #print(type(v))
        #continue
        if type(v) == list and len(v) > 0 and type(v[0]) == dict:
            sub = sub = "#"*lev + " "+str(k).strip() +": " + " ".join(["\n"+dic2str(vv, lev+1) for vv in v])
        elif type(v) == list:
            sub = "#"*lev + " "+str(k).strip() +": " + " ".join([str(x) for x in v])
        elif type(v) == dict:
            sub = "#"*lev + " "+str(k).strip() +": \n"+dic2str(v, lev+1)
        elif type(v) == str:
            sub = "#"*lev + " "+str(k).strip() +": " + v
        else:
            print(type(v))
            print("gg")
            exit()
        s += sub +"\n"
        #if k == "button":
        #    sset.append(v)

    #print(sset)
    #exit()
    s = s.replace("'","")
    s = s.replace("button", "action")
    return s
        

def convert2markdown(s):
    return markdown.markdown(s)


def format_in_md(dic):
    return convert2markdown(dic2str(dic))

def prepare_wf_sequence(path="./data/kb.json"):
    with open(path, "r") as fh:
        data =json.load(fh)

    #print(data)
    #print(dic2str(data))
    #print(convert2markdown(dic2str(data)))
    s =  convert2markdown(dic2str(data))
    s = s.replace("'","")
    return s

def prepare_wf_text(path="./data/guidelines.json"):
    with open(path, "r") as fh:
        data =json.load(fh)

    #print(data)
    #print(convert2markdown(dic2str(data)))
    
    s = convert2markdown(dic2str(data))
    #s.replace('\'','')
    s = s.replace("'","")
    return s

if __name__ == "__main__":
    #fire.Fire(test)

    #print(fire.Fire(prepare_wf_sequence))
    fire.Fire(prepare_wf_text)