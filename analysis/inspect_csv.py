import fire
import csv
import glob
from model.constants import *
#def inspect(models=["b1" , "b2", "utt_prediction_oracle_wf", "utt_prediction_future_actions_oracle_wf", "utt_prediction_cascade_wf" ]):
#def inspect(models=["utt_prediction_oracle_wf", "utt_prediction_cascade_wf" ]):
def inspect(models=["b1" , "b2", "utt_prediction_oracle_wf",  "utt_prediction_cascade_wf" ]):
    #models = ["utt_prediction_e2e"] # remove
    #fprefix = "./test_results/st/"
    fprefix = "./test_results/st"
    models = glob.glob(fprefix+"/**/epoch2/")
    models = [ x.replace(fprefix,"") for x in models]
    #print(models)
    # if cascade:
    #     cascade_datapath="./test_results/utt_prediction_cascade_wf/evaluation_tf.csv"

    eval_data = {}
    for model in models:
        eval_data[model] = []
        datapath = fprefix + model + "/evaluation_tf.csv"
        with open(datapath, 'r') as data:
            for line in csv.DictReader(data):
                context = line["context"]#+line["response_1"].strip()+"\nsystem: "
                response = line["response_1"].strip()
                dic = { "context": context, "response": response}
                if "true_response" in line:
                    true_response = line["true_response"]
                else:
                    true_response = None
                if "true_wf" in line:
                    true_wf = line["true_wf"]
                else:
                    true_wf = "Oracle"
                dic["true_response"] = true_response
                dic["true_wf"] = true_wf
                eval_data[model].append(dic)
            #dic = { "context": context, "response": response, "true_wf": true_wf }

    print([len(x) for x in eval_data.values()])    
    assert len(set([len(x) for x in eval_data.values()])) == 1, "the models being compared do not have equal-sized generation sets!"

    for i in range(len(list(eval_data.values())[0])):
        print("="*30)
        #print(dic) #
        
        true = None
        twf = None
        for model in models:
            tr = eval_data[model][i]["true_response"]
            if tr is not None and "wf" not in model:
                true = tr
            true_wf = eval_data[model][i]["true_wf"]
            if true_wf != "Oracle":
               twf = true_wf
            
            if "utt_prediction_e2e" in model:
                context = model+": "+ eval_data[model][i]["context"]
            #print("true_respone:", true_response)
        for stoken in SPECIAL_TOKEN_SET:
            #context = context.replace(stoken, "")    
            twf = twf.replace(stoken, "")   
            true = true.replace(stoken, "")       
        print("Context:")
        print(context)
        print("-"*20)
        print("GT Workflow action:", twf)
        print("true response:", true)
        for model in models:
            response = eval_data[model][i]["response"]
            for stoken in SPECIAL_TOKEN_SET:
                response = response.replace(stoken, "")
            print(f"{model} response:", response)
        input()
            #pairs += [dic]

    #context_response_pairs = pd.DataFrame(pairs)

if __name__ == "__main__":
    fire.Fire(inspect)