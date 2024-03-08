import json

from nltk import sent_tokenize

train_path="./data/generated_negatives_train_10000_with_none_with_random.json"
val_path="./data/generated_negatives_dev_1000_with_none_with_random.json"

K = 3

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"


if False:
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    paraphraser = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


    def paraphrase(
        question,
        num_beams=K,
        num_beam_groups=K,
        num_return_sequences=K,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    ):
        input_ids = tokenizer(
            f'paraphrase: {question}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        
        outputs = paraphraser.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res

else:
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = device #s'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def paraphrase(input_texts,num_return_sequences=K,num_beams=K):
        batch = tokenizer(input_texts,truncation=True,padding='longest',max_length=256, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=128,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text



with open(train_path, "r") as fh:
    train = json.load(fh)#[:10] # remove [:10]

with open(val_path, "r") as fh:
    val = json.load(fh)#[:10]

temps = []
for split in [train, val]:
    temp = []
    for d in tqdm(split[1:]):
        dialog = d["dialog"]
        # if dialog.strip() == "":
        #     continue
        pos_act = d["original_action"]
        neg_act = d["negative_action"]
        random_act = d["random_action"]
        pos_response = d["pos_response"].lower()
        neg_response = d["generated"].lower()
        #neg_response = neg_response.lower(#sent_tokenize(neg_response)[0].lower()

        random_response = d["random_response"].lower()

        pos_paraphrases = paraphrase(pos_response)
        neg_paraphrases = paraphrase(neg_response)
        rand_paraphrases = paraphrase(random_response)

        pos_paraphrases = [x.lower() for x in pos_paraphrases]
        neg_paraphrases = [x.lower() for x in neg_paraphrases]
        rand_paraphrases =[ x.lower() for x in rand_paraphrases ]

        # pos_paraphrases = list(set(pos_paraphrases))
        # neg_paraphrases = list(set(neg_paraphrases))

        if False:
            print("="*30)
            print("pos:", pos_response)
            print(pos_paraphrases)
            print()
            print("neg:", neg_response)
            print(neg_paraphrases)
            print()
            print("rand:", random_response)
            print(rand_paraphrases)
            print()
            input()

        dic = { "original_action":pos_act,
                "negative_action": neg_act,
                "random_action": random_act,
                "pos_response": pos_response,
                "neg_response": neg_response,
                "random_response": random_response,
                "pos_paraphrases": pos_paraphrases,
                "neg_paraphrases": neg_paraphrases,
                "rand_paraphrases": rand_paraphrases,
                "dialog": dialog
        }
        temp.append(dic)

    temps.append(temp)

with open(f"./data/gen_neg_para_train_{K}_10000.json", "w") as fh:
    json.dump(temps[0], fh, indent=4)

with open(f"./data/gen_neg_para_dev_{K}_1000.json", "w") as fh:
    json.dump(temps[1], fh, indent=4)
