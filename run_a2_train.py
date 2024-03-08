import json
import numpy as np
from tqdm import tqdm
import fire
from transformers import set_seed
"""
TODO: 
- decide the approaches to implement
    -   Filter & Finetune
        Decision Transformers: https://arxiv.org/pdf/2106.01345.pdf
        ILQL: https://arxiv.org/pdf/2206.11871.pdf
        Quark: https://arxiv.org/pdf/2205.13636.pdf
        Training with negatives: CRINGE loss: https://arxiv.org/pdf/2211.05826.pdf

- training loop
- testing loop: at a different script
""" 
from argparse import Namespace
from Quark.main import *
def quark(output_dir='outputs',
    dataset='data/wc_seed_one_convo.json',
    dataset_type='a2', 
    reward_model_path = "./save/block_evaluator_scorer/230809/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/",
    # reward_model_path = "./save/context_block_evaluator_scorer/230817/random/evaluator-roberta-base-tf-lr2e-5-bs64-epoch1-ws0-gas1-1gpu/",
    kl_coef=0.05, adaptive_kl=False, target_kl=3, entropy_coef=0.06, adaptive_entropy=False,target_entropy=40,
    #init_model='./save/dist_st/230626/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/', # fp16 weight
    #ref_model='./save/dist_st/230626/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/',  # fp16 weight
    init_model="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",
    ref_model="./save/dist_st/230809/distilgpt2/utt_prediction-distilgpt2-tf-lr5e-4-bs16-epoch10-ws0-gas1-1gpu/",
    eval_model=None, #"distilgpt2", #None,
    #init_model="gpt2",
    #ref_model="gpt2",
    bert_filter = False,    
    oracle = True, 
    only_top = False, train_prompt_len=2000, val_prompt_len=2000,
    n_extra_tokens=5, horizon=2500,  # need to make sure that there is enough sample in the sample dataset otherwise it repeats (learning on same data)
    top_p=1.0, seed=1, xsample_interval=500, log_interval=100, save_interval=2000, eval_interval=2000, # sample_interval = train_prompt_len // batch_size
    response_length=64, temperature=1.0, total_episodes=160000, batch_size=16,
    lr=5e-4, num_warmup_steps=500, clip_grad=True, max_grad_norm=1.0, #num_samples=25, # what does num_samples do?
    cuda_deterministic=True, cuda=True,
    reward_mode = "block",
    limit_none = True, 
    interactive = True, # use interactive generation with the ref policy
    action_end_enforce=True,
    repetition_penalty = False,
    context_reward = False,
    only_standard_data = True
    # reward_mode = "single"
    ):
    """
    TODO: implement when oracle = False
          Change the prompt collator
                 and change the split by special_tokens_set (to exclude workflow)
        - play with kl divergence coefficient
        - reward scaling injection
    """

    # setting the intervals size this way
    # sample_interval reason: need to make sure that the data is replenished every one epoch over data
    #         and 1 epoch in # steps is train_prompt_len // batch_size
    # other intervals for basically same logic, 1 epoch is a sensible time to do
    # NOTE / TODO: I use a different logic in step, sample, eval... 
    # Basically when sampler runs out depleted = True
    # So far I'm passing it to sample
    # but can also pass to eval and save
    #sample_interval = 2000#train_prompt_len // batch_size #2000
    #eval_interval = train_prompt_len // batch_size

    #save_interval = 20000#train_prompt_len // batch_size
    log_interval = train_prompt_len // batch_size

    args = Namespace(**locals())
    if args.reward_mode == "llm":
        eval_interval = 100000000

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    args.save_dir = os.path.join(args.output_dir, date_time)
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
        ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')



    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] + \
                  [' _TREE_TOKEN_ZERO_COMMENTS']

    log.info(f'Initializing models ...')
    ref_policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device, oracle=args.oracle, reward_mode=args.reward_mode, reference=True)
    #ref_policy = set_tokenizer_for_policy(ref_policy)
    
    if args.eval_model == None:
        eval_policy = None
    else:
        eval_policy = Policy(model_name=args.eval_model, temperature=args.temperature, device=device, oracle=args.oracle, eval_model=True)
        #eval_policy = set_tokenizer_for_policy(eval_policy)

    policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device,
                    reward_cond=True, tree_tokens=tree_tokens, oracle=args.oracle, reward_mode=args.reward_mode)
    #policy = set_tokenizer_for_policy(policy)
    
    if args.reward_mode == "llm":
        reward = LLMReward()
    elif args.reward_mode == "block":
        reward = BlockReward(reward_model_path=args.reward_model_path, batch_size=args.batch_size,\
         save_dir = args.reward_dir, bert_filter=args.bert_filter, repetition_penalty=args.repetition_penalty, \
         action_end_enforce= args.action_end_enforce, context_reward=context_reward)
    else:
        reward = Reward(reward_model_path=args.reward_model_path, batch_size=args.batch_size, save_dir = args.reward_dir, bert_filter=args.bert_filter)
    data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens, only_top = args.only_top)
    log.info(f'Initialization done!')

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
    
    train_dataset = PromptDataset(dataset_type=args.dataset_type, path=args.dataset, split="train", data_len=args.train_prompt_len,\
     oracle=args.oracle, reward_mode = args.reward_mode, limit_none=args.limit_none, only_standard_data=args.only_standard_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(dataset_type=args.dataset_type, path=args.dataset, split="dev", data_len=args.val_prompt_len,\
     oracle=args.oracle, reward_mode = args.reward_mode, limit_none=args.limit_none, only_standard_data=args.only_standard_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    # set up optimizer and scheduler
    #optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-5)
    optimizer = AdamW(policy.model.parameters(), lr=args.lr, eps=1e-5)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, eval_policy=eval_policy,
                               data_pool=data_pool,
                               score_model=reward, tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               optimizer=optimizer, scheduler=scheduler, oracle=args.oracle, reward_mode=args.reward_mode, interactive = args.interactive)

    for step_num in range(args.total_steps):
        try:
            trainer.step(step_num)
        except RuntimeError:
            torch.cuda.empty_cache()
            continue

    policy.model.save_pretrained(args.model_dir)
    policy.tokenizer.save_pretrained(args.model_dir)
    return

def controller(method="quark", **kwargs):

    if method == "ff":
        quark(only_top=True, **kwargs)
    elif method == "quark":
        quark(**kwargs)
    elif method == "dt":
        pass
    elif method == "ilql":
        pass
        


if __name__ == "__main__":
    # should i implement ff as a special case of quark?
    # should i implement ff separately as dt
    # dt is (kinda) special case of quark
    """
    Methods (choose a few of them)
    - "quark": quantized reward token method
    - "ff": filter and finetune / most basic
    - "dt": decision transformer with reward token provided

    - "ilql": implicit q-learning

    - "unlikelihood": unlikelihood loss
    - "cringe": cringe loss

    Shared args
    - "reward_model_path": reward model path (as hf pretrained model)
    - args regarding quantization / ranking / filtering
    
    Method-specific args
    """
    fire.Fire(controller)


    """
    DEBUG
    - the output logits become nan.... why?
        ==> Current guess: could be fp16 warmup to nonfp16 training?
            ==> non fp16 model also resulted in nan, but trained without the bug for longer (40 steps vs 3xx steps...)
        ==> RL training instability: learning rate, gradient clipping...
        ==> May need to use a bigger model like gpt2 large
    The nan problem is alleviated only when
    The warm-start model is not fp16 trained
    Learning rate is very low (2e-10, compared to 1e-5 originally used for the gpt2-large model of the Quark task)


    TODOs
    2. turn the dataset into abcd, neg [semi v] 
        - need to change datapool
        - read directly from data/wc_seed_one_convo
        - prompts, responses, true_wfs
    3. substitute the reward model [semi v]

    4. make sure the policy model is compatible / comparable with distilgpt2 medium that i'm using
        - right now is gpt2-large ==> too large [semi v]
        - probably need to warmstart from supervised training model
    1. replace args with fire arguments [todo]
    """ 