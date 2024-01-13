#!/usr/bin/env python3
import os
import torch
from typing import Union
import torch.multiprocessing as multiprocessing
from model.HRL.hierarchical_ppo import HierarchicalPPO, MHAOptionPolicy

from sampler import Sampler
from utils.common_utils import lr_factor_func, get_dirs, reward_validate, set_seed
from utils.logger import Logger
from utils.config import Config, ARGConfig
from default_config import risk_factor_config

def sample_batch(policy, agent, n_step):
    sample = agent.collect(policy.state_dict(), n_step, fixed=False)
    rsum = sum([sxar[-1].sum().item() for sxar in sample]) / len(sample)
    return sample, rsum

def learn(config: Config, msg="default"):
    # load the configuration
    env_type = config.env_type
    from envir.risk_factor_env_hrl import Risk_Factor_Env as Env
    env_name = config.env_name
    n_sample = config.n_sample
    n_epoch = config.n_epoch
    seed = config.seed
    set_seed(seed)

    # set up the save and log directory
    log_dir, save_dir, _, _, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    # register the ecnironment
    env = Env(env_name)
    dim_s, dim_c, dim_a = env.state_option_action_size()
    assert config.dim_c == dim_c
    print("The dimension info of the environment: ", dim_s, dim_c, dim_a) 

    # register the policy and algorithm
    policy = MHAOptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
    ppo = HierarchicalPPO(config, policy, dim_s)

    # load the pretrained model
    model_dict = policy.state_dict()
    if os.path.exists(config.policy_model):
        pretrained_model = torch.load(config.policy_model, map_location=config.device)
        pretrained_dict = pretrained_model
        new_dict = {}
        for k, v in model_dict.items():
            if ('embed_option' in k) or ('act_doe' in k):
                for pk, pv in pretrained_dict.items():
                    if k in pk:
                        print("Essential: ", k, pk)
                        new_dict[k] = pv
        model_dict.update(new_dict)
    else:
        print("Please check config.policy_model.")
    policy.load_state_dict(model_dict)

    # sample and train
    sampling_agent = Sampler(seed, env, policy, is_expert=False)
    for i in range(n_epoch):
        sample_sxar, sample_r = sample_batch(policy, sampling_agent, n_sample)
        lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
        # if sample_r > 1150 and i > 50:
        #     lr_mult = 0.01
        ppo.step(sample_sxar, lr_mult=lr_mult, train_policy=config.train_low_policy)
        if i % 50 == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, policy) # testing performance
            torch.save(policy.state_dict(), save_name_f(i))
            logger.log_test_info(info_dict, i)
        print(f"{i}: r-sample-avg={sample_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i) # a very important metric
        logger.flush()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    arg = ARGConfig()
    arg.add_arg("env_type", "risk_factor", "Environment type, can be [risk_factor, ...]")
    arg.add_arg("env_name", "sh300", "Environment name, can be [sh300, sp500]") #???
    arg.add_arg("algo", "hier_ppo_tran", "Algorithm type, can be [hier_ppo_tran]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_epoch", 2000, "Number of training epochs")
    arg.add_arg("seed", 0, "Random seed")
    arg.add_arg("train_low_policy", True, "whether to train the low level policy (i.e., skills)")
    arg.parser()
    assert arg.env_type == "risk_factor"
    config = risk_factor_config
    config.update(arg)

    config.use_option = True
    config.train_option = True
    '''sh300'''
    config.policy_model = "./option_model/100.torch" 
    #config.critic_model = "./option_model/100_critic.torch"
    '''sp500'''
    #config.policy_model = "./option_model/50.torch" 
    #config.critic_model = "./option_model/50_critic.torch"
    print(f">>>> Training {config.algo} on {config.env_name} environment, on {config.device}")
    learn(config, msg=config.tag)