#!/usr/bin/env python3

import os
import torch
from typing import Union
import torch.multiprocessing as multiprocessing
from model.HRL.option_ppo import PPO, OptionPPO
from model.HRL.MHA_option_ppo import MHAOptionPPO
from model.HRL.hierarchical_ppo import HierarchicalPPO, MHAOptionPolicy
from model.HRL.option_policy import OptionPolicy, Policy
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
    env_type = config.env_type
    from envir.risk_factor_env_hrl import Risk_Factor_Env as Env
    env_name = config.env_name
    n_sample = config.n_sample
    n_epoch = config.n_epoch
    seed = config.seed
    set_seed(seed)

    log_dir, save_dir, _, _, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)

    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    dim_s, dim_c, dim_a = env.state_option_action_size()
    assert config.dim_c == dim_c
    print("The dimension info of the environment: ", dim_s, dim_c, dim_a)

    if config.algo == 'dac':
        policy = MHAOptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
        ppo = MHAOptionPPO(config, policy, dim_s)
    elif config.algo == 'hier_ppo':
        policy = MHAOptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
        ppo = HierarchicalPPO(config, policy, dim_s)
    else:
        raise ValueError("Invalid algorithm. Please choose 'dac' or 'hier_ppo'.")
        

    sampling_agent = Sampler(seed, env, policy, is_expert=False)

    for i in range(n_epoch):
        sample_sxar, sample_r = sample_batch(policy, sampling_agent, n_sample)
        lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
        ppo.step(sample_sxar, lr_mult=lr_mult)
        if i % 50 == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, policy) # testing performance
            torch.save(policy.state_dict(), save_name_f(i))
            logger.log_test_info(info_dict, i)
        print(f"{i}: r-sample-avg={sample_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i) # a very important metric
        logger.flush()


if __name__ == "__main__":
    # learn the expert policy/option-policy based on the environment rewards using PPO
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "risk_factor", "Environment type, can be [risk_factor, ...]")
    arg.add_arg("env_name", "sp500", "Environment name, can be [sh300, sp500]")
    arg.add_arg("algo", "hier_ppo", "Environment type, can be [ppo, dac, hier_ppo]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_epoch", 2000, "Number of training epochs")
    arg.add_arg("seed", 0, "Random seed")
    arg.parser()

    assert arg.env_type == "risk_factor"
    config = risk_factor_config
    config.update(arg)

    config.use_option = True
    config.train_option = True
    if config.algo == 'ppo':
        config.use_option = False
        config.train_option = False

    print(f">>>> Training {config.algo} on {config.env_name} environment, on {config.device}")
    learn(config, msg=config.tag)