from utils.config import Config

default_config = Config({
    # global program config
    "n_sample": 4096,
    "n_epoch": 1000,  # 4000
    "n_pretrain_epoch": 0,
    "pretrain_log_interval": 250, #1000
    
    # global policy config
    "activation": "relu",
    "hidden_policy": (512, 512), # 256, 256
    "shared_policy": True, # consider setting this as true, if the number of options is too high
    "log_clamp_policy": (-20., 0.),
    "optimizer_lr_policy": 3.e-3, # 3.e-4

    "dim_c": 6, # number of options, should be set as 6
    "hidden_option": (256, 256),  # 64, 64
    "optimizer_lr_option": 3.e-3, # 3.e-4

    # ppo config
    "hidden_critic": (256, 256),    # 64, 64
    "shared_critic": True, # consider setting this as true, if the number of options is too high
    "train_policy": True,
    "optimizer_lr_critic": 3.e-3, # 3.e-4

    "use_gae": True,
    "gamma": 0.99,
    "gae_tau": 0.99,#0.95
    "clip_eps": 0.2,
    "mini_batch_size": 1024, # 256
    "lambda_entropy_policy": 0., # TODO: fine-tune
    "lambda_entropy_option": 1.e-5, # TODO: probably the most important parameter 1.e-4

    "log_interval": 5,

    # MHA-related
    "dmodel": 40, # dimension of the embedding
    "mha_nhead": 1, # number of attention head
    "mha_nlayers": 1,
    "mha_nhid": 50,
    "dropout": 0.2,
    "use_MHA_critic": False, # we suggest use false here, so that the algorithm for learning the hier policy would be DAC
    "use_MHA_policy": True,
    "use_hppo": True, # we suggest use false here
 })

risk_factor_config = default_config.copy()
