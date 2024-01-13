import math
import torch
from torch import nn
from typing import Type
import numpy as np


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == np.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)

    # print("max_norm=%s, norm_type=%s, total_norm=%s, clip_coef=%s" % (max_norm, norm_type, total_norm, clip_coef))


def init_layer(module, gain=math.sqrt(2)):
    with torch.no_grad():
        nn.init.orthogonal_(module.weight.data, gain=gain)
        nn.init.constant_(module.bias.data, 0)
    return module

def init_weights_zeros(weights):
    # init weights with zeros value
    return nn.init.zeros_(weights)


def init_weights_normal(weights, mean=0.0, std=1.0):
    # init weights with normal distribution
    return nn.init.normal_(weights, mean=mean, std=std)


def make_module(in_size, out_size, hidden, activation: Type[nn.Module] = nn.ReLU):
    n_in = in_size
    l_hidden = []
    for h in hidden:
        l_hidden.append(init_layer(torch.nn.Linear(n_in, h)))
        l_hidden.append(activation())
        n_in = h
    l_hidden.append(init_layer(torch.nn.Linear(n_in, out_size), gain=0.1))
    return torch.nn.Sequential(*l_hidden)


def make_module_list(in_size, out_size, hidden, n_net, activation: Type[nn.Module] = nn.ReLU):
    return nn.ModuleList([make_module(in_size, out_size, hidden, activation) for _ in range(n_net)])


def make_activation(act_name):
    return (torch.nn.ReLU if act_name == "relu" else
            torch.nn.Tanh if act_name == "tanh" else
            torch.nn.Sigmoid if act_name == "sigmoid" else
            torch.nn.Softplus if act_name == "softplus" else None)


