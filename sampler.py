import random
import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.HRL.option_policy import OptionPolicy
from model.HRL.MHA_option_policy_critic import MHAOptionPolicy
from utils.common_utils import set_seed


__all__ = ["Sampler"]
def option_loop(env, policy, is_expert, fixed):
    with torch.no_grad():
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        s, done = env.reset(), False
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        c_array.append(ct)
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            ct = policy.sample_option(st, ct, fixed=fixed).detach()
            at = policy.sample_action(st, ct, fixed=fixed).detach()
            s_array.append(st)
            c_array.append(ct)
            a_array.append(at)
            s, r, done = env.step(ct.cpu().squeeze(dim=0).numpy()[0], at.cpu().squeeze(dim=0).numpy()[0])
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        c_array = torch.cat(c_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)

        # print("1: ", s_array.shape, c_array.shape, a_array.shape, r_array.shape)
        # raise NotImplementedError

    return s_array, c_array, a_array, r_array

def loop(env, policy, is_expert, fixed):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []

        s, done = env.reset(), False
        # print("1: ", s)
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            at = policy.sample_action(st, fixed=fixed).detach()
            s_array.append(st)
            a_array.append(at)
            # print("1: ", at)
            s, r, done = env.step(at.cpu().squeeze(dim=0)[0].numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
        # print(r_array.shape)
    return s_array, a_array, r_array



# _SamplerCommon
class _SamplerCommon(object):

    def __init__(self, seed, policy):
        self.device = policy.device
    def collect(self, policy_param, n_sample, fixed=False):
        raise NotImplementedError()
    def filter_demo(self, sa_array):
        print("No filters are adopted.")
        return sa_array

class _SamplerSS(_SamplerCommon):
    def __init__(self, seed, env, policy, loop_func, is_expert=False):
        super(_SamplerSS, self).__init__(seed, policy)
        self.env = deepcopy(env)
        self.policy = deepcopy(policy)
        self.loop_func = loop_func
        self.is_expert = is_expert

    def collect(self, policy_param, n_sample, fixed=False):
        self.policy.load_state_dict(policy_param)
        counter = n_sample
        rets = []
        if counter > 0:
          
            while counter > 0:
                
                traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed)
               
                rets.append(traj)
               
                counter -= traj[0].size(0)
        else:
            # assert self.task_list is not None
           
            while counter < 0: # only used for testing, so don't need repeated sampling
                traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed)
                rets.append(traj)
                counter += 1
        return rets

def Sampler(seed, env, policy, is_expert) -> _SamplerCommon:

    if isinstance(policy, OptionPolicy) or isinstance(policy, MHAOptionPolicy):
      
        loop_func = option_loop
    else:
        loop_func = loop
        # raise NotImplementedError

    class_m = _SamplerSS
    return class_m(seed, env, policy, loop_func, is_expert)


# 判断当前文件是否是主模块，如果是，表示直接运行这个文件
if __name__ == "__main__":
    # 用于设置多进程的启动方式
    from torch.multiprocessing import set_start_method
    set_start_method("spawn")
