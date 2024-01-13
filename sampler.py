import random
import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.HRL.option_policy import OptionPolicy
from model.HRL.MHA_option_policy_critic import MHAOptionPolicy
from utils.common_utils import set_seed


__all__ = ["Sampler"]
# loop作用：从一个给定的策略中采样一条轨迹数据，返回一个包含状态、动作、奖励的元组
# policy：指导动作选择；is_expert：是否使用专家策略；fixed：是否固定选项的选择；task_list：要执行的任务
def option_loop(env, policy, is_expert, fixed):
    # 使用torch.no_grad()上下文管理器：不需要计算梯度，节省内存
    with torch.no_grad():
        # 创建四个空列表：存储动作、选项、状态、奖励的数组
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        # 重置环境，并获取初始状态和终止标志
        s, done = env.reset(), False
        # 初始化当前的选项
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        # 存储初始的选项
        c_array.append(ct)
        # 使用while循环，当done为False时继续执行
        while not done:
            # 将当前状态转换为策略的输入格式。
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            # 根据当前状态和选项，采样一个新选项并切断梯度传播
            ct = policy.sample_option(st, ct, fixed=fixed).detach()
            at = policy.sample_action(st, ct, fixed=fixed).detach()
            # 存储当前状态
            s_array.append(st)
            c_array.append(ct)
            a_array.append(at)
            # 执行动作，获取环境的反馈
            s, r, done = env.step(ct.cpu().squeeze(dim=0).numpy()[0], at.cpu().squeeze(dim=0).numpy()[0])
            # 存储当前奖励
            r_array.append(r)
        # 动作的列表转换为一个张量
        a_array = torch.cat(a_array, dim=0)
        c_array = torch.cat(c_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        # 将奖励的列表转换为一个张量
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



# _SamplerCommon：一个采样器的基类，用于从环境中收集轨迹数据
class _SamplerCommon(object):

    def __init__(self, seed, policy):
        self.device = policy.device
    # 根据策略参数、采样数量和是否固定选项或动作的标志来收集轨迹数据
    def collect(self, policy_param, n_sample, fixed=False):
        # # 抛出异常，表示这个方法需要在子类中实现
        raise NotImplementedError()
    # 没有使用任何过滤器
    def filter_demo(self, sa_array):
        print("No filters are adopted.")
        # 返回原始的状态和动作数组
        return sa_array

# 实现特定的采样器逻辑
class _SamplerSS(_SamplerCommon):
    def __init__(self, seed, env, policy, loop_func, is_expert=False):
        super(_SamplerSS, self).__init__(seed, policy)
        # 将环境对象的深拷贝赋值给self.env，并调用它的init方法，设置display为False，表示不显示环境的图形界面
        self.env = deepcopy(env)
        self.policy = deepcopy(policy)
        # 采样一条轨迹的函数
        self.loop_func = loop_func
        self.is_expert = is_expert

    # 根据策略参数、采样数量和是否固定选项或动作的标志来收集轨迹数据，覆盖父类的同名方法
    def collect(self, policy_param, n_sample, fixed=False):
        self.policy.load_state_dict(policy_param)
        # 还需要采样的轨迹数
        counter = n_sample
        # 存储采样的轨迹
        rets = []
        # 是否需要继续采样
        if counter > 0:
            # 使用while循环，直到counter小于等于0为止
            while counter > 0:
                # 调用self.loop_func方法，传入self.env、self.policy、self.is_expert和fixed四个参数，
                # 返回一条轨迹traj四元组：状态序列、选项序列、动作序列和奖励序列
                traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed)
                # 将轨迹traj添加到rets列表中
                rets.append(traj)
                # 将counter减去轨迹的状态数，表示已经采样了一条轨迹
                counter -= traj[0].size(0)
        else:
            # assert self.task_list is not None
            # 使用while循环，直到counter大于等于0为止。
            while counter < 0: # only used for testing, so don't need repeated sampling
                traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed)
                rets.append(traj)
                counter += 1
        return rets

def Sampler(seed, env, policy, is_expert) -> _SamplerCommon:

    if isinstance(policy, OptionPolicy) or isinstance(policy, MHAOptionPolicy):
        # 采样一条轨迹的函数
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