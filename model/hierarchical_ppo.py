import copy
import torch
from typing import Union
from .MHA_option_policy_critic import MHAOptionPolicy
from .option_policy import OptionPolicy
from .option_critic import OptionCritic
from utils.config import Config

class HierarchicalPPO(torch.nn.Module):
    def __init__(self, config: Config , policy: Union[MHAOptionPolicy, OptionPolicy], dim_s):
        super(HierarchicalPPO, self).__init__()
        self.gamma = config.gamma
        self.gae_tau = config.gae_tau
        self.use_gae = config.use_gae
        self.lr_policy = config.optimizer_lr_policy
        self.lr_option = config.optimizer_lr_option
        self.mini_bs = config.mini_batch_size
        self.clip_eps = config.clip_eps
        self.lambda_entropy_policy = config.lambda_entropy_policy
        self.lambda_entropy_option = config.lambda_entropy_option

        self.policy = policy
        self.critic_hi = OptionCritic(config, dim_s=dim_s, dim_c=self.policy.dim_c+1) # the last-step option has one more dim
        self.critic_lo = OptionCritic(config, dim_s=self.policy.dim_s, dim_c=self.policy.dim_c)  # in SA, only one Critic is required.

    def _calc_adv(self, sample_scar):
        with torch.no_grad():
            s_array = []
            c_array = []
            c_1array = []
            a_array = []
            ret_array = []
            adv_hi_array = []
            vel_hi_array = []
            adv_lo_array = []
            vel_lo_array = []
            for s, c, a, r in sample_scar:
                vc_l = self.critic_lo.get_value(s)    # N x dim_c
                vl = vc_l.gather(dim=-1, index=c[1:]).detach()  # (N, 1)

                vc_h = self.critic_hi.get_value(s)
                vh = vc_h.gather(dim=-1, index=c[:-1]).detach() # (N, 1)

                advantages_hi = torch.zeros_like(r)
                advantages_lo = torch.zeros_like(r)
                returns = torch.zeros_like(r)
                next_value_hi = 0.
                next_value_lo = 0.
                adv_hi = 0.
                adv_lo = 0.
                ret = 0.

                for i in reversed(range(r.size(0))):
                    ret = r[i] + self.gamma * ret # share the same return estimated through MC,
                    # the last step return is always 0 (masked), rather than the output of the value function
                    # since the "done" is always true for the last timestep based on the collection process of data
                    returns[i] = ret

                    if not self.use_gae:
                        advantages_hi[i] = ret - vh[i]
                        advantages_lo[i] = ret - vl[i]
                    else:
                        delta_hi = r[i] + self.gamma * next_value_hi - vh[i] # for the same reason as above, the init of next_values are set as 0
                        delta_lo = r[i] + self.gamma * next_value_lo - vl[i]
                        adv_hi = delta_hi + self.gamma * self.gae_tau * adv_hi # for the same reason as above, the init of advs are set as 0
                        adv_lo = delta_lo + self.gamma * self.gae_tau * adv_lo
                        advantages_hi[i], advantages_lo[i] = adv_hi, adv_lo
                        next_value_hi, next_value_lo = vh[i], vl[i]

                s_array.append(s)
                c_array.append(c[1:])
                c_1array.append(c[:-1])
                a_array.append(a)
                ret_array.append(returns)
                adv_hi_array.append(advantages_hi)
                adv_lo_array.append(advantages_lo)
                vel_hi_array.append(vh)
                vel_lo_array.append(vl)
            s_array = torch.cat(s_array, dim=0)
            c_array = torch.cat(c_array, dim=0)
            c_1array = torch.cat(c_1array, dim=0)
            a_array = torch.cat(a_array, dim=0)
            ret_array = torch.cat(ret_array, dim=0)
            adv_hi_array = torch.cat(adv_hi_array, dim=0)
            adv_lo_array = torch.cat(adv_lo_array, dim=0)
            vel_hi_array = torch.cat(vel_hi_array, dim=0)
            vel_lo_array = torch.cat(vel_lo_array, dim=0)
        return s_array, c_array, c_1array, a_array, ret_array, adv_hi_array, adv_lo_array, vel_hi_array, vel_lo_array

    def step(self, sample_scar, lr_mult=1.0, train_policy=True):
        # two normal PPO processes for the high- and low-level policies, sharing the low critic, nothing new.
        # sample_scar => N x [s, c, a, r], s = T x dim_s, c = T+1 x 1, a = T x dim_a, r = T x 1, tensor
        # for name, para in self.policy.named_parameters():
        #     print(name)

        optim_hi = torch.optim.Adam(self.critic_hi.get_param() + self.policy.get_param(low_policy=False),
                                    lr=self.lr_option * lr_mult, weight_decay=1.e-3, eps=1e-5)
        optim_lo = torch.optim.Adam(self.critic_lo.get_param() + self.policy.get_param(low_policy=True),
                                    # only the critic for the low-level policy is required
                                    lr=self.lr_policy * lr_mult, weight_decay=1.e-3, eps=1e-5)

        # optim = torch.optim.Adam(self.critic_lo.get_param() + self.critic_hi.get_param()
        #                          + list(self.policy.parameters()), lr=self.lr_option * lr_mult, weight_decay=1.e-3, eps=1e-5)
        
        # 计算优势函数和固定的对数概率时不需要计算梯度
        with torch.no_grad():
            states, options, options_1, actions, returns, advantages_hi, advantages_lo, vel_hi_array, vel_lo_array = \
                self._calc_adv(sample_scar) # TODO: time consuming
            # print("1: ", states.shape, options_1.shape, options.shape, actions.shape)
            fixed_log_p_hi = self.policy.log_prob_option(states, options_1, options).detach()
            fixed_log_p_lo = self.policy.log_prob_action(states, options, actions).detach()
            fixed_pc = self.policy.log_trans(states, options_1).exp().detach()
        
        for _ in range(10):
            # 生成一个随机排列，用于打乱样本的顺序
            inds = torch.randperm(states.size(0))
            # 循环将样本分成小批量，并对每个小批量执行一步优化
            for ind_b in inds.split(self.mini_bs):
                s_b, c_b, c_1b, a_b, ret_b, adv_hi_b, adv_lo_b, fixed_log_hi_b, fixed_log_lo_b, fixed_pc_b, fixed_vh_b, fixed_vl_b = \
                    states[ind_b], options[ind_b], options_1[ind_b], actions[ind_b], returns[ind_b], advantages_hi[ind_b], \
                    advantages_lo[ind_b], fixed_log_p_hi[ind_b], fixed_log_p_lo[ind_b], fixed_pc[ind_b], vel_hi_array[ind_b], vel_lo_array[ind_b]

                # update the high-level policy
                # 标准化优势函数
                adv_hi_b = (adv_hi_b - adv_hi_b.mean()) / (adv_hi_b.std() + 1e-8) if ind_b.size(0) > 1 else 0.
                # 计算对数概率和熵
                logp, entropy = self.policy.option_log_prob_entropy(s_b, c_1b, c_b)
                # 使用高级策略的critic来预测值函数
                vpred = self.critic_hi.get_value(s_b, c_1b)
                vpred_clip = fixed_vh_b + (vpred - fixed_vh_b).clamp(-self.clip_eps, self.clip_eps) # TODO: too small clip_eps
                vf_loss = torch.max((vpred - ret_b).square(), (vpred_clip - ret_b).square()).mean() # necessary
                # 计算策略比率
                ratio = (logp - fixed_log_hi_b).clamp_max(15.).exp()
                # 计算损失函数：策略梯度损失、值函数损失和熵的线性组合
                pg_loss = -torch.min(adv_hi_b * ratio,
                                     adv_hi_b * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
                loss = pg_loss + vf_loss * 0.5 - self.lambda_entropy_option * entropy.mean()  # TODO: adjust the weight 0.5
                # TODO: try moving the entropy term to the return as shown in the paper rather than using it as an independent term here
                # 将优化器的梯度清零
                optim_hi.zero_grad()
                # 计算损失函数的梯度
                loss.backward()
                # after many experiments i find that do not clamp performs the best
                # torch.nn.utils.clip_grad_norm_(self.policy.get_param(low_policy=not is_option), 0.5)
                optim_hi.step()
                # 训练一个HRL的低级策略
                if train_policy:
                    # update the low-level policy
                    # 标准化低级策略的优势函数
                    adv_lo_b = (adv_lo_b - adv_lo_b.mean()) / (adv_lo_b.std() + 1e-8) if ind_b.size(0) > 1 else 0.
                    # 计算策略对动作的对数概率和熵。熵用于鼓励策略探索更多的动作
                    logp, entropy = self.policy.policy_log_prob_entropy(s_b, c_b, a_b)
                    # 获取低级策略的价值函数预测
                    vpred = self.critic_lo.get_value(s_b, c_b)
                    # 价值函数的裁剪，以防止预测的价值函数和目标价值函数之间的差距过大
                    vpred_clip = fixed_vl_b + (vpred - fixed_vl_b).clamp(-self.clip_eps, self.clip_eps)
                    # note that the high critic and low critic are sharing the same return function, as mentioned in the paper
                    vf_loss = torch.max((vpred - ret_b).square(), (vpred_clip - ret_b).square()).mean()
                    # 计算价值函数的损失
                    ratio = (logp - fixed_log_lo_b).clamp_max(15.).exp()
                    # 计算策略梯度的损失
                    pg_loss = -torch.min(adv_lo_b * ratio, adv_lo_b * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
                    loss = pg_loss + vf_loss * 0.5 - self.lambda_entropy_policy * entropy.mean()
                    optim_lo.zero_grad()
                    loss.backward()
                    optim_lo.step()