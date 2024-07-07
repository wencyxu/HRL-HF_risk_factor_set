import torch
from .option_critic import OptionCritic, Critic
from .option_policy import OptionPolicy, Policy
from utils.config import Config
import datetime

class PPO(object):
    def __init__(self, config: Config, policy: Policy):
        self.policy = policy
        self.clip_eps = config.clip_eps
        self.lr = config.optimizer_lr_policy
        self.gamma = config.gamma
        self.gae_tau = config.gae_tau
        self.use_gae = config.use_gae
        self.mini_bs = config.mini_batch_size
        self.lambda_entropy = config.lambda_entropy_policy # they set this as 0, which is not so appropriate

        self.critic = Critic(config, self.policy.dim_s)

    def _calc_adv(self, sample_sar):
        with torch.no_grad():
            s_array = []
            a_array = []
            ret_array = []
            adv_array = []
            vel_array = []
            for s, a, r in sample_sar:
                v = self.critic.get_value(s).detach()
                advantages = torch.zeros_like(v)
                returns = torch.zeros_like(v)
                next_value = 0.
                adv = 0.
                ret = 0.

                for i in reversed(range(r.size(0))):
                    ret = r[i] + self.gamma * ret
                    returns[i] = ret

                    if not self.use_gae:
                        advantages[i] = ret - v[i]
                    else:
                        delta = r[i] + self.gamma * next_value - v[i]
                        adv = delta + self.gamma * self.gae_tau * adv
                        advantages[i] = adv
                        next_value = v[i]

                s_array.append(s)
                a_array.append(a)
                ret_array.append(returns)
                adv_array.append(advantages)
                vel_array.append(v)
            s_array = torch.cat(s_array, dim=0)
            a_array = torch.cat(a_array, dim=0)
            ret_array = torch.cat(ret_array, dim=0)
            adv_array = torch.cat(adv_array, dim=0)
            vel_array = torch.cat(vel_array, dim=0)
        return s_array, a_array, ret_array, adv_array, vel_array

    def step(self, sample_sar, lr_mult=1.):
        # policy_remap => params, log_pi, log_prob_entropy
        # sample_r => N x [s, a, r], s = T x dim_s, a = T x dim_a, r = T x 1, tensor

        optim = torch.optim.Adam(self.critic.get_param() + self.policy.get_param(),
                                 lr=self.lr * lr_mult, weight_decay=1.e-3, eps=1e-5)

        with torch.no_grad():
            states, actions, returns, advantages, vel_array = self._calc_adv(sample_sar)
            fixed_log_probs = self.policy.log_prob_action(states, actions).detach() # old_pi

        for _ in range(10):
            inds = torch.randperm(states.size(0))

            for ind_b in inds.split(self.mini_bs):
                state_b, action_b, return_b, advantages_b, fixed_log_b, fixed_v_b = \
                    states[ind_b], actions[ind_b], returns[ind_b], advantages[ind_b], fixed_log_probs[ind_b], vel_array[ind_b]

                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8) if ind_b.size(0) > 1 else 0.

                logp, entropy = self.policy.policy_log_prob_entropy(state_b, action_b)
                vpred = self.critic.get_value(state_b)

                vpred_clip = fixed_v_b + (vpred - fixed_v_b).clamp(-self.clip_eps, self.clip_eps)
                vf_loss = torch.max((vpred - return_b).square(), (vpred_clip - return_b).square()).mean() # TODO: have not seen this before

                ratio = (logp - fixed_log_b).clamp_max(15.).exp()
                pg_loss = -torch.min(advantages_b * ratio, advantages_b * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
                loss = pg_loss + vf_loss * 0.5 - self.lambda_entropy * entropy.mean()
                optim.zero_grad()
                loss.backward()
                # after many experiments i find that do not clamp performs the best
                # torch.nn.utils.clip_grad_norm_(self.policy.get_param(), 0.5)
                optim.step()


class OptionPPO(torch.nn.Module):
    def __init__(self, config: Config , policy: OptionPolicy):
        super(OptionPPO, self).__init__()
        self.train_policy = config.train_policy
        self.train_option = config.train_option
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

        self.critic_lo = OptionCritic(config, dim_s=self.policy.dim_s, dim_c=self.policy.dim_c) # in DAC, only one Critic is required.

    def _calc_adv(self, sample_scar, train_policy=True, train_option=True):
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
                vc = self.critic_lo.get_value(s)    # N x dim_c
                if train_option:
                    pc = self.policy.log_trans(s, c[:-1]).exp()  # N x dim_c
                    vh = (vc * pc).sum(dim=-1, keepdim=True).detach() # this is why only one critic is required # (N, 1)
                else:
                    vh = torch.zeros_like(r)
                vl = vc.gather(dim=-1, index=c[1:]).detach() if train_policy else torch.zeros_like(r) # (N, 1)

                advantages_hi = torch.zeros_like(r)
                advantages_lo = torch.zeros_like(r)
                returns = torch.zeros_like(r)
                next_value_hi = 0.
                next_value_lo = 0.
                adv_hi = 0.
                adv_lo = 0.
                ret = 0.

                for i in reversed(range(r.size(0))):
                    ret = r[i] + self.gamma * ret # share the same return estimated through MC
                    returns[i] = ret

                    if not self.use_gae:
                        advantages_hi[i] = ret - vh[i]
                        advantages_lo[i] = ret - vl[i]
                    else:
                        delta_hi = r[i] + self.gamma * next_value_hi - vh[i]
                        delta_lo = r[i] + self.gamma * next_value_lo - vl[i]
                        adv_hi = delta_hi + self.gamma * self.gae_tau * adv_hi
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

    def _step_elem(self, sample_scar, lr_mult=1.0, train_policy=True, train_option=True):
        # two normal PPO processes for the high- and low-level policies, sharing the low critic, nothing new.
        # policy_remap => params, log_pi, log_prob_entropy
        # sample_scar => N x [s, c, a, r], s = T x dim_s, c = T+1 x 1, a = T x dim_a, r = T x 1, tensor
        optim_hi = torch.optim.Adam(self.critic_lo.get_param() + self.policy.get_param(low_policy=False),
                                    lr=self.lr_option * lr_mult, weight_decay=1.e-3, eps=1e-5)
        optim_lo = torch.optim.Adam(self.critic_lo.get_param() + self.policy.get_param(low_policy=True), # only the critic for the low-level policy is required
                                    lr=self.lr_policy * lr_mult, weight_decay=1.e-3, eps=1e-5)

        print("Calculating the advantages......")
        time_s = datetime.datetime.now()
        with torch.no_grad():
            states, options, options_1, actions, returns, advantages_hi, advantages_lo, vel_hi_array, vel_lo_array = \
                self._calc_adv(sample_scar, train_policy=train_policy, train_option=train_option)
            fixed_log_p_hi = self.policy.log_prob_option(states, options_1, options).detach() if train_option else torch.zeros_like(advantages_hi)
            fixed_log_p_lo = self.policy.log_prob_action(states, options, actions).detach() if train_policy else torch.zeros_like(advantages_lo)
            fixed_pc = self.policy.log_trans(states, options_1).exp().detach() if train_option else torch.zeros_like(advantages_lo)
        time_e = datetime.datetime.now()
        print("Time cost: ", (time_e - time_s).seconds)

        print("Updating with PPO......")
        time_s = datetime.datetime.now()
        for _ in range(10):
            inds = torch.randperm(states.size(0))

            for ind_b in inds.split(self.mini_bs):
                s_b, c_b, c_1b, a_b, ret_b, adv_hi_b, adv_lo_b, fixed_log_hi_b, fixed_log_lo_b, fixed_pc_b, fixed_vh_b, fixed_vl_b = \
                    states[ind_b], options[ind_b], options_1[ind_b], actions[ind_b], returns[ind_b], advantages_hi[ind_b],\
                    advantages_lo[ind_b], fixed_log_p_hi[ind_b], fixed_log_p_lo[ind_b], fixed_pc[ind_b], vel_hi_array[ind_b], vel_lo_array[ind_b]

                if train_option:
                    adv_hi_b = (adv_hi_b - adv_hi_b.mean()) / (adv_hi_b.std() + 1e-8) if ind_b.size(0) > 1 else 0.
                    logp, entropy = self.policy.option_log_prob_entropy(s_b, c_1b, c_b)
                    vpred = (self.critic_lo.get_value(s_b) * fixed_pc_b).sum(dim=-1, keepdim=True) # only one critic is available

                    vpred_clip = fixed_vh_b + (vpred - fixed_vh_b).clamp(-self.clip_eps, self.clip_eps)
                    vf_loss = torch.max((vpred - ret_b).square(), (vpred_clip - ret_b).square()).mean()

                    ratio = (logp - fixed_log_hi_b).clamp_max(15.).exp()
                    pg_loss = -torch.min(adv_hi_b * ratio, adv_hi_b * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
                    loss = pg_loss + vf_loss * 0.5 - self.lambda_entropy_option * entropy.mean() # TODO: adjust the weight 0.5
                    optim_hi.zero_grad()
                    loss.backward()
                    # after many experiments i find that do not clamp performs the best
                    # torch.nn.utils.clip_grad_norm_(self.policy.get_param(low_policy=not is_option), 0.5)
                    optim_hi.step()

                if train_policy:
                    adv_lo_b = (adv_lo_b - adv_lo_b.mean()) / (adv_lo_b.std() + 1e-8) if ind_b.size(0) > 1 else 0.
                    logp, entropy = self.policy.policy_log_prob_entropy(s_b, c_b, a_b)
                    vpred = self.critic_lo.get_value(s_b, c_b)

                    vpred_clip = fixed_vl_b + (vpred - fixed_vl_b).clamp(-self.clip_eps, self.clip_eps)
                    vf_loss = torch.max((vpred - ret_b).square(), (vpred_clip - ret_b).square()).mean()

                    ratio = (logp - fixed_log_lo_b).clamp_max(15.).exp()
                    pg_loss = -torch.min(adv_lo_b * ratio, adv_lo_b * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
                    loss = pg_loss + vf_loss * 0.5 - self.lambda_entropy_policy * entropy.mean()
                    optim_lo.zero_grad()
                    loss.backward()
                    # after many experiments i find that do not clamp performs the best
                    # torch.nn.utils.clip_grad_norm_(self.policy.get_param(low_policy=not is_option), 0.5)
                    optim_lo.step()
        time_e = datetime.datetime.now()
        print("Time cost: ", (time_e - time_s).seconds)

    def step(self, sample_scar, lr_mult=1.0):
        self._step_elem(sample_scar, lr_mult=lr_mult, train_policy=self.train_policy, train_option=self.train_option)




