import math
import torch
import torch.nn.functional as F
from utils.model_util import make_module, make_module_list, make_activation
from utils.config import Config
from torch.distributions import kl_divergence

# this policy uses one-step option, the initial option is fixed as o=dim_c


class Policy(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(Policy, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.device = torch.device(config.device)

        activation = make_activation(config.activation)
        n_hidden_pi = config.hidden_policy

        self.policy = make_module(self.dim_s, self.dim_a, n_hidden_pi, activation)

        self.to(self.device)

    def a_mean_logstd(self, s):
        y = self.policy(s)
        return y

    def log_prob_action(self, s, a):
        logits = self.a_mean_logstd(s)
        log_pas = logits.log_softmax(dim=-1)  # (bs, dim_a)
        return log_pas.gather(dim=-1, index=a)

    def sample_action(self, s, fixed=False):
        logits = self.a_mean_logstd(s)
        log_pas = logits.log_softmax(dim=-1)  # (bs, dim_a)

        if fixed:
            return log_pas.argmax(dim=-1, keepdim=True)
        else:
            return F.gumbel_softmax(log_pas, hard=False).multinomial(1).long()

    def policy_log_prob_entropy(self, s, a):

        log_prob = self.log_prob_action(s, a)
        entropy = -(log_prob * log_prob.exp()).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def get_param(self, low_policy=True):
        if not low_policy:
            print("WARNING >>>> policy do not have high policy params, returning low policy params instead")
        return list(self.parameters())


class OptionPolicy(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(OptionPolicy, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dim_c = config.dim_c
        self.device = torch.device(config.device)
        self.log_clamp = config.log_clamp_policy
        self.is_shared = config.shared_policy
        activation = make_activation(config.activation)
        n_hidden_pi = config.hidden_policy
        n_hidden_opt = config.hidden_option
        self.use_vae = config.use_vae
        if self.use_vae:
            assert config.use_d_info_gail

        if self.is_shared: # TODO: add the option to the input
            # output prediction p(ct| st, ct-1) with shape (N x ct-1 x ct)
            self.option_policy = make_module(self.dim_s, (self.dim_c+1) * self.dim_c, n_hidden_opt, activation)
            self.policy = make_module(self.dim_s, self.dim_c * self.dim_a, n_hidden_pi, activation)

            self.a_log_std = torch.nn.Parameter(torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))
        else:
            self.policy = make_module_list(self.dim_s, self.dim_a, n_hidden_pi, self.dim_c, activation)
            self.a_log_std = torch.nn.ParameterList([
                torch.nn.Parameter(torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.)) for _ in range(self.dim_c)])
            # i-th model output prediction p(ct|st, ct-1=i)
            self.option_policy = make_module_list(self.dim_s, self.dim_c, n_hidden_opt, self.dim_c+1, activation)

        if self.use_vae:
            self.policy = make_module(self.dim_s + self.dim_c, self.dim_a, n_hidden_pi, activation)
            self.a_log_std = torch.nn.Parameter(torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))

        self.to(self.device)

    def a_mean_logstd(self, st, ct=None):
        # ct: None or long(N x 1)
        # ct: None for all c, return (N x dim_c x dim_a); else return (N x dim_a)
        # s: N x dim_s, c: N x 1, c should always < dim_c
        if not self.use_vae:
            if self.is_shared:
                mean = self.policy(st).view(-1, self.dim_c, self.dim_a)
                logstd = self.a_log_std.expand_as(mean[:, 0, :])
            else:
                mean = torch.stack([m(st) for m in self.policy], dim=-2) # (N x dim_c x dim_a)
                logstd = torch.stack([m.expand_as(mean[:, 0, :]) for m in self.a_log_std], dim=-2)
            if ct is not None:
                ind = ct.view(-1, 1, 1).expand(-1, 1, self.dim_a)
                mean = mean.gather(dim=-2, index=ind).squeeze(dim=-2)
                logstd = logstd.gather(dim=-2, index=ind).squeeze(dim=-2)
        else:
            assert ct is not None and ct.shape[-1] == self.dim_c
            mean = self.policy(torch.cat([st, ct], dim=-1))
            # print("1: ", mean.shape)
            logstd = self.a_log_std.expand_as(mean[:, :])

        return mean.clamp(-10, 10), logstd.clamp(self.log_clamp[0], self.log_clamp[1]) # TODO

    def switcher(self, s):
        if self.is_shared:
            return self.option_policy(s).view(-1, self.dim_c+1, self.dim_c)
        else:
            return torch.stack([m(s) for m in self.option_policy], dim=-2) # (N x ct_1 x ct)

    def get_param(self, low_policy=True):
        if low_policy:
            if self.is_shared:
                return list(self.policy.parameters()) + [self.a_log_std]
            else:
                if self.use_vae:
                    return list(self.policy.parameters()) + [self.a_log_std]
                else:
                    return list(self.policy.parameters()) + list(self.a_log_std.parameters())
        else:
            return list(self.option_policy.parameters())

    # ===================================================================== #

    def log_trans(self, st, ct_1=None): # this part should not contain the gumbel noise, since it's only used for sampling
        # ct_1: long(N x 1) or None
        # ct_1: None: direct output p(ct|st, ct_1): a (N x ct_1 x ct) array where ct is log-normalized
        unnormed_pcs = self.switcher(st)
        log_pcs = unnormed_pcs.log_softmax(dim=-1)

        if ct_1 is None:
            return log_pcs
        else:
            return log_pcs.gather(dim=-2, index=ct_1.view(-1, 1, 1).expand(-1, 1, self.dim_c)).squeeze(dim=-2)

    def log_prob_action(self, st, ct, at): # Gaussian Distribution, independence assumption on the dimensions of actions
        # if c is None, return (N x dim_c x 1), else return (N x 1)
        mean, logstd = self.a_mean_logstd(st, ct)
        if ct is None:
            at = at.view(-1, 1, self.dim_a)
        return (-((at - mean).square()) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

    def log_prob_option(self, st, ct_1, ct):
        log_tr = self.log_trans(st, ct_1)
        return log_tr.gather(dim=-1, index=ct)

    def sample_action(self, st, ct, fixed=False):
        action_mean, action_log_std = self.a_mean_logstd(st, ct)
        if fixed:
            action = action_mean
        else:
            eps = torch.empty_like(action_mean).normal_()
            action = action_mean + action_log_std.exp() * eps
        return action

    def sample_option(self, st, ct_1, fixed=False, tau=1.0):
        log_tr = self.log_trans(st, ct_1)
        if fixed:
            return log_tr.argmax(dim=-1, keepdim=True)
        else:
            # print(F.gumbel_softmax(log_tr, hard=False)) # (N, c_dim)
            return F.gumbel_softmax(log_tr, hard=False, tau=tau).multinomial(1).long() # (N, 1) it's a surprise that option-gail has implemented this

    def vae_forward(self, st, ct_1, at, temperature):
        # encoder
        log_tr = self.log_trans(st, ct_1)
        latent_v = F.gumbel_softmax(log_tr, hard=False, tau=temperature)
        posterior_dist = torch.distributions.Categorical(logits=log_tr) # this implementation from another author also supports that when getting log_tr we don't need the gumbel loss
        prior_dist = torch.distributions.Categorical(probs=torch.ones_like(log_tr) / self.dim_c)
        # print("2: ", kl_divergence(posterior_dist, prior_dist))
        kl = kl_divergence(posterior_dist, prior_dist).sum(-1)
        # decoder
        reconstruction_loss = self.log_prob_action(st, latent_v, at)

        return kl - reconstruction_loss # elbo loss


    def policy_entropy(self, st, ct):
        _, log_std = self.a_mean_logstd(st, ct)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
        return entropy.sum(dim=-1, keepdim=True)

    def option_entropy(self, st, ct_1):
        log_tr = self.log_trans(st, ct_1)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return entropy

    def policy_log_prob_entropy(self, st, ct, at):
        mean, logstd = self.a_mean_logstd(st, ct)
        log_prob = (-(at - mean).pow(2) / (2 * (logstd * 2).exp()) - logstd - 0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def option_log_prob_entropy(self, st, ct_1, ct):
        # c1 can be dim_c, c2 should always < dim_c
        log_tr = self.log_trans(st, ct_1)
        log_opt = log_tr.gather(dim=-1, index=ct)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return log_opt, entropy

    def viterbi_path(self, s_array, a_array):
        assert not self.use_vae
        # exactly follow the Eq. (8)-(10) in the paper
        with torch.no_grad():
            log_pis = self.log_prob_action(s_array, None, a_array).view(-1, 1, self.dim_c)  # demo_len x 1 x ct
            log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1+1) x ct
            log_prob = log_trs[:, :-1] + log_pis # demo_len x ct_1 x ct
            log_prob0 = log_trs[0, -1] + log_pis[0, 0] # (ct, )
            # forward
            max_path = torch.empty(s_array.size(0), self.dim_c, dtype=torch.long, device=self.device)
            accumulate_logp = log_prob0 # (ct, )
            max_path[0] = self.dim_c
            for i in range(1, s_array.size(0)):
                accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
            # backward
            c_array = torch.zeros(s_array.size(0)+1, 1, dtype=torch.long, device=self.device)
            log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
            for i in range(s_array.size(0), 0, -1):
                c_array[i-1] = max_path[i-1][c_array[i]]
        return c_array.detach(), log_prob_traj.detach()




