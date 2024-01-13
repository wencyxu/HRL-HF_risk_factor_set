import math
import torch
from torch import nn
from utils.MHA_model_util import layer_init, range_tensor, DoeDecoderFFN, SkillPolicy, DoeSingleTransActionNet, DoeCriticNet
from utils.config import Config
import torch.nn.functional as F

#??? since the action now is discrete, we cannot use Gaussian distribution anymore
class MHAOptionPolicy(nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(MHAOptionPolicy, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dmodel = config.dmodel
        self.dim_c = config.dim_c
        self.device = torch.device(config.device)

        # option embedding
        # 创建用于选项的嵌入层，并使用正交初始化
        self.embed_option = nn.Embedding(config.dim_c, self.dmodel) # embedding matrix and the kernel matrix of the attention module
        nn.init.orthogonal(self.embed_option.weight) # initialization

        ## option policy:
        # 创建前馈神经网络初始化选项策略
        self.init_option_policy = DoeDecoderFFN(dim_s, hidden_units=(64, self.dim_c))  # used for the first step when the previous option is not available

        self.de_state_lc = layer_init(nn.Linear(dim_s, self.dmodel))
        self.de_state_norm = nn.LayerNorm(self.dmodel)
        self.de_logtis_lc = layer_init(nn.Linear(2 * self.dmodel, self.dim_c))
        self.doe = SkillPolicy(self.dmodel, config.mha_nhead, config.mha_nlayers, config.mha_nhid, config.dropout)

        # policy
        # self.act_concat_norm = nn.LayerNorm(self.dim_s + self.dmodel)
        # 创建线性层、层归一化、多头注意力层
        self.act_doe = DoeSingleTransActionNet(self.dim_s + self.dmodel, self.dim_a, hidden_units=config.hidden_policy)

        self.to(self.device)
    # 计算动作的均值和对数标准差
    def a_mean_logstd(self, st, ct=None):
        # ct: None or long(N x 1)
        # ct: None for all c, return (N x dim_c x dim_a); else return (N x dim_a)
        # s: N x dim_s, c: N x 1, c should always < dim_c
        # 如果输入的状态维度大于预设的状态空间维度，只取前dim_s维的状态
        if st.shape[-1] > self.dim_s:
            stt = st[:, :self.dim_s]
            # print("1: ", stt.shape, self.dim_s)
        else:
            stt = st
        # 如果选项不为空，将选项嵌入并与状态拼接，然后通过动作网络得到输出
        if ct is not None:
            ct = ct.squeeze(-1) # (N, )
            ct_emb = self.embed_option(ct.unsqueeze(0)).detach().squeeze(0) # (N, dim_e)
            act_inp = torch.cat([stt, ct_emb], dim=-1)
            # act_inp = self.act_concat_norm(act_inp) # TODO
            logits = self.act_doe(act_inp)
        # 如果选项为空，那么将所有选项嵌入并与状态拼接，然后通过动作网络得到输出
        else:
            # embed_all_idx: [bs, dim_c]
            bs = stt.shape[0]
            embed_all_idx = range_tensor(self.dim_c, self.device).repeat(bs, 1) # (bs, dim_c)
            embed_all = self.embed_option(embed_all_idx).detach() # (bs, dim_c, dim_e)
            stt = stt.unsqueeze(1).repeat(1, self.dim_c, 1) # (bs, dim_c, dim_s)
            act_inp = torch.cat([stt, embed_all], dim=-1).view(-1, self.dim_s + self.dmodel)
            logits = self.act_doe(act_inp)
            logits = logits.view(bs, self.dim_c, self.dim_a)

        return logits
    # 计算选项切换的概率
    def switcher(self, s):
        # s: (N, dim_s)
        # output: (N x ct_1 x ct)
        # encoder inputs
        # 将所有选项嵌入
        bs = s.shape[0]
        # embed_all_idx: [dim_c, bs]
        embed_all_idx = range_tensor(self.dim_c, self.device).repeat(bs, 1).t()
        wt = self.embed_option(embed_all_idx) # (dim_c, bs, dim_e) # this is the attention kernel matrix

        # state input
        # 将状态通过线性层和激活函数，然后进行层归一化
        s_rep = s.unsqueeze(1).repeat(1, self.dim_c, 1).view(-1, self.dim_s) # (bs*dim_c, dim_s)
        s_hat = F.relu(self.de_state_lc(s_rep)) # (bs*dim_c, dim_e)
        s_hat = self.de_state_norm(s_hat) # (bs*dim_c, dim_e)
        # option input
        # 所有选项嵌入
        embed_all_idx = range_tensor(self.dim_c, self.device).repeat(bs, 1)  # (bs, dim_c)
        prev_options = embed_all_idx.view(-1, 1) # (bs*dim_c, 1)
        ct_1 = self.embed_option(prev_options.t()).detach()  # (1, bs*dim_c, dim_e)
        # concat
        # 将状态和选项拼接，然后通过多头注意力层和线性层得到输出
        opt_cat_1 = torch.cat([s_hat.unsqueeze(0), ct_1], dim=0) # (2, bs*dim_c, dim_e)
        rdt = self.doe(wt, opt_cat_1) # (2, bs*dim_c, dim_e)
        dt = torch.cat([rdt[0].squeeze(0), rdt[1].squeeze(0)], dim=-1) # (bs*dim_c, 2*dim_e)
        opt_logits = self.de_logtis_lc(dt) # (bs*dim_c, dim_c)
        opt_logits = opt_logits.view(bs, self.dim_c, self.dim_c)
        # 初始化选项策略的逻辑值
        ## deal with the init option
        opt_logits_init = self.init_option_policy(s) # (bs, dim_c)
        opt_logits_init = opt_logits_init.unsqueeze(1) # (bs, 1, dim_c)
        # 将初始化的选项策略逻辑值与已有的逻辑值进行拼接
        logits = torch.cat([opt_logits, opt_logits_init], 1) # (bs, dim_c+1, dim_c), note that the first step option is dim_c
        # 返回拼接后的逻辑值
        return logits
    # 计算状态转移的对数概率
    def log_trans(self, st, ct_1=None):
        # ct_1: long(N x 1) or None
        # ct_1: None: direct output p(ct|st, ct_1): a (N x ct_1 x ct) array where ct is log-normalized
        unnormed_pcs = self.switcher(st) # (bs, dim_c+1, dim_c)
        log_pcs = unnormed_pcs.log_softmax(dim=-1) # (bs, dim_c+1, dim_c)
        if ct_1 is None:
            return log_pcs
        else:
            return log_pcs.gather(dim=-2, index=ct_1.view(-1, 1, 1).expand(-1, 1, self.dim_c)).squeeze(dim=-2) # (bs, dim_c)

    def get_certain_param(self, low_policy=True):
        parameter = []
        if low_policy:
            for name, para in self.named_parameters():
                if ('act_doe' in name) or ('embed_option' in name):
                    # print("Low parameters: ", name)
                    parameter.append(para)
        else:
            for name, para in self.named_parameters():
                if ('act_doe' not in name) and ('embed_option' not in name):
                    # print(name)
                    # print("High parameters: ", name)
                    parameter.append(para)

        return parameter

    def get_param(self, low_policy=True):
        return self.get_certain_param(low_policy)
    # 计算动作和选项的对数概率
    def log_prob_action(self, st, ct, at):
        logits = self.a_mean_logstd(st, ct)
        log_pas = logits.log_softmax(dim=-1) # (bs, dim_a)
        return log_pas.gather(dim=-1, index=at)

    def log_prob_option(self, st, ct_1, ct):
        log_tr = self.log_trans(st, ct_1)
        return log_tr.gather(dim=-1, index=ct)
    # sample_action 和 sample_option 函数分别用于采样动作和选项
    def sample_action(self, st, ct, fixed=False, tau=1.0):
        logits = self.a_mean_logstd(st, ct)
        log_pas = logits.log_softmax(dim=-1)  # (bs, dim_a)

        if fixed:
            return log_pas.argmax(dim=-1, keepdim=True)
        else:
            return F.gumbel_softmax(log_pas, hard=False, tau=tau).multinomial(1).long()

    def sample_option(self, st, ct_1, fixed=False, tau=1.0):
        log_tr = self.log_trans(st, ct_1)
        if fixed:
            return log_tr.argmax(dim=-1, keepdim=True)
        else:
            # print(F.gumbel_softmax(log_tr, hard=False)) # (N, c_dim)
            # Note that the sampling result does not contain gradients.
            return F.gumbel_softmax(log_tr, hard=False, tau=tau).multinomial(1).long()
    # 计算策略和选项的熵
    def policy_entropy(self, st, ct):
        logits = self.a_mean_logstd(st, ct)
        log_pas = logits.log_softmax(dim=-1)  # (bs, dim_a)
        entropy = -(log_pas * log_pas.exp()).sum(dim=-1, keepdim=True)
        return entropy # (bs, 1)

    def option_entropy(self, st, ct_1):
        log_tr = self.log_trans(st, ct_1)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return entropy # (bs, 1)
    # 计算策略和选项的对数概率和熵
    def policy_log_prob_entropy(self, st, ct, at):
        logits = self.a_mean_logstd(st, ct)
        log_pas = logits.log_softmax(dim=-1)  # (bs, dim_a)
        entropy = -(log_pas * log_pas.exp()).sum(dim=-1, keepdim=True)
        return log_pas, entropy

    def option_log_prob_entropy(self, st, ct_1, ct):
        # c1 can be dim_c, c2 should always < dim_c
        log_tr = self.log_trans(st, ct_1)
        log_opt = log_tr.gather(dim=-1, index=ct)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return log_opt, entropy
# 一个多头注意力选项批评家包含一个批评家网络和一个值网络
class MHAOptionCritic(torch.nn.Module):
    def __init__(self, config, dim_s, dim_c):
        super(MHAOptionCritic, self).__init__()
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.dmodel = config.dmodel
        self.device = torch.device(config.device)

        critic_dim = self.dim_s + self.dmodel
        self.q_concat_norm = nn.LayerNorm(critic_dim)
        self.q_o_st = DoeCriticNet(critic_dim, self.dim_c, config.hidden_critic)

        self.to(self.device)

    def get_param(self):
        return list(self.parameters())

    def get_value(self, option_emb, st, ct, return_all=True): # TODO: try the original get_value
        # st: (bs, dim_s), option_emb: coming from the current policy, ct: (bs, 1)
        # this logic is weird, but the orginal code of SA uses this and get good performance
        # the best way to solve this may be to train separate critics for the low- and high-level policy
        assert ct is not None
        # 移除 ct 的最后一个维度
        ct = ct.squeeze(-1)  # (N, )
        ct_emb = option_emb(ct.unsqueeze(0)).detach().squeeze(0)  # (N, dim_e)
        v_inp = torch.cat([st, ct_emb], dim=-1) # (N, dim_s + dim_e)
        v_inp = self.q_concat_norm(v_inp)
        q_o_st = self.q_o_st(v_inp) # (N, dim_c)

        if return_all:
            return q_o_st
        else:
            return q_o_st.gather(dim=-1, index=ct.unsqueeze(-1))