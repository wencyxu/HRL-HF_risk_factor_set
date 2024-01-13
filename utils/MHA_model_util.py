import torch
from torch import nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def range_tensor(end, config_device):
    return torch.arange(end).long().to(config_device)


############################################# start from here
class SkillMhaLayer(BaseNet):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # print(tgt.shape, memory.shape)
        tgt2 = self.multihead_attn(tgt, memory, memory)[
            0]  # probably the memory or say embedding matrix will be updated here
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        # "add tgt and then norm" seems to be a module
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SkillPolicy(BaseNet):

    def __init__(self, dmodel, nhead, nlayers, nhid, dropout):
        super().__init__()
        self.layers = nn.ModuleList([SkillMhaLayer(dmodel, nhead, nhid, dropout) for i in range(nlayers)])
        self.norm = nn.LayerNorm(dmodel)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, tgt):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)
        output = self.norm(output)
        return output


class DoeDecoderFFN(BaseNet):

    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super().__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList([
            layer_init(nn.Linear(dim_in, dim_out))
            for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.out_dim = hidden_units[-1]
        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs):
        for layer in self.layers:
            obs = self.gate(layer(obs))
        return obs


# change for the discrete case
class DoeSingleTransActionNet(BaseNet):

    def __init__(self, concat_dim, action_dim, hidden_units=(64, 64)):
        super().__init__()
        self.decoder = DoeDecoderFFN(concat_dim, hidden_units)
        self.logit_fc = layer_init(nn.Linear(self.decoder.out_dim, action_dim), 1e-3)

    def forward(self, obs):
        out = self.decoder(obs)
        logits = self.logit_fc(out)
        return logits


class DoeCriticNet(BaseNet):

    def __init__(self, concat_dim, num_options, hidden_units=(64, 64)):
        super().__init__()
        self.decoder = DoeDecoderFFN(concat_dim, hidden_units)
        self.logits_lc = layer_init(nn.Linear(self.decoder.out_dim, num_options))

    def forward(self, obs):
        out = self.decoder(obs)
        q_o = self.logits_lc(out)
        return q_o
