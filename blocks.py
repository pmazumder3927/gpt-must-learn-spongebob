import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import GPTConfig


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1
        # bias mask for causal attention
        self.register_buffer("bias", torch.tril(torch.ones(
            config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # visualize the bias mask
        # the view is to make the bias mask broadcastable with the attention score
        # shape of bias: (1, 1, block_size, block_size)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, n_embd
        # query, key, value
        qkv = self.c_attn(x)
        # split into q, k, v, use split instead of chunk to ensure contiguity
        # for example, if we use chunk, the q, k, v will be in different memory locations
        q, k, v = qkv.split(self.n_embd, dim=2)
        # batch B, nh for pytorch multihead attention, n_embd for qkv
        # B, T, nh, C // nh
        # we need to transpose the tensor to make the shape (B, nh, T, C // nh)
        # general operation
        q, k, v = map(lambda x: x.view(B, T, self.n_head, C //
                                       self.n_head).transpose(1, 2), (q, k, v))
        # attention
        # calculation explanation:
        # q @ k.transpose(-2, -1) -> (B, nh, T, C // nh) @ (B, nh, C // nh, T) -> (B, nh, T, T)
        # then scale the attention score by 1 / sqrt(k.size(-1))
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # mask attent to sequence inputs before the current attended token
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # softmax along the sequence length dimension so that the attention score sums to 1
        att = F.softmax(att, dim=-1)
        # (B, nh, T, T) @ (B, nh, T, C // nh) -> (B, nh, T, C // nh)
        y = att @ v
        # now fix the dimensions to be (B, T, nh, C // nh) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
