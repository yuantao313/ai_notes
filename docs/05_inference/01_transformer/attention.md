# Attention 机制

## Attention

### 公式

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

|符号|全称|多头Shape `(B, N, S, d_k)`|单头Shape `(B, S, H)`|含义|
|---|---|---|---|---|
|Q|Query|`(B, N, S_q, d_k)`|`(B, S_q, H)`|我想找什么|
|K|Key|`(B, N, S_k, d_k)`|`(B, S_k, H)`|我有什么特征|
|V|Value|`(B, N, S_k, d_k)`|`(B, S_k, H)`|我提供什么信息|

- 维度约定（全文统一）

|符号|含义|关系|
|---|---|---|
|$B$|Batch size|一次推理多少句子|
|$S$ / $T$|Sequence length / Token length|一个句子多少个 token|
|$N$|Number of heads|注意力头数|
|$d_k$|每头维度（per-head dim）|$d_k = H / N$|
|$H$|Hidden size（模型总维度）|$H = N \cdot d_k$|

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

# BSH: 单头注意力，不拆多头，整个 H 作为一个头计算
class SelfAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.W_q = nn.Linear(H, H)
        self.W_k = nn.Linear(H, H)
        self.W_v = nn.Linear(H, H)
        self.W_o = nn.Linear(H, H)

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)   # (B, S_q, H)
        K = self.W_k(key)     # (B, S_k, H)
        V = self.W_v(value)   # (B, S_k, H)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.H ** 0.5)  # (B, S_q, S_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)                               # (B, S_q, S_k)
        output = torch.matmul(weights, V)                                 # (B, S_q, H)
        return self.W_o(output)
```


## 多头注意力 MHA

每个头有独立的 Q/K/V，并行计算后再拼接回 H 维。

```py
# BSND: 拆成 N 个头，每头 d_k = H // N 维
class MultiHeadAttention(nn.Module):
    def __init__(self, H, N):
        super().__init__()
        self.H = H            # hidden size
        self.N = N            # num heads
        self.d_k = H // N     # per-head dim

        self.W_q = nn.Linear(H, H)
        self.W_k = nn.Linear(H, H)
        self.W_v = nn.Linear(H, H)
        self.W_o = nn.Linear(H, H)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        S_q, S_k = query.size(1), key.size(1)

        # 投影后拆头: (B, S, H) -> (B, S, N, d_k) -> (B, N, S, d_k)
        Q = self.W_q(query).view(B, S_q, self.N, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, S_k, self.N, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, S_k, self.N, self.d_k).transpose(1, 2)

        # (B, N, S_q, d_k) x (B, N, d_k, S_k) -> (B, N, S_q, S_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        # (B, N, S_q, S_k) x (B, N, S_k, d_k) -> (B, N, S_q, d_k)
        output = torch.matmul(weights, V)

        # 合并多头: (B, N, S_q, d_k) -> (B, S_q, H)
        output = output.transpose(1, 2).contiguous().view(B, S_q, self.H)
        return self.W_o(output)
```


## 多查询注意力 MQA

所有头共享同一份 K 和 V，只有 Q 是多头的。极大减小 KV Cache。
`W_k` / `W_v` 只映射到 $d_k$（而非 $H$）。

```py
class MultiQueryAttention(nn.Module):
    def __init__(self, H, N):
        super().__init__()
        self.H = H
        self.N = N
        self.d_k = H // N

        # Q 投影到完整 H (N * d_k)
        self.W_q = nn.Linear(H, H)
        # K / V 只投影到 d_k（所有头共享）
        self.W_k = nn.Linear(H, self.d_k)
        self.W_v = nn.Linear(H, self.d_k)
        self.W_o = nn.Linear(H, H)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        S_q, S_k = query.size(1), key.size(1)

        # Q: (B, S_q, N, d_k) 每个头独立
        Q = self.W_q(query).view(B, S_q, self.N, self.d_k).transpose(1, 2)

        # K / V: (B, S_k, d_k) -> unsqueeze 成 (B, 1, S_k, d_k) 广播到所有头
        K = self.W_k(key).unsqueeze(1)    # (B, 1, S_k, d_k)
        V = self.W_v(value).unsqueeze(1)  # (B, 1, S_k, d_k)

        # (B, N, S_q, d_k) x (B, 1, d_k, S_k) -> (B, N, S_q, S_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        # (B, N, S_q, S_k) x (B, 1, S_k, d_k) -> (B, N, S_q, d_k)
        output = torch.matmul(weights, V)

        output = output.transpose(1, 2).contiguous().view(B, S_q, self.H)
        return self.W_o(output)
```


## 分组查询注意力 GQA

$N$ 个头分成 $G$ 组，组内共享 K/V。$G=N$ 退化为 MHA，$G=1$ 退化为 MQA。
`W_k` / `W_v` 映射到 $G \cdot d_k$（而非 $H$）。

```py
class GroupedQueryAttention(nn.Module):
    def __init__(self, H, N, G):
        super().__init__()
        self.H = H
        self.N = N              # 总头数
        self.G = G              # KV 组数
        self.d_k = H // N       # 每头维度
        self.heads_per_group = N // G  # 每组几个 Q 头共享同一 KV

        self.W_q = nn.Linear(H, H)                # H -> N * d_k
        self.W_k = nn.Linear(H, G * self.d_k)     # H -> G * d_k
        self.W_v = nn.Linear(H, G * self.d_k)
        self.W_o = nn.Linear(H, H)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        S_q, S_k = query.size(1), key.size(1)

        # Q: (B, S_q, H) -> (B, N, S_q, d_k)
        Q = self.W_q(query).view(B, S_q, self.N, self.d_k).transpose(1, 2)

        # K / V: (B, S_k, H) -> (B, S_k, G, d_k) -> repeat 到 N 个头
        K = self.W_k(key).view(B, S_k, self.G, self.d_k)        # (B, S_k, G, d_k)
        V = self.W_v(value).view(B, S_k, self.G, self.d_k)      # (B, S_k, G, d_k)
        K = K.repeat_interleave(self.heads_per_group, dim=2)    # (B, S_k, N, d_k)
        V = V.repeat_interleave(self.heads_per_group, dim=2)    # (B, S_k, N, d_k)
        K = K.transpose(1, 2)  # (B, N, S_k, d_k)
        V = V.transpose(1, 2)  # (B, N, S_k, d_k)

        # (B, N, S_q, d_k) x (B, N, d_k, S_k) -> (B, N, S_q, S_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)  # (B, N, S_q, d_k)
        output = output.transpose(1, 2).contiguous().view(B, S_q, self.H)
        return self.W_o(output)
```


## 多头潜在注意力 MLA

DeepSeek-V2/V3 提出。核心思想：对 KV 做低秩联合压缩，大幅减小 KV Cache。
推理时只存压缩后的潜在向量 $c_t^{KV}$（维度 $d_c \ll H$），大幅节约显存。
Q 同样可压缩以减少参数量。

$$
\begin{aligned}
c_t^{KV} &= W^{DKV} \cdot h_t &
&\in \mathbb{R}^{d_c} \\[4pt]
k_t &= W^{UK} \cdot c_t^{KV} &
&\in \mathbb{R}^{N \cdot d_k} \\[4pt]
v_t &= W^{UV} \cdot c_t^{KV} &
&\in \mathbb{R}^{N \cdot d_k} \\[4pt]
c_t^{Q} &= W^{DQ} \cdot h_t &
&\in \mathbb{R}^{d_c} \\[4pt]
q_t &= W^{UQ} \cdot c_t^{Q} &
&\in \mathbb{R}^{N \cdot d_k}
\end{aligned}
$$

```py
class MLA(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2 style)
    KV 低秩联合压缩: H -> d_c -> H, 推理时只缓存 d_c 维的 latent 向量
    """
    def __init__(self, H, N, d_c):
        super().__init__()
        self.H = H          # hidden size
        self.N = N          # num heads
        self.d_k = H // N   # per-head dim
        self.d_c = d_c      # latent dim for KV compression (d_c << H)

        # KV 联合压缩: 下投影 -> 上投影 (类似 AutoEncoder)
        self.W_DKV = nn.Linear(H, d_c, bias=False)   # H -> d_c  KV 压缩
        self.W_UK  = nn.Linear(d_c, H, bias=False)   # d_c -> H  K 解码
        self.W_UV  = nn.Linear(d_c, H, bias=False)   # d_c -> H  V 解码

        # Q 也可压缩 (减少 Q 的参数量)
        self.W_DQ  = nn.Linear(H, d_c, bias=False)   # H -> d_c  Q 压缩
        self.W_UQ  = nn.Linear(d_c, H, bias=False)   # d_c -> H  Q 解码

        self.W_o = nn.Linear(H, H, bias=False)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        S_q, S_k = query.size(1), key.size(1)

        # Q: H -> d_c -> H, 然后拆成多头
        c_q = self.W_DQ(query)                                   # (B, S_q, d_c)
        Q = self.W_UQ(c_q).view(B, S_q, self.N, self.d_k).transpose(1, 2)  # (B, N, S_q, d_k)

        # K, V: H -> d_c (共享压缩) -> H (分别解码)
        c_kv = self.W_DKV(key)                                   # (B, S_k, d_c)  只存这个!
        K = self.W_UK(c_kv).view(B, S_k, self.N, self.d_k).transpose(1, 2)  # (B, N, S_k, d_k)
        V = self.W_UV(c_kv).view(B, S_k, self.N, self.d_k).transpose(1, 2)  # (B, N, S_k, d_k)

        # 标准多头注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)  # (B, N, S_q, d_k)

        output = output.transpose(1, 2).contiguous().view(B, S_q, self.H)
        return self.W_o(output)
```

> **关键区别**: MLA 推理时 KV Cache 只存 $c_t^{KV} \in \mathbb{R}^{d_c}$，而 MHA 存 $\mathbb{R}^{N \cdot d_k}$。
> 当 $d_c \ll H$ 时（如 DeepSeek-V2 中 $d_c = H/4$），KV Cache 减少约 $4\times$。
