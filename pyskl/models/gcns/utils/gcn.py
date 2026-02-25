import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer

from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=False,
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        if self.adaptive == 'offset':
            A = self.A + self.PA
        elif self.adaptive == 'importance':
            A = self.A * self.PA
        else:
            A = self.A

        if self.conv_pos == 'pre':
            x = self.conv(x)                                          # (N, K*C, T, V)
            x = (x.reshape(n, self.num_subsets, -1, v) @ A).sum(1).reshape(n, -1, t, v)
        elif self.conv_pos == 'post':
            x = (x.reshape(n, 1, -1, v) @ A).reshape(n, -1, t, v)   # (N, K*C, T, V)
            x = self.conv(x)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass


class unit_aagcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, attention=True):
        super(unit_aagcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

        if self.attention:
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A):

        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)


class unit_transformer(nn.Module):
    """Transformer unit as a drop-in replacement for unit_gcn.

    Applies multi-head self-attention over the joint (V) dimension,
    with T treated as part of the batch for efficiency.
    Input/output: (N, C, T, V) — identical interface to unit_gcn.

    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels. Must be divisible by num_heads.
        A: Adjacency matrix (used to optionally initialize graph_bias).
        num_heads (int): Number of attention heads. Default: 8.
        with_res (bool): Add residual connection. Default: False.
        norm (str or dict): Normalization layer type. 'BN' (default), 'GN'
            (GroupNorm with num_groups=1, i.e. LN per position), or a full
            mmcv norm cfg dict e.g. dict(type='GN', num_groups=4).
        act (str): Activation layer type. Default: 'ReLU'.
        attn_dropout (float): Dropout on attention weights. Default: 0.
        use_graph_bias (bool): Add learnable graph-structured bias to attention
            logits, initialized from A. Default: False.
        ffn_ratio (float): Hidden-dim expansion ratio for the feed-forward
            network appended after attention. 0 disables the FFN. Default: 2.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 num_heads=8,
                 with_res=False,
                 norm='BN',
                 act='ReLU',
                 attn_dropout=0.,
                 use_graph_bias=False,
                 use_pos_embed=True,
                 ffn_ratio=2.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_res = with_res
        self.use_graph_bias = use_graph_bias
        self.use_pos_embed = use_pos_embed

        assert out_channels % num_heads == 0, (
            f'out_channels ({out_channels}) must be divisible by num_heads ({num_heads})')
        self.head_dim = out_channels // num_heads
        self.attn_dropout_p = attn_dropout
        self.attn_weights = None   # populated during eval forward for visualization
        self.logit_scale = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(math.sqrt(out_channels // num_heads)))

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        # GroupNorm requires num_groups; default to 1 (all channels in one group ≈ LN).
        if self.norm_cfg.get('type') == 'GN' and 'num_groups' not in self.norm_cfg:
            self.norm_cfg = {**self.norm_cfg, 'num_groups': 1}
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, in_channels)[1]  # pre-norm: normalizes input
        self.act = build_activation_layer(self.act_cfg)

        if use_pos_embed:
            num_joints = A.size(1)
            # (C, V) — broadcast over N and T in forward
            self.pos_embed = nn.Parameter(torch.zeros(in_channels, num_joints))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.qkv = nn.Conv2d(in_channels, 3 * out_channels, 1)
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

        if use_graph_bias:
            # Learnable per-head bias initialized from the mean adjacency matrix
            A_mean = A.mean(0)  # (V, V)
            self.graph_bias = nn.Parameter(A_mean[None].expand(num_heads, -1, -1).clone())

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

        if ffn_ratio > 0:
            ffn_hidden = int(out_channels * ffn_ratio)
            self.ffn = nn.Sequential(
                nn.Conv2d(out_channels, ffn_hidden, 1),
                nn.GELU(),
                nn.Conv2d(ffn_hidden, out_channels, 1),
            )
            self.bn_ffn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        h, d = self.num_heads, self.head_dim
        res = self.down(x) if self.with_res else 0

        # Pre-norm before attention
        x = self.bn(x)

        if self.use_pos_embed:
            x = x + self.pos_embed[None, :, None, :]  # (1, C, 1, V) broadcast

        # Fused QKV projection then reshape to (3, N*T, h, V, d)
        qkv = self.qkv(x).view(n, 3, h, d, t, v).permute(1, 0, 4, 2, 5, 3).contiguous().reshape(3, n * t, h, v, d)
        q, k, val = qkv.unbind(0)  # each (N*T, h, V, d)
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        q, k = F.normalize(q, dim=-1) * scale, F.normalize(k, dim=-1)

        # Flash Attention over V (joints), treating N*T as batch
        attn_mask = self.graph_bias[None] if self.use_graph_bias else None
        if not self.training:
            # Compute weights explicitly for visualization
            logits = (q @ k.transpose(-2, -1)) / math.sqrt(d)
            if attn_mask is not None:
                logits = logits + attn_mask
            self.attn_weights = logits.softmax(dim=-1).detach().cpu()  # (N*T, h, V, V)
        x = F.scaled_dot_product_attention(
            q, k, val,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.
        )  # (N*T, h, V, d)

        # Reshape back to (N, out_channels, T, V)
        x = x.reshape(n, t, h, v, d).permute(0, 2, 4, 1, 3).reshape(n, self.out_channels, t, v)
        x = self.proj(x)

        x = self.act(x + res)

        if hasattr(self, 'ffn'):
            x = x + self.ffn(self.bn_ffn(x))

        return x

    def init_weights(self):
        pass


class JointPool(nn.Module):
    """Attention-weighted joint pooling: (N, C, T, V) -> (N, C, T, V').

    Scores each joint with a learned pointwise conv, softmax-normalizes the
    scores within each pre-defined body-part group, then computes a weighted
    sum to form super-joint features.

    Zero-initializing ``score_net`` means the module starts as plain mean
    pooling and gradually learns to up/down-weight joints during training.

    Args:
        groups (list[list[int]]): Each inner list contains the joint indices
            that belong to one super-joint (output node).
        in_channels (int): Number of input feature channels.
    """

    def __init__(self, groups, in_channels):
        super().__init__()
        self.groups = groups
        V_in = sum(len(g) for g in groups)
        V_out = len(groups)

        # Additive mask for grouped softmax: 0 where joint i ∈ group j, -inf elsewhere
        mask = torch.full((V_in, V_out), float('-inf'))
        for j, group in enumerate(groups):
            for i in group:
                mask[i, j] = 0.0
        self.register_buffer('group_mask', mask)  # (V_in, V_out)

        # Per-joint scalar score; zero-init → uniform weights at start
        self.score_net = nn.Conv2d(in_channels, 1, 1, bias=False)
        nn.init.zeros_(self.score_net.weight)

    def forward(self, x):
        # x: (N, C, T, V)
        score = self.score_net(x).squeeze(1)          # (N, T, V)
        # Masked softmax: for each group column, softmax over its member joints
        weights = (score.unsqueeze(-1) + self.group_mask).softmax(dim=-2)  # (N, T, V, V')
        return torch.einsum('nctv,ntvj->nctj', x, weights)                 # (N, C, T, V')

    def init_weights(self):
        nn.init.zeros_(self.score_net.weight)


class unit_sgn(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: N, C, T, V; A: N, T, V, V
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))


class dggcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        A = self.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
            ada_graph = getattr(self, self.ada_act)(ada_graph)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        return self.act(self.bn(x) + res)
