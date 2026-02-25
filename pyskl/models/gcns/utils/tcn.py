import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from .init_func import bn_init, conv_init


class tcn_transformer(nn.Module):
    """Temporal transformer as a drop-in replacement for unit_tcn / mstcn.

    Applies multi-head self-attention over the time (T) dimension, treating
    N*V as the batch.  Strided downsampling is handled by average pooling
    after attention, matching the interface of the convolutional TCN modules.

    Input / output shape: (N, C, T, V) -> (N, C, T // stride, V).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels. Must be divisible by num_heads.
        stride (int): Temporal downsampling factor. Default: 1.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (float): Output dropout probability. Default: 0.
        attn_dropout (float): Attention weight dropout. Default: 0.
        max_len (int): Maximum clip length supported. Default: 32.
        use_pos_embed (bool): Add learnable relative position bias to attention
            logits. Translation-equivariant: only relative frame distance matters.
            Default: True.
        kernel_size (int or None): Local attention window. Frame i may only
            attend to frames j where |i-j| <= kernel_size // 2, matching the
            receptive field of unit_tcn. None means full (global) attention.
            Default: None.
        ffn_ratio (float): Hidden-dim expansion ratio for the feed-forward
            network appended after attention. 0 disables the FFN entirely.
            Standard transformer blocks use 4; 2 is a lighter option.
            Default: 2.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 num_heads=None,
                 head_dim=32,
                 dropout=0.,
                 attn_dropout=0.,
                 max_len=32,
                 use_pos_embed=True,
                 kernel_size=None,
                 qkv_kernel=1,
                 norm='BN',
                 ffn_ratio=0):
        super().__init__()
        # num_heads takes priority; otherwise derive from fixed head_dim
        if num_heads is None:
            num_heads = max(1, out_channels // head_dim)
        assert out_channels % num_heads == 0, (
            f'out_channels ({out_channels}) must be divisible by num_heads ({num_heads})')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.stride = stride
        self.use_pos_embed = use_pos_embed
        self.logit_scale = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(math.sqrt(out_channels // num_heads)))
        self.attn_dropout_p = attn_dropout
        self.kernel_size = kernel_size

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        if self.norm_cfg.get('type') == 'GN' and 'num_groups' not in self.norm_cfg:
            self.norm_cfg = {**self.norm_cfg, 'num_groups': 1}

        # Precompute pairwise distance and index tables (shared by RPE and local mask).
        coords = torch.arange(max_len)
        dist = (coords[:, None] - coords[None, :]).abs()        # (max_len, max_len)
        rel_idx = coords[None, :] - coords[:, None] + (max_len - 1)  # (max_len, max_len)
        self.register_buffer('rel_pos_index', rel_idx.long())

        if use_pos_embed:
            # Relative position bias: one scalar per head per signed offset.
            # Offsets range from -(max_len-1) to +(max_len-1) → 2*max_len-1 entries.
            self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        if kernel_size is not None:
            # Boolean mask: True = frame pair within the local window.
            self.register_buffer('local_mask', dist <= kernel_size // 2)  # (max_len, max_len)

        self.qk = nn.Conv2d(in_channels, 2 * out_channels, 1)
        self.v  = nn.Conv2d(in_channels, out_channels,
                            kernel_size=(qkv_kernel, 1),
                            padding=((qkv_kernel - 1) // 2, 0))
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.bn_out = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.drop = nn.Dropout(dropout)

        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=(stride, 1), stride=(stride, 1)) \
            if stride > 1 else nn.Identity()

        if ffn_ratio > 0:
            ffn_hidden = int(out_channels * ffn_ratio)
            self.ffn = nn.Sequential(
                nn.Conv2d(out_channels, ffn_hidden, 1),
                nn.GELU(),
                nn.Conv2d(ffn_hidden, out_channels, 1),
            )
            self.bn_ffn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x):
        n, _, t, v = x.shape
        h, d = self.num_heads, self.head_dim

        x_in = x
        x = self.bn(x)

        # Q and K are pointwise; V uses a local kernel.  All shaped (N*V, h, T, d).
        q, k = self.qk(x).view(n, 2, h, d, t, v).permute(1, 0, 5, 2, 4, 3).reshape(2, n * v, h, t, d).unbind(0)
        val  = self.v(x).view(n, h, d, t, v).permute(0, 4, 1, 3, 2).reshape(n * v, h, t, d)
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        q, k = F.normalize(q, dim=-1) * scale, F.normalize(k, dim=-1)

        # Build additive attention bias: RPE + local window mask.
        attn_bias = None
        if self.use_pos_embed:
            # (h, T, T) — relative position bias for each head
            attn_bias = self.rel_pos_bias[:, self.rel_pos_index[:t, :t].reshape(-1)].reshape(h, t, t)

        if self.kernel_size is not None:
            # Set out-of-window logits to -inf so softmax zeros them out.
            local = self.local_mask[:t, :t]  # (t, t) bool
            fill = torch.full((h, t, t), float('-inf'), dtype=q.dtype, device=q.device)
            if attn_bias is not None:
                attn_bias = torch.where(local[None], attn_bias, fill)
            else:
                attn_bias = fill.masked_fill(local[None], 0.)

        # Flash Attention over T, treating N*V as batch
        x = F.scaled_dot_product_attention(
            q, k, val,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout_p if self.training else 0.
        )  # (N*V, h, T, d)

        # Reshape back to (N, out_channels, T, V)
        x = x.reshape(n, v, h, t, d).permute(0, 2, 4, 3, 1).reshape(n, self.out_channels, t, v)
        x = self.proj(x) + x_in

        # Temporal downsampling (replaces strided conv)
        x = self.pool(x)

        x = self.bn_out(x)
        x = self.drop(x)

        if hasattr(self, 'ffn'):
            x = x + self.ffn(self.bn_ffn(x))

        return x

    def init_weights(self):
        # Small init for QK/V keeps early attention logits manageable.
        nn.init.trunc_normal_(self.qk.weight, std=0.02)
        nn.init.constant_(self.qk.bias, 0)
        nn.init.trunc_normal_(self.v.weight, std=0.02)
        nn.init.constant_(self.v.bias, 0)
        # Zero-init output projection so the attention branch starts as identity.
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)
        bn_init(self.bn, 1)
        bn_init(self.bn_out, 1)
        if self.stride > 1:
            conv_init(self.pool)
        if hasattr(self, 'ffn'):
            # Zero-init the last FFN layer for the same reason.
            nn.init.constant_(self.ffn[-1].weight, 0)
            nn.init.constant_(self.ffn[-1].bias, 0)
            bn_init(self.bn_ffn, 1)


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass


class dgmstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat

        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
