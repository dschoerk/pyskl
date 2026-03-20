import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth: randomly drop entire residual branches during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One random value per sample, broadcast over all other dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob
        return x * mask / keep_prob


class SpatialAttentionBlock(nn.Module):
    """Multi-head self-attention across joints within each frame.

    Input:  (B, T, V, D)
    Output: (B, T, V, D)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, drop_path=0.0,
                 use_graph_bias=False, A=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(math.sqrt(embed_dim // num_heads)))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_graph_bias = use_graph_bias
        if use_graph_bias:
            assert A is not None, 'A must be provided when use_graph_bias=True'
            A_mean = A.mean(0)  # (V, V) — mean over subsets
            self.graph_bias = nn.Parameter(A_mean[None].expand(num_heads, -1, -1).clone())

    def forward(self, x):
        B, T, V, D = x.shape
        h, d = self.num_heads, self.head_dim
        x_flat = x.reshape(B * T, V, D)

        # Pre-norm + MHSA
        residual = x_flat
        x_flat = self.norm1(x_flat)

        # (B*T, V, 3D) -> (B*T, h, V, d)
        qkv = self.qkv(x_flat).reshape(B * T, V, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)

        # Slice to actual V (graph_bias was allocated for num_nodes), cast for AMP
        attn_mask = self.graph_bias[:, :V, :V].unsqueeze(0).to(q.dtype) if self.use_graph_bias else None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B*T, h, V, d)
        out = out.permute(0, 2, 1, 3).reshape(B * T, V, D)
        out = self.proj(out)

        x_flat = residual + self.drop_path(out)

        # Pre-norm + FFN
        x_flat = x_flat + self.drop_path(self.ffn(self.norm2(x_flat)))

        return x_flat.reshape(B, T, V, D)


class TemporalAttentionBlock(nn.Module):
    """Multi-head self-attention across frames for each joint.

    Uses learnable relative position bias so the model captures that
    "2 frames apart" is the same relationship regardless of absolute position.

    Input:  (B, T, V, D)
    Output: (B, T, V, D)
    """

    def __init__(self, embed_dim, num_heads, max_T=300,
                 dropout=0.0, drop_path=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_T = max_T
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(math.sqrt(embed_dim // num_heads)))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Relative position bias table: (num_heads, 2*max_T - 1)
        # Index 0 = offset -(max_T-1), index max_T-1 = offset 0, etc.
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_T - 1))

    def _get_rel_pos_bias(self, T):
        """Build (num_heads, T, T) relative position bias for sequence length T."""
        coords = torch.arange(T, device=self.rel_pos_bias.device)
        rel_pos = coords.unsqueeze(1) - coords.unsqueeze(0)  # (T, T)
        rel_pos = rel_pos + self.max_T - 1  # shift so min index is 0
        bias = self.rel_pos_bias[:, rel_pos]  # (num_heads, T, T)
        return bias

    def forward(self, x):
        B, T, V, D = x.shape
        h, d = self.num_heads, self.head_dim
        x_flat = x.permute(0, 2, 1, 3).reshape(B * V, T, D)

        # Pre-norm + MHSA with relative position bias
        residual = x_flat
        x_flat = self.norm1(x_flat)

        # (B*V, T, 3D) -> (B*V, h, T, d)
        qkv = self.qkv(x_flat).reshape(B * V, T, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)

        rel_bias = self._get_rel_pos_bias(T).unsqueeze(0).to(q.dtype)  # (1, h, T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=rel_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B*V, h, T, d)
        out = out.permute(0, 2, 1, 3).reshape(B * V, T, D)
        out = self.proj(out)

        x_flat = residual + self.drop_path(out)

        # Pre-norm + FFN
        x_flat = x_flat + self.drop_path(self.ffn(self.norm2(x_flat)))

        return x_flat.reshape(B, V, T, D).permute(0, 2, 1, 3)


class CrossPersonAttentionBlock(nn.Module):
    """Multi-head self-attention across persons for interaction modeling.

    For each (frame, joint) position, attends across the M person dimension.

    Input:  (N, M, T, V, D)
    Output: (N, M, T, V, D)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, drop_path=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(math.sqrt(embed_dim // num_heads)))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        N, M, T, V, D = x.shape
        h, d = self.num_heads, self.head_dim
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(N * T * V, M, D)

        residual = x_flat
        x_flat = self.norm1(x_flat)

        # (N*T*V, M, 3D) -> (N*T*V, h, M, d)
        qkv = self.qkv(x_flat).reshape(N * T * V, M, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        q = F.normalize(q, dim=-1) * scale
        k = F.normalize(k, dim=-1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (N*T*V, h, M, d)
        out = out.permute(0, 2, 1, 3).reshape(N * T * V, M, D)
        out = self.proj(out)

        x_flat = residual + self.drop_path(out)
        x_flat = x_flat + self.drop_path(self.ffn(self.norm2(x_flat)))

        return x_flat.reshape(N, T, V, M, D).permute(0, 3, 1, 2, 4)


class JointSpatioTemporalBlock(nn.Module):
    """Multi-head self-attention over the full T*V token sequence.

    Instead of factorizing into separate spatial and temporal passes, this
    block flattens (T, V) into a single sequence of length T*V and runs
    full self-attention.  The attention bias is additively decomposed:

        bias(t_i, v_i, t_j, v_j) = rel_pos[t_i - t_j] + graph[v_i, v_j]

    This lets every joint attend to every other joint at every frame in a
    single attention operation while still injecting structural priors.

    Input:  (B, T, V, D)
    Output: (B, T, V, D)
    """

    def __init__(self, embed_dim, num_heads, max_T=300,
                 dropout=0.0, drop_path=0.0,
                 use_graph_bias=False, A=None,
                 split_factor=None, near=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_T = max_T
        self.dropout = dropout
        self.split_factor = split_factor
        self.near = near

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(
            torch.ones(num_heads, 1, 1) * math.log(math.sqrt(embed_dim // num_heads)))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Temporal relative position bias: (num_heads, 2*max_T - 1)
        # Initialize with Gaussian decay so nearby frames dominate (like TCN)
        offsets = torch.arange(2 * max_T - 1) - (max_T - 1)  # -(max_T-1) .. +(max_T-1)
        temporal_sigma = 4.0  # std in frames — ~8-frame effective window at init
        temporal_init = -(offsets.float() ** 2) / (2 * temporal_sigma ** 2)
        temporal_init = temporal_init.clamp(min=-10.0)
        self.rel_pos_bias = nn.Parameter(
            temporal_init.unsqueeze(0).expand(num_heads, -1).clone())

        # Spatial graph bias
        self.use_graph_bias = use_graph_bias
        if use_graph_bias:
            assert A is not None, 'A must be provided when use_graph_bias=True'
            # Binary connectivity: 1 where any edge exists, 0 elsewhere
            A_binary = (A.sum(0) > 0).float()  # (V, V)
            # Strong negative for non-neighbors, positive for neighbors (like GCN)
            graph_init = torch.where(A_binary > 0,
                                     torch.tensor(1.0),
                                     torch.tensor(-5.0))
            self.graph_bias = nn.Parameter(
                graph_init[None].expand(num_heads, -1, -1).clone())

    def _build_bias(self, T, V, dtype):
        """Build (num_heads, T*V, T*V) attention bias.

        Decomposes into temporal relative position bias tiled over joints
        plus spatial graph bias tiled over frames.
        """
        # Temporal: (h, T, T) -> expand to (h, T, 1, T, 1) -> (h, T, V, T, V)
        t_coords = torch.arange(T, device=self.rel_pos_bias.device)
        t_rel = t_coords.unsqueeze(1) - t_coords.unsqueeze(0) + self.max_T - 1  # (T, T)
        t_bias = self.rel_pos_bias[:, t_rel]  # (h, T, T)
        # Kronecker-style expansion: repeat for every (v_i, v_j) pair
        t_bias = t_bias[:, :, None, :, None].expand(-1, -1, V, -1, V)  # (h, T, V, T, V)

        bias = t_bias

        # Spatial: (h, V, V) -> expand to (h, 1, V, 1, V) -> (h, T, V, T, V)
        if self.use_graph_bias:
            s_bias = self.graph_bias[:, :V, :V]  # slice to actual V
            s_bias = s_bias[:, None, :, None, :].expand(-1, T, -1, T, -1)  # (h, T, V, T, V)
            bias = bias + s_bias

        # Flatten to (h, T*V, T*V)
        bias = bias.reshape(self.num_heads, T * V, T * V)
        return bias.unsqueeze(0).to(dtype)  # (1, h, T*V, T*V)

    def _build_split_bias(self, Tk, V, stride, dtype):
        """Build (1, h, Tk*V, Tk*V) attention bias for split attention.

        Args:
            Tk: number of attended temporal positions (the kept axis).
            V: number of joints.
            stride: real frame distance between consecutive attended frames.
                    1 for near (consecutive), T2 for far (strided).
            dtype: output dtype.
        """
        # Temporal: relative offsets in real frame units are 0, ±stride, ±2*stride, ...
        t_coords = torch.arange(Tk, device=self.rel_pos_bias.device)
        t_rel = (t_coords.unsqueeze(1) - t_coords.unsqueeze(0)) * stride  # (Tk, Tk)
        t_rel = t_rel + self.max_T - 1  # shift to index into bias table
        t_bias = self.rel_pos_bias[:, t_rel]  # (h, Tk, Tk)
        t_bias = t_bias[:, :, None, :, None].expand(-1, -1, V, -1, V)  # (h, Tk, V, Tk, V)

        bias = t_bias

        if self.use_graph_bias:
            s_bias = self.graph_bias[:, :V, :V]  # (h, V, V)
            s_bias = s_bias[:, None, :, None, :].expand(-1, Tk, -1, Tk, -1)
            bias = bias + s_bias

        bias = bias.reshape(self.num_heads, Tk * V, Tk * V)
        return bias.unsqueeze(0).to(dtype)  # (1, h, Tk*V, Tk*V)

    def forward(self, x):
        B, T, V, D = x.shape
        h, d = self.num_heads, self.head_dim
        sf = self.split_factor

        # Find the largest divisor of T that is <= split_factor
        T1 = None
        if sf is not None and T > sf:
            for f in range(sf, 1, -1):
                if T % f == 0:
                    T1 = f
                    break

        # Pre-norm
        x_flat = x.reshape(B, T * V, D)
        residual = x_flat
        x_flat = self.norm1(x_flat)

        if T1 is not None:
            T2 = T // T1

            if self.near:
                # Near: merge T1 into batch, attend over T2*V consecutive tokens
                # (B, T1, T2, V, D) -> (B*T1, T2*V, D)
                x_split = x_flat.reshape(B, T1, T2, V, D).reshape(B * T1, T2 * V, D)
                Tk, stride = T2, 1
            else:
                # Far: merge T2 into batch, attend over T1*V strided tokens
                # (B, T1, T2, V, D) -> (B, T1, T2, V, D)
                #   -> permute to (B, T2, T1, V, D) -> (B*T2, T1*V, D)
                x_split = (x_flat.reshape(B, T1, T2, V, D)
                           .permute(0, 2, 1, 3, 4)
                           .reshape(B * T2, T1 * V, D))
                Tk, stride = T1, T2

            Bb = x_split.shape[0]  # B*T1 or B*T2
            L = Tk * V

            qkv = self.qkv(x_split).reshape(Bb, L, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # each (Bb, h, L, d)

            scale = self.logit_scale.clamp(max=math.log(100)).exp()
            q = F.normalize(q, dim=-1) * scale
            k = F.normalize(k, dim=-1)

            attn_bias = self._build_split_bias(Tk, V, stride, q.dtype)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (Bb, h, L, d)
            out = out.permute(0, 2, 1, 3).reshape(Bb, L, D)
            out = self.proj(out)

            # Restore to (B, T*V, D)
            if self.near:
                out = out.reshape(B, T1, T2, V, D).reshape(B, T * V, D)
            else:
                out = (out.reshape(B, T2, T1, V, D)
                       .permute(0, 2, 1, 3, 4)
                       .reshape(B, T * V, D))
        else:
            # Full attention fallback
            qkv = self.qkv(x_flat).reshape(B, T * V, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            scale = self.logit_scale.clamp(max=math.log(100)).exp()
            q = F.normalize(q, dim=-1) * scale
            k = F.normalize(k, dim=-1)

            attn_bias = self._build_bias(T, V, q.dtype)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )
            out = out.permute(0, 2, 1, 3).reshape(B, T * V, D)
            out = self.proj(out)

        x_flat = residual + self.drop_path(out)

        # Pre-norm + FFN
        x_flat = x_flat + self.drop_path(self.ffn(self.norm2(x_flat)))

        return x_flat.reshape(B, T, V, D)


class TemporalPooling(nn.Module):
    """Halve the temporal dimension via strided 1-D avg-pool over T.

    Input:  (*, T, V, D)
    Output: (*, T//2, V, D)
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # x: (..., T, V, D) — pool along T
        *leading, T, V, D = x.shape
        # Reshape to (B, D, T, V) for pool2d (pool only T, keep V)
        x = x.reshape(-1, T, V, D).permute(0, 3, 1, 2)  # (B, D, T, V)
        x = F.avg_pool2d(x, (self.kernel_size, 1), (self.stride, 1))  # (B, D, T', V)
        T_new = x.shape[2]
        x = x.permute(0, 2, 3, 1)  # (B, T', V, D)
        return x.reshape(*leading, T_new, V, D)


class JointTransformerBlock(nn.Module):
    """One layer of joint spatio-temporal attention + optional cross-person.

    Joint ST attention (T*V tokens) -> Cross-person attention (optional)
    -> Temporal pooling (optional) -> Channel projection (when in_dim != out_dim).

    Input:  (N, M, T, V, in_dim)
    Output: (N, M, T', V, out_dim)   where T' = T // stride if downsample else T
    """

    def __init__(self, in_dim, out_dim, num_heads, num_person=2, max_T=300,
                 dropout=0.0, drop_path=0.0,
                 use_graph_bias=False, A=None,
                 use_cross_person=True,
                 downsample=False,
                 split_factor=None, near=True):
        super().__init__()
        self.use_cross_person = use_cross_person
        self.downsample = downsample

        self.st_attn = JointSpatioTemporalBlock(
            in_dim, num_heads, max_T, dropout, drop_path,
            use_graph_bias, A=A, split_factor=split_factor, near=near)

        if use_cross_person and num_person > 1:
            self.cross_person_attn = CrossPersonAttentionBlock(
                in_dim, num_heads, dropout, drop_path)
        else:
            self.use_cross_person = False

        if downsample:
            self.temporal_pool = TemporalPooling()

        self.need_proj = (in_dim != out_dim)
        if self.need_proj:
            self.proj_norm = nn.LayerNorm(in_dim)
            self.channel_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (N, M, T, V, D)
        N, M, T, V, D = x.shape

        # Joint spatio-temporal attention on (N*M, T, V, D)
        x_flat = x.reshape(N * M, T, V, D)
        x_flat = self.st_attn(x_flat)
        x = x_flat.reshape(N, M, T, V, -1)

        # Cross-person attention
        if self.use_cross_person:
            x = self.cross_person_attn(x)

        # Temporal downsampling (before channel proj so proj maps to new dim)
        if self.downsample:
            x = self.temporal_pool(x)

        if self.need_proj:
            x = self.channel_proj(self.proj_norm(x))

        return x


class FactorizedTransformerBlock(nn.Module):
    """One layer of factorized spatial-temporal-person attention.

    Spatial attention -> Temporal attention -> Cross-person attention (optional)
    -> Channel projection (when in_dim != out_dim).

    Input:  (N, M, T, V, in_dim)
    Output: (N, M, T, V, out_dim)
    """

    def __init__(self, in_dim, out_dim, num_heads, num_person=2, max_T=300,
                 dropout=0.0, drop_path=0.0,
                 use_graph_bias=False, A=None,
                 use_cross_person=True):
        super().__init__()
        self.use_cross_person = use_cross_person

        self.spatial_attn = SpatialAttentionBlock(
            in_dim, num_heads, dropout, drop_path,
            use_graph_bias, A=A)
        self.temporal_attn = TemporalAttentionBlock(
            in_dim, num_heads, max_T, dropout, drop_path)

        if use_cross_person and num_person > 1:
            self.cross_person_attn = CrossPersonAttentionBlock(
                in_dim, num_heads, dropout, drop_path)
        else:
            self.use_cross_person = False

        self.need_proj = (in_dim != out_dim)
        if self.need_proj:
            self.proj_norm = nn.LayerNorm(in_dim)
            self.channel_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (N, M, T, V, D)
        N, M, T, V, D = x.shape

        # Spatial + temporal: operate on (N*M, T, V, D)
        x_flat = x.reshape(N * M, T, V, D)
        x_flat = self.spatial_attn(x_flat)
        x_flat = self.temporal_attn(x_flat)
        x = x_flat.reshape(N, M, T, V, -1)

        # Cross-person attention: operate on (N, M, T, V, D)
        if self.use_cross_person:
            x = self.cross_person_attn(x)

        if self.need_proj:
            x = self.channel_proj(self.proj_norm(x))

        return x
