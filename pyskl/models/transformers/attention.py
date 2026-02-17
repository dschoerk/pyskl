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
                 use_graph_bias=False, num_nodes=25):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
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
            self.graph_bias = nn.Parameter(
                torch.zeros(num_heads, num_nodes, num_nodes))

    def forward(self, x):
        B, T, V, D = x.shape
        x_flat = x.reshape(B * T, V, D)

        # Pre-norm + MHSA
        residual = x_flat
        x_flat = self.norm1(x_flat)

        attn_mask = None
        if self.use_graph_bias:
            attn_mask = self.graph_bias.unsqueeze(0).expand(B * T, -1, -1, -1)
            attn_mask = attn_mask.reshape(B * T * self.num_heads, V, V)

        x_flat = residual + self.drop_path(
            self.attn(x_flat, x_flat, x_flat, attn_mask=attn_mask)[0])

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
        self.num_heads = num_heads
        self.max_T = max_T
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
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
        # relative_position[i, j] = i - j, shifted to non-negative index
        rel_pos = coords.unsqueeze(1) - coords.unsqueeze(0)  # (T, T)
        rel_pos = rel_pos + self.max_T - 1  # shift so min index is 0
        bias = self.rel_pos_bias[:, rel_pos]  # (num_heads, T, T)
        return bias

    def forward(self, x):
        B, T, V, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * V, T, D)

        # Build relative position bias: (num_heads, T, T) -> (B*V*num_heads, T, T)
        rel_bias = self._get_rel_pos_bias(T)
        rel_bias = rel_bias.unsqueeze(0).expand(B * V, -1, -1, -1)
        rel_bias = rel_bias.reshape(B * V * self.num_heads, T, T)

        # Pre-norm + MHSA with relative position bias
        residual = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = residual + self.drop_path(
            self.attn(x_flat, x_flat, x_flat, attn_mask=rel_bias)[0])

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
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
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
        # Reshape so each (frame, joint) pair attends across persons
        # (N*T*V, M, D)
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(N * T * V, M, D)

        residual = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = residual + self.drop_path(
            self.attn(x_flat, x_flat, x_flat)[0])

        x_flat = x_flat + self.drop_path(self.ffn(self.norm2(x_flat)))

        return x_flat.reshape(N, T, V, M, D).permute(0, 3, 1, 2, 4)


class FactorizedTransformerBlock(nn.Module):
    """One layer of factorized spatial-temporal-person attention.

    Spatial attention -> Temporal attention -> Cross-person attention (optional)
    -> Channel projection (when in_dim != out_dim).

    Input:  (N, M, T, V, in_dim)
    Output: (N, M, T, V, out_dim)
    """

    def __init__(self, in_dim, out_dim, num_heads, num_person=2, max_T=300,
                 dropout=0.0, drop_path=0.0,
                 use_graph_bias=False, num_nodes=25,
                 use_cross_person=True):
        super().__init__()
        self.use_cross_person = use_cross_person

        self.spatial_attn = SpatialAttentionBlock(
            in_dim, num_heads, dropout, drop_path,
            use_graph_bias, num_nodes)
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
