import torch
import torch.nn as nn


class SpatialAttentionBlock(nn.Module):
    """Multi-head self-attention across joints within each frame.

    Input:  (B, T, V, D)
    Output: (B, T, V, D)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0,
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
            # nn.MultiheadAttention accepts float attn_mask as additive bias
            # Shape: (B*T*num_heads, V, V)
            attn_mask = self.graph_bias.unsqueeze(0).expand(B * T, -1, -1, -1)
            attn_mask = attn_mask.reshape(B * T * self.num_heads, V, V)

        x_flat = residual + self.attn(
            x_flat, x_flat, x_flat, attn_mask=attn_mask)[0]

        # Pre-norm + FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        return x_flat.reshape(B, T, V, D)


class TemporalAttentionBlock(nn.Module):
    """Multi-head self-attention across frames for each joint.

    Input:  (B, T, V, D)
    Output: (B, T, V, D)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
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

    def forward(self, x):
        B, T, V, D = x.shape
        # Transpose to (B*V, T, D) for temporal attention
        x_flat = x.permute(0, 2, 1, 3).reshape(B * V, T, D)

        # Pre-norm + MHSA
        residual = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = residual + self.attn(x_flat, x_flat, x_flat)[0]

        # Pre-norm + FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        return x_flat.reshape(B, V, T, D).permute(0, 2, 1, 3)


class FactorizedTransformerBlock(nn.Module):
    """One layer of factorized spatial-temporal attention.

    Spatial attention first, then temporal attention.
    When in_dim != out_dim, applies a linear projection for channel expansion.

    Input:  (B, T, V, in_dim)
    Output: (B, T, V, out_dim)
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0,
                 use_graph_bias=False, num_nodes=25):
        super().__init__()
        self.spatial_attn = SpatialAttentionBlock(
            in_dim, num_heads, dropout, use_graph_bias, num_nodes)
        self.temporal_attn = TemporalAttentionBlock(
            in_dim, num_heads, dropout)

        self.need_proj = (in_dim != out_dim)
        if self.need_proj:
            self.proj_norm = nn.LayerNorm(in_dim)
            self.channel_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        if self.need_proj:
            x = self.channel_proj(self.proj_norm(x))
        return x
