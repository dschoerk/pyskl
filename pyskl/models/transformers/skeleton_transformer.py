import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .attention import FactorizedTransformerBlock


@BACKBONES.register_module()
class SkeletonTransformer(nn.Module):
    """Factorized Spatial-Temporal Transformer for skeleton action recognition.

    Alternates spatial attention (across joints per frame) and temporal
    attention (across frames per joint) in each transformer block.

    Args:
        graph_cfg (dict): Config for skeleton Graph (layout, mode).
        in_channels (int): Input coordinate channels (e.g. 3 for xyz).
        embed_dim (int): Base embedding dimension.
        num_heads (int): Number of attention heads.
        depth (int): Number of factorized transformer blocks.
        expand_stages (list[int]): Block indices where channels double.
        dropout (float): Dropout rate.
        use_graph_bias (bool): Use skeleton adjacency as spatial attention bias.
        data_bn_type (str): Data batch norm type ('VC', 'MVC', or 'none').
        num_person (int): Number of persons (used when data_bn_type='MVC').
        max_T (int): Maximum temporal length for positional encoding.
        pretrained (str | None): Pretrained checkpoint path.
    """

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 embed_dim=64,
                 num_heads=8,
                 depth=8,
                 expand_stages=[3, 6],
                 dropout=0.1,
                 use_graph_bias=True,
                 data_bn_type='VC',
                 num_person=2,
                 max_T=300,
                 pretrained=None):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        num_nodes = self.graph.num_node
        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)

        self.data_bn_type = data_bn_type
        self.use_graph_bias = use_graph_bias
        self.pretrained = pretrained

        # Data batch normalization (same pattern as STGCN)
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        else:
            self.data_bn = nn.Identity()

        # Input projection from coordinates to embedding space
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Learnable factorized positional encodings
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, num_nodes, embed_dim))
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_T, 1, embed_dim))

        # Build transformer blocks with channel expansion
        self.blocks = nn.ModuleList()
        current_dim = embed_dim
        for i in range(depth):
            in_dim = current_dim
            if i in expand_stages:
                out_dim = current_dim * 2
                current_dim = out_dim
            else:
                out_dim = current_dim
            self.blocks.append(FactorizedTransformerBlock(
                in_dim, out_dim, num_heads, dropout,
                use_graph_bias, num_nodes))

        self.out_channels = current_dim
        self.final_norm = nn.LayerNorm(current_dim)

        # Store adjacency for graph bias initialization
        if use_graph_bias:
            self.register_buffer('A', A)

    def init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Initialize graph bias from skeleton adjacency
        if self.use_graph_bias:
            # A has shape (K, V, V) with K subsets; sum to get connectivity
            connectivity = (self.A.sum(0) > 0).float()
            for block in self.blocks:
                bias = block.spatial_attn.graph_bias
                nn.init.zeros_(bias)
                # Set connected joints to small positive bias across all heads
                for h in range(bias.shape[0]):
                    bias.data[h] = connectivity * 0.1

        # Standard init for linear and layernorm
        self.apply(self._init_weights)

        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        N, M, T, V, C = x.size()

        # Data batch normalization (identical to STGCN pattern)
        x = x.permute(0, 1, 3, 4, 2).contiguous()       # (N, M, V, C, T)
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 4, 2, 3).contiguous()       # (N, M, T, V, C)

        # Merge batch and person dimensions
        x = x.view(N * M, T, V, C)                       # (N*M, T, V, C)

        # Project to embedding dimension
        x = self.input_proj(x)                            # (N*M, T, V, D)

        # Add factorized positional encodings
        x = x + self.spatial_pos_embed[:, :, :V, :]
        x = x + self.temporal_pos_embed[:, :T, :, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)                                  # (N*M, T, V, D_i)

        x = self.final_norm(x)                            # (N*M, T, V, C')

        # Reshape to pyskl backbone output format: (N, M, C', T, V)
        C_out = x.shape[-1]
        x = x.permute(0, 3, 1, 2)                        # (N*M, C', T, V)
        x = x.reshape(N, M, C_out, T, V)                 # (N, M, C', T, V)
        return x
