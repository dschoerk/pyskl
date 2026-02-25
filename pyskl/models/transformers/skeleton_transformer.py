import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .attention import JointTransformerBlock


@BACKBONES.register_module()
class SkeletonTransformer(nn.Module):
    """Transformer backbone for skeleton-based action recognition.

    Each token represents one joint at one frame (T*V tokens per person).
    A single joint spatio-temporal attention block attends over the full
    T*V sequence with an additive bias decomposed into temporal relative
    position bias and spatial graph bias.

    Supports a temporal pyramid: `down_stages` specifies block indices
    after which T is halved (via avg-pool).  Pair with `expand_stages`
    at the same indices to double channels and keep information capacity.

    Output shape: (N, M, C, T', V) — compatible with GCNHead.

    Args:
        graph_cfg (dict): Config for the skeleton graph (layout, mode).
        in_channels (int): Input feature channels per joint (e.g. 3 for xyz).
        embed_dim (int): Transformer embedding dimension.
        head_dim (int): Dimension per attention head. num_heads = dim // head_dim
            and scales automatically when channels expand.
        depth (int): Number of transformer blocks.
        num_person (int): Max number of persons. Default: 2.
        temporal_patch_size (int): Group this many consecutive frames into the
            channel dimension before projection. Input T must be divisible by
            this value. Effective T becomes T // temporal_patch_size.
            Default: 1 (no grouping).
        expand_stages (list[int]): Block indices at which to double channels.
        down_stages (list[int]): Block indices at which to halve T.
        max_T (int): Maximum temporal length for relative position bias.
        dropout (float): Attention / FFN dropout rate.
        drop_path (float): Stochastic depth rate (max, linearly ramped).
        use_graph_bias (bool): Use adjacency matrix as spatial attention bias.
        use_cross_person (bool): Use cross-person attention blocks.
        pretrained (str | None): Path to pretrained checkpoint.
    """

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 embed_dim=64,
                 head_dim=16,
                 depth=4,
                 num_person=2,
                 temporal_patch_size=1,
                 expand_stages=None,
                 down_stages=None,
                 max_T=300,
                 dropout=0.0,
                 drop_path=0.1,
                 use_graph_bias=True,
                 use_cross_person=True,
                 pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.temporal_patch_size = temporal_patch_size

        graph = Graph(**graph_cfg)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer('A', A)
        num_joints = A.shape[1]

        if expand_stages is None:
            expand_stages = []
        if down_stages is None:
            down_stages = []
        expand_set = set(expand_stages)
        down_set = set(down_stages)

        # Input projection: per-joint features -> embed_dim
        # With temporal patching, in_channels is multiplied by patch_size
        self.input_proj = nn.Linear(in_channels * temporal_patch_size, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Learnable per-joint embedding (spatial identity for each joint)
        self.joint_embed = nn.Parameter(torch.zeros(1, 1, 1, num_joints, embed_dim))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        # Build transformer blocks with optional channel expansion + temporal pooling
        self.blocks = nn.ModuleList()
        cur_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        for i in range(depth):
            out_dim = cur_dim * 2 if i in expand_set else cur_dim
            num_heads = cur_dim // head_dim
            self.blocks.append(
                JointTransformerBlock(
                    in_dim=cur_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    num_person=num_person,
                    max_T=max_T,
                    dropout=dropout,
                    drop_path=dpr[i],
                    use_graph_bias=use_graph_bias,
                    A=A,
                    use_cross_person=use_cross_person,
                    downsample=(i in down_set),
                ))
            cur_dim = out_dim

        self.out_dim = cur_dim
        self.final_norm = nn.LayerNorm(cur_dim)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x, mask=None, mask_token=None):
        # x: (N, M, T, V, C) from dataloader
        N, M, T, V, C = x.shape

        # Temporal patching: group consecutive frames into channel dim
        p = self.temporal_patch_size
        if p > 1:
            assert T % p == 0, f'T={T} must be divisible by temporal_patch_size={p}'
            # (N, M, T, V, C) -> (N, M, T//p, p, V, C) -> (N, M, T//p, V, p*C)
            x = x.reshape(N, M, T // p, p, V, C).permute(0, 1, 2, 4, 3, 5)
            x = x.reshape(N, M, T // p, V, p * C)
            T = T // p

        # Project input features and add joint identity embedding
        x = self.input_proj(x)                    # (N, M, T, V, D)
        x = x + self.joint_embed[:, :, :, :V, :]  # broadcast over N, M, T
        x = self.input_norm(x)

        # MAE: replace masked positions with learnable mask token
        if mask is not None and mask_token is not None:
            x = torch.where(mask.unsqueeze(-1), mask_token, x)

        # Transformer blocks (T may shrink at down_stages)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Reshape to (N, M, C_out, T', V) for GCNHead compatibility
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x
