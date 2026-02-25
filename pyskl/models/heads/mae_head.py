import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from ..transformers.attention import JointSpatioTemporalBlock
from ...utils import Graph


@HEADS.register_module()
class MAEHead(nn.Module):
    """Lightweight decoder for MAE-style skeleton reconstruction.

    Takes encoder features (possibly at reduced temporal resolution due to
    temporal pooling), upsamples T back to the original resolution, runs
    a few transformer blocks, and projects back to input coordinate space.
    Loss is computed only at masked positions.

    Args:
        encoder_dim (int): Channel dimension from the encoder output.
        decoder_dim (int): Internal dimension of decoder transformer blocks.
        decoder_depth (int): Number of decoder transformer blocks.
        decoder_heads (int): Number of attention heads in decoder.
        in_channels (int): Original input channels to reconstruct (e.g. 3 for xyz).
        target_T (int): Original temporal length to reconstruct (before pooling).
        max_T (int): Maximum temporal length for relative position bias.
        use_graph_bias (bool): Use graph bias in decoder attention.
        graph_cfg (dict | None): Graph config to build adjacency for graph bias.
    """

    def __init__(self,
                 encoder_dim,
                 decoder_dim=128,
                 decoder_depth=2,
                 decoder_heads=4,
                 in_channels=3,
                 target_T=32,
                 max_T=300,
                 use_graph_bias=True,
                 graph_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.target_T = target_T

        # Build adjacency matrix for graph bias in decoder
        A = None
        if use_graph_bias and graph_cfg is not None:
            graph = Graph(**graph_cfg)
            A = torch.tensor(graph.A, dtype=torch.float32)

        # Project from encoder dim to decoder dim
        self.dec_proj = nn.Linear(encoder_dim, decoder_dim)

        # Lightweight transformer decoder blocks (operate at full T resolution)
        self.dec_blocks = nn.ModuleList([
            JointSpatioTemporalBlock(
                decoder_dim, decoder_heads, max_T,
                dropout=0.0, drop_path=0.0,
                use_graph_bias=use_graph_bias, A=A)
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(decoder_dim)

        # Output projection back to input coordinate space
        self.output_proj = nn.Linear(decoder_dim, in_channels)

    def init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, feat, target, mask):
        """Decode encoder features and compute reconstruction loss.

        Args:
            feat (torch.Tensor): Encoder output (N, M, C_enc, T_enc, V).
                T_enc may be smaller than target_T due to temporal pooling.
            target (torch.Tensor): Reconstruction target (N, M, T_orig, V, C_in).
            mask (torch.Tensor): Boolean mask (N, M, T_orig, V), True = masked.

        Returns:
            torch.Tensor: Scalar MSE reconstruction loss at masked positions.
        """
        # Unpermute encoder output to (N, M, T_enc, V, C_enc)
        x = feat.permute(0, 1, 3, 4, 2)
        N, M, T_enc, V, C_enc = x.shape
        T_orig = target.shape[2]

        # Flatten person dim into batch
        x = x.reshape(N * M, T_enc, V, C_enc)

        # Project to decoder dim
        x = self.dec_proj(x)  # (N*M, T_enc, V, decoder_dim)

        # Upsample T back to original resolution if needed
        if T_enc < T_orig:
            # Interpolate along T: (N*M, T_enc, V, D) -> (N*M, T_orig, V, D)
            D = x.shape[-1]
            # Reshape to (N*M*V, D, T_enc) for 1D interpolation along T
            x = x.permute(0, 2, 3, 1).reshape(N * M * V, D, T_enc)
            x = F.interpolate(x, size=T_orig, mode='linear', align_corners=False)
            x = x.reshape(N * M, V, D, T_orig).permute(0, 3, 1, 2)  # (N*M, T_orig, V, D)

        # Decoder transformer blocks at full resolution
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        pred = self.output_proj(x)  # (N*M, T_orig, V, C_in)

        # Reshape back
        pred = pred.reshape(N, M, T_orig, V, self.in_channels)

        # MSE loss only on masked positions
        loss = F.mse_loss(pred[mask], target[mask])
        return loss
