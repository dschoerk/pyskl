import copy as cp

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from ..transformers.attention import DropPath
from .stgcn import STGCNBlock

EPS = 1e-4


class StatisticalInvarianceCoding(nn.Module):
    """Statistical Invariance Encoding from BlockGCN (CVPR 2024).

    Computes the temporal mean of pairwise relative joint coordinates and
    encodes them into a compact statistical code via a two-layer MLP.

    Formally, for a block with temporal mean joint positions x̄ ∈ R^(V×C):
        r̄_ij = x̄_i - x̄_j           (pairwise relative coordinate)
        C_ij  = f_θ(r̄_ij)            (learned 2-layer MLP, R^C → R^D)

    The per-block code is obtained by mean-pooling C over all (i, j) pairs:
        code = mean_{i,j}(C_ij) ∈ R^D

    The code is then added to the GCN-pooled block feature vectors, giving
    each block an additional "static pose" bias that is robust to temporal
    noise (human joint movements are back-and-forth, so their mean is stable).

    Args:
        in_channels (int): Raw coordinate channels (e.g. 3 for xyz).
        out_channels (int): Output statistical code dimension (= GCN output
            channels so the code can be added directly to block features).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Temporal mean of raw joint coordinates,
               shape ``(N, V, C)``.
        Returns:
            Statistical code, shape ``(N, out_channels)``.
        """
        # Pairwise relative coordinates: r̄_ij = x̄_i - x̄_j
        r_ij = x.unsqueeze(2) - x.unsqueeze(1)   # (N, V, V, C)
        # Encode each edge and pool over all joint pairs
        C = self.mlp(r_ij)                        # (N, V, V, out_channels)
        return C.mean(dim=(1, 2))                  # (N, out_channels)


class BlockTransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer for inter-block attention.

    Attends across the sequence of block-level feature vectors produced by
    the STGCN++ stages.  Each block is a single token.

    Input/Output shape: (B, num_blocks, C)
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
        # Pre-norm self-attention
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        # Pre-norm FFN
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


@BACKBONES.register_module()
class BlockSTGCN(nn.Module):
    """Block-wise STGCN++ with inter-block transformer attention.

    Splits the temporal dimension into non-overlapping blocks of ``block_size``
    frames, applies STGCN++ to every block simultaneously (via the batch
    dimension), then uses a multi-head self-attention transformer to capture
    long-range temporal dependencies between the resulting block features.

    Pipeline
    --------
    1. Apply data BN on the full input ``(N, M, T, V, C)``.
    2. Run STGCN++ stages on the full temporal sequence
       ``(N*M, C, T, V)`` → ``(N*M, C_block, T', V')``.
    3. Mean-pool over the spatial dimension V' → ``(N*M, C_block, T')``.
    4. Split ``T'`` into ``num_blocks`` equal segments (pad if needed) and
       mean-pool each segment → ``(N*M, num_blocks, C_block)``.
       ``num_blocks`` is derived from the raw-frame ``block_size``.
    5. Optionally add SIC codes computed from per-block temporal means of
       the raw BN-normalised input.
    6. Add learnable temporal positional embeddings, then merge persons and
       blocks into one sequence ``(N, M*num_blocks, C_block)`` and run the
       transformer so it attends across both temporal blocks and persons.
    7. Apply final LayerNorm and return
       ``(N, M, C_block, num_blocks, 1)`` for ``GCNHead`` compatibility.

    Args:
        graph_cfg (dict): Config for skeleton ``Graph`` (layout, mode).
        block_size (int): Frames per temporal block. Default: 8.
        in_channels (int): Input coordinate channels (e.g. 3 for xyz).
            Default: 3.
        base_channels (int): STGCN++ base channel width. Default: 64.
        data_bn_type (str): ``'VC'``, ``'MVC'``, or ``'none'``. Default: ``'VC'``.
        ch_ratio (float): Channel expansion factor at inflate stages.
            Default: 2.
        num_person (int): Persons per sample (only used with ``'MVC'`` BN).
            Default: 2.
        num_stages (int): Number of STGCN++ stages. Default: 10.
        inflate_stages (list[int]): Stages where channels expand by
            ``ch_ratio``. Default: ``[5, 8]``.
        down_stages (list[int]): Stages with temporal stride 2.
            Default: ``[5, 8]``.
        num_heads (int): Attention heads in the inter-block transformer.
            Default: 4.
        num_transformer_layers (int): Depth of the inter-block transformer.
            Default: 2.
        transformer_dropout (float): Dropout inside attention & FFN.
            Default: 0.1.
        transformer_drop_path (float): Max stochastic-depth rate (linearly
            increases across layers). Default: 0.1.
        max_blocks (int): Maximum expected number of blocks, used to
            pre-allocate the learnable positional embedding table.
            Default: 64.
        use_sic (bool): Whether to add a Statistical Invariance Coding
            module (BlockGCN, CVPR 2024) that computes per-block temporal
            mean pairwise joint differences and adds the resulting code to
            the pooled block features. Default: True.
        pretrained (str | None): Path to a pretrained checkpoint.
            Default: None.
        **kwargs: Extra keyword arguments forwarded to each
            :class:`STGCNBlock` (e.g. ``gcn_adaptive``, ``gcn_with_res``,
            ``tcn_type``).
    """

    def __init__(self,
                 graph_cfg,
                 block_size=8,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 num_heads=4,
                 num_transformer_layers=2,
                 transformer_dropout=0.1,
                 transformer_drop_path=0.1,
                 max_blocks=64,
                 use_sic=True,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.block_size = block_size

        # ── Data batch-norm (applied once on the full temporal sequence) ──
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        # ── STGCN++ stages (identical construction logic to STGCN) ──────────
        lw_kwargs = [cp.deepcopy(kwargs) for _ in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(),
                                  stride=1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        out_channels = base_channels  # fallback if loop body never executes
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels  # reuse local var (same pattern as STGCN)
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(),
                                      stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        gcn_out_channels = out_channels  # channel dim coming out of GCN stages

        # ── Inter-block transformer ──────────────────────────────────────────
        # Learnable temporal and person positional embeddings
        self.block_pos_embed = nn.Parameter(
            torch.zeros(1, max_blocks, gcn_out_channels))
        self.person_pos_embed = nn.Parameter(
            torch.zeros(1, num_person, gcn_out_channels))
        self.num_person = num_person

        dp_rates = [
            transformer_drop_path * i / max(num_transformer_layers - 1, 1)
            for i in range(num_transformer_layers)
        ]
        self.transformer_layers = nn.ModuleList([
            BlockTransformerLayer(
                gcn_out_channels, num_heads,
                dropout=transformer_dropout,
                drop_path=dp_rates[i])
            for i in range(num_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(gcn_out_channels)
        self.out_channels = gcn_out_channels

        # ── Statistical Invariance Coding (BlockGCN, CVPR 2024) ─────────────
        # self.in_channels holds the raw coordinate channels (e.g. 3 for xyz)
        # set at line 197.
        self.use_sic = use_sic
        if use_sic:
            self.sic = StatisticalInvarianceCoding(self.in_channels, gcn_out_channels)

        self.pretrained = pretrained

    def init_weights(self):
        nn.init.trunc_normal_(self.block_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.person_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            self._load_pretrained(self.pretrained)

    def _load_pretrained(self, path):
        """Load a pretrained checkpoint, handling full-model saves.

        Full RecognizerGCN checkpoints store backbone weights under the
        ``backbone.*`` key prefix.  This method strips that prefix so the
        STGCN++ stages (``data_bn`` + ``gcn``) are initialised from the
        pretrained weights while the inter-block transformer layers are left
        with their random initialisation.
        """
        import logging
        logger = logging.getLogger(__name__)

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # Detect and strip the backbone prefix.
        # Plain RecognizerGCN save  → 'backbone.*'
        # torch.compile save        → '_orig_mod.backbone.*'
        for prefix in ('_orig_mod.backbone.', 'backbone.'):
            if any(k.startswith(prefix) for k in state_dict):
                state_dict = {
                    k[len(prefix):]: v
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }
                break

        # Filter to only keys whose shape matches (handles in_channels mismatch
        # when the pretrained model was trained with a different input modality,
        # e.g. in_channels=12 for j+b+jm+bm vs in_channels=3 for joints only).
        own_state = self.state_dict()
        compat = {k: v for k, v in state_dict.items()
                  if k in own_state and own_state[k].shape == v.shape}
        skipped = [k for k in state_dict if k in own_state
                   and own_state[k].shape != state_dict[k].shape]

        missing, _ = self.load_state_dict(compat, strict=False)
        # Keys truly absent from the checkpoint (not merely shape-mismatched)
        truly_missing_gcn = [k for k in missing
                             if k.startswith(('data_bn', 'gcn'))
                             and k not in skipped]

        if skipped:
            logger.warning(
                f'Skipped {len(skipped)} shape-mismatched key(s) '
                f'(pretrained in_channels differs): {skipped}')
        if truly_missing_gcn:
            logger.warning(
                f'GCN keys absent from checkpoint: {truly_missing_gcn}')
        else:
            logger.info(
                f'Loaded STGCN++ weights from {path}. '
                f'Randomly initialised: transformer layers'
                + (f' + shape-mismatched: {skipped}' if skipped else '')
            )

    def forward(self, x):
        N, M, T, V, C = x.size()

        # ── 1. Data BN (identical permutation pattern to STGCN) ─────────────
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, V, C, T)
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = (x.view(N, M, V, C, T)
              .permute(0, 1, 3, 4, 2)                # (N, M, C, T, V)
              .contiguous()
              .view(N * M, C, T, V))                  # (N*M, C, T, V)

        # ── SIC: save BN-normalised raw coords for per-block stats ───────────
        if self.use_sic:
            x_raw = x                                # (N*M, C, T, V)

        # ── 2. Determine num_blocks from raw-frame block_size ────────────────
        bs = self.block_size
        pad_raw = (bs - T % bs) % bs
        num_blocks = (T + pad_raw) // bs

        # ── 3. Run GCN stages on full temporal sequence ──────────────────────
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        # x: (N*M, C_block, T', V')

        # ── 4. Mean-pool over spatial (V') dimension ─────────────────────────
        x = x.mean(dim=-1)                           # (N*M, C_block, T')
        T_prime = x.shape[2]

        # ── 5. Split T' into num_blocks segments, mean-pool each ─────────────
        pad_out = (num_blocks - T_prime % num_blocks) % num_blocks
        if pad_out > 0:
            x = F.pad(x, (0, pad_out))
        seg_len = (T_prime + pad_out) // num_blocks
        x = x.view(N * M, x.shape[1], num_blocks, seg_len).mean(dim=-1)
        x = x.permute(0, 2, 1).contiguous()         # (N*M, num_blocks, C_block)

        # ── SIC: per-block temporal mean from raw BN-normalised input ────────
        if self.use_sic:
            if pad_raw > 0:
                x_raw = F.pad(x_raw, (0, 0, 0, pad_raw))
            x_mean = (x_raw
                      .view(N * M, C, num_blocks, bs, V)
                      .mean(dim=3)                   # (N*M, C, num_blocks, V)
                      .permute(0, 2, 3, 1)           # (N*M, num_blocks, V, C)
                      .contiguous()
                      .view(N * M * num_blocks, V, C))
            sic_codes = self.sic(x_mean)             # (N*M*num_blocks, C_block)
            x = x + sic_codes.view(N * M, num_blocks, -1)

        # ── 6-7. Positional embeddings + inter-block/person transformer ─────
        if self.transformer_layers:
            x = x + self.block_pos_embed[:, :num_blocks, :]   # (N*M, num_blocks, C_block)
            x = x.view(N, M, num_blocks, -1)
            x = x + self.person_pos_embed[:, :M, :].unsqueeze(2)  # (1, M, 1, C_block)
            x = x.view(N, M * num_blocks, -1)                     # (N, M*num_blocks, C_block)
            for layer in self.transformer_layers:
                x = layer(x)
            x = x.view(N * M, num_blocks, -1)                     # (N*M, num_blocks, C_block)

        # ── 8. Final norm & reshape to GCNHead format (N, M, C, T, V) ───────
        x = self.final_norm(x)                           # (N*M, num_blocks, C_block)
        x = x.view(N, M, num_blocks, x.shape[-1])
        # (N, M, C_block, num_blocks, 1)  ←  T=num_blocks, V=1 for GCNHead
        x = x.permute(0, 1, 3, 2).unsqueeze(-1).contiguous()
        return x
