"""
PoseTransformerBackbone and PoseSTGCNBackbone — pyskl-compatible wrappers
around the PoseTransformer and PoseSTGCN models from posedrop.

Backbone input/output convention (matches pyskl's RecognizerGCN + GCNHead):
  Input:  (N, M, T, V, C)  — batch, persons, frames, joints, coord-channels
  Output: (N, M, C', T', V')  — with C' the embedding dim, T' and V' depending
          on the model (may be 1 for globally-pooled models).

Models for pose-based action recognition.

PoseTransformer: Transformer with five missing-data strategies:
  - zero_fill:       zero out missing keypoint features
  - interpolation:   linearly interpolate missing keypoints from temporal neighbors
  - token_drop:      remove missing tokens; use attention mask for variable-length
  - mask_token:      replace missing keypoints with a learned [MASK] embedding
  - mae_mask_token:  same as mask_token, but model is first pretrained with MAE

PoseSTGCN: Spatial-Temporal Graph Convolutional Network baseline.
  Always applies zero-fill for missing keypoints. The skeleton graph encodes
  joint connectivity; ST-GCN blocks alternate spatial graph conv with temporal conv.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES


# ---------------------------------------------------------------------------
# Skeleton graph definitions
# ---------------------------------------------------------------------------

# COCO-17 anatomical edges
_COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6), (0, 5), (0, 6),                   # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# NTU RGB+D 25-joint Kinect edges
_NTU25_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

# NTU 19-joint (after dropping 6 noisy extremities via NTU_REDUCED_JOINTS)
# Joints kept: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,20]
# Original→new: 16→15, 17→16, 18→17, 20→18
_NTU19_EDGES = [
    (0, 1), (1, 18), (18, 2), (2, 3),
    (18, 4), (4, 5), (5, 6), (6, 7),
    (18, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17),
]

_EDGE_MAP = {17: _COCO17_EDGES, 25: _NTU25_EDGES, 19: _NTU19_EDGES}

# Parent joint for each joint in a BFS-rooted spanning tree (root = joint 0).
# parents[j] = index of j's parent; -1 for the root.
# Used to compute bone vectors: bone[j] = pose[j] - pose[parents[j]].
_COCO17_PARENTS = [-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
_NTU25_PARENTS  = [-1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
# NTU-19: NTU-25 with joints [15,19,21,22,23,24] dropped, indices remapped.
_NTU19_PARENTS  = [-1, 0, 18, 2, 18, 4, 5, 6, 18, 8, 9, 10, 0, 12, 13, 0, 15, 16, 1]

_PARENT_MAP = {17: _COCO17_PARENTS, 25: _NTU25_PARENTS, 19: _NTU19_PARENTS}


def _build_parents(num_keypoints: int) -> torch.Tensor:
    """Return a (K,) int64 tensor of parent indices (-1 = root)."""
    if num_keypoints in _PARENT_MAP:
        return torch.tensor(_PARENT_MAP[num_keypoints], dtype=torch.long)
    # Fallback: linear chain rooted at joint 0
    parents = torch.arange(num_keypoints, dtype=torch.long) - 1
    parents[0] = -1
    return parents


def _build_adjacency(num_keypoints: int) -> torch.Tensor:
    """
    Return a symmetrically normalised adjacency matrix A_hat (K, K).

    Uses a known skeleton topology when available; falls back to a linear
    chain graph for unrecognised joint counts.
    """
    edges = _EDGE_MAP.get(num_keypoints)
    if edges is None:
        edges = [(i, i + 1) for i in range(num_keypoints - 1)]

    A = torch.zeros(num_keypoints, num_keypoints)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += torch.eye(num_keypoints)  # self-loops

    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = A.sum(dim=1).pow(-0.5)
    D_inv_sqrt[D_inv_sqrt == float("inf")] = 0.0
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)


# ---------------------------------------------------------------------------
# ST-GCN building blocks
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):
    """
    One Spatial-Temporal GCN block.

    Spatial: aggregate neighbour features via the normalised adjacency A,
             then mix channels with a 1×1 conv.
    Temporal: conv along the time axis with a (t_kernel × 1) kernel.
    Residual: identity or channel-matching 1×1 conv.

    Input / output shape: (B, C, T, K)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        t_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("A", A)  # (K, K)

        pad = (t_kernel - 1) // 2
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(t_kernel, 1),
                stride=(stride, 1),
                padding=(pad, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial graph aggregation: (B, C, T, K) × (K, K) → (B, C, T, K)
        x_agg = torch.einsum("bctv,vw->bctw", x, self.A)
        x_s = self.spatial(x_agg)
        return self.relu(self.temporal(x_s) + self.residual(x))


# ---------------------------------------------------------------------------
# PoseSTGCN
# ---------------------------------------------------------------------------

class PoseSTGCN(nn.Module):
    """
    ST-GCN baseline for pose-based action recognition.

    Missing keypoints are zero-filled before processing; the graph topology
    then propagates information from present neighbours.

    Input:  pose (B, T, K, D), mask (B, T, K) bool — True = present
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int = 17,
        keypoint_dim: int = 2,
        base_channels: int = 64,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        A = _build_adjacency(num_keypoints)  # (K, K)

        # Channel schedule: double every 2 blocks (capped at 4×base)
        _mults = [1, 1, 2, 2, 4, 4]
        if num_layers <= len(_mults):
            mults = _mults[:num_layers]
        else:
            mults = _mults + [4] * (num_layers - len(_mults))
        channels = [base_channels * m for m in mults]

        self.input_bn = nn.BatchNorm2d(keypoint_dim)

        in_ch = keypoint_dim
        self.blocks = nn.ModuleList()
        for out_ch in channels:
            self.blocks.append(STGCNBlock(in_ch, out_ch, A, dropout=dropout))
            in_ch = out_ch

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C_last, 1, 1)
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward_features(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run ST-GCN blocks and return feature maps before the classification head.

        Args:
            pose: (B, T, K, D)
            mask: (B, T, K) bool — True = present
        Returns:
            (B, C_last, T, K) feature tensor
        """
        pose = pose.clone()
        pose[~mask.unsqueeze(-1).expand_as(pose)] = 0.0

        # (B, T, K, D) → (B, D, T, K)
        x = pose.permute(0, 3, 1, 2)
        x = self.input_bn(x)

        for block in self.blocks:
            x = block(x)

        return x  # (B, C_last, T, K)

    def forward(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (B, T, K, D)
            mask: (B, T, K) bool — True = present
        Returns:
            logits: (B, num_classes)
        """
        x = self.forward_features(pose, mask)
        return self.head(x)


# ---------------------------------------------------------------------------
# Factored space-time attention blocks
# ---------------------------------------------------------------------------

class FactoredSTBlock(nn.Module):
    """
    One block of divided space-time attention (TimeSformer-style).

    Spatial attention:  for each frame independently, attends over K keypoints.
                        Reshape (B, T, K, d) → (B*T, K, d), run MHA, reshape back.
    Temporal attention: for each keypoint independently, attends over T frames.
                        Reshape (B, T, K, d) → (B*K, T, d), run MHA, reshape back.

    Complexity per block: O(T·K² + K·T²)  vs  O((T·K)²) for full attention.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        layer_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.spatial_layer  = nn.TransformerEncoderLayer(**layer_kwargs)
        self.temporal_layer = nn.TransformerEncoderLayer(**layer_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        spatial_mask:  Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:             (B, T, K, d_model)
            spatial_mask:  (B, T, K) bool  — True = present keypoint
            temporal_mask: (B, T, K) bool  — True = present keypoint
        Returns:
            (B, T, K, d_model)
        """
        B, T, K, d = x.shape

        # ---- Spatial: attend over K keypoints within each frame ----
        xs = x.reshape(B * T, K, d)

        if spatial_mask is not None:
            # key_padding_mask convention: True = IGNORE
            spad = ~spatial_mask.reshape(B * T, K)             # (B*T, K)
            # Rows where every keypoint is absent would cause NaN in softmax.
            # Clear the mask for those rows so attention runs on zeros instead;
            # then zero out their outputs. Using masked_fill avoids data-dependent
            # indexing (xs[bool_idx]), which breaks torch.compile.
            all_absent = spad.all(dim=1, keepdim=True)         # (B*T, 1)
            spad = spad & ~all_absent                           # (B*T, K) — no all-masked rows
            xs = self.spatial_layer(xs, src_key_padding_mask=spad)
            xs = xs.masked_fill(all_absent.unsqueeze(-1), 0.0) # (B*T, 1, 1) broadcasts over K, d
        else:
            xs = self.spatial_layer(xs)

        x = xs.reshape(B, T, K, d)

        # ---- Temporal: attend over T frames for each keypoint ----
        xt = x.permute(0, 2, 1, 3).reshape(B * K, T, d)       # (B*K, T, d)

        if temporal_mask is not None:
            tpad = ~temporal_mask.permute(0, 2, 1).reshape(B * K, T)  # (B*K, T)
            all_absent = tpad.all(dim=1, keepdim=True)          # (B*K, 1)
            tpad = tpad & ~all_absent
            xt = self.temporal_layer(xt, src_key_padding_mask=tpad)
            xt = xt.masked_fill(all_absent.unsqueeze(-1), 0.0)
        else:
            xt = self.temporal_layer(xt)

        x = xt.reshape(B, K, T, d).permute(0, 2, 1, 3)      # (B, T, K, d)
        return x


class FactoredSTEncoder(nn.Module):
    """
    Stack of FactoredSTBlocks with optional hierarchical temporal downsampling.

    temporal_pool_every > 0: after every N blocks, halve T via avg-pool.
    Example: 4 blocks, temporal_pool_every=2 → T→T/2 after block 1, T/2→T/4 after block 3.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        temporal_pool_every: int = 0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FactoredSTBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.temporal_pool_every = temporal_pool_every

    @staticmethod
    def _pool_temporal(
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """Halve the time axis: avg-pool features, OR-pool the presence mask."""
        B, T, K, d = x.shape

        # Features: (B, T, K, d) → (B*K, d, T) → avg_pool → (B, T/2, K, d)
        xp = x.permute(0, 2, 3, 1).reshape(B * K, d, T)
        xp = F.avg_pool1d(xp, kernel_size=2, stride=2, ceil_mode=True)
        T_new = xp.shape[-1]
        x = xp.reshape(B, K, d, T_new).permute(0, 3, 1, 2)   # (B, T_new, K, d)

        if mask is not None:
            # Mask: (B, T, K) → (B*K, 1, T) → max_pool → (B, T_new, K)
            mp = mask.permute(0, 2, 1).float().reshape(B * K, 1, T)
            mp = F.max_pool1d(mp, kernel_size=2, stride=2, ceil_mode=True)
            mask = (mp > 0).reshape(B, K, T_new).permute(0, 2, 1)

        return x, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pool: bool = True,
    ):
        """
        Args:
            x:    (B, T, K, d_model)
            mask: (B, T, K) bool — True = present (None = all present)
            pool: if False, skip temporal downsampling (used by MAE to keep T intact)
        Returns:
            x:    (B, T', K, d_model)  where T' ≤ T (reduced by pooling when pool=True)
            mask: (B, T', K) bool or None  (pooled in sync with x)
        """
        for i, block in enumerate(self.blocks):
            x = block(x, spatial_mask=mask, temporal_mask=mask)
            if pool and self.temporal_pool_every > 0 and (i + 1) % self.temporal_pool_every == 0:
                if x.shape[1] >= 2:
                    x, mask = self._pool_temporal(x, mask)
        return x, mask


class PoseTransformer(nn.Module):
    """
    Transformer for action recognition from pose sequences.

    Input:  pose (B, T, K, D), mask (B, T, K)
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int = 17,
        keypoint_dim: int = 2,
        num_frames: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        strategy: str = "zero_fill",
        use_factored_attention: bool = True,
        temporal_pool_every: int = 0,
        use_velocity: bool = False,
        use_bone: bool = False,
    ):
        super().__init__()
        self.strategy = strategy
        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.num_frames = num_frames
        self.d_model = d_model
        self.use_factored_attention = use_factored_attention
        self.use_velocity = use_velocity
        self.use_bone = use_bone

        # Register parent indices so they move to the right device automatically.
        self.register_buffer("parents", _build_parents(num_keypoints))  # (K,)

        # --- Input projection ---
        # Input streams: [coords] + [bone?] + [velocity?]
        n_streams = 1 + int(use_bone) + int(use_velocity)
        proj_in_dim = keypoint_dim * n_streams
        self.input_proj = nn.Linear(proj_in_dim, d_model)

        # --- Positional encodings ---
        # Temporal: sinusoidal buffer — generalises to any T without retraining.
        # Pre-compute table up to 2048 frames; look up by frame index at runtime.
        _max_len = 2048
        pe = torch.zeros(_max_len, d_model)
        pos = torch.arange(_max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("temporal_pe_table", pe)   # (max_len, d_model)

        # Spatial: learned — K is fixed per dataset, no generalisation needed
        self.spatial_pe = nn.Embedding(num_keypoints, d_model)

        # --- Special tokens ---
        if not use_factored_attention:
            # CLS token only needed for the flat-attention path (mean-pool for factored)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if strategy in ("mask_token", "mae_mask_token"):
            self.mask_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Encoder ---
        if use_factored_attention:
            self.encoder = FactoredSTEncoder(
                num_layers=num_layers,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                temporal_pool_every=temporal_pool_every,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Classification head ---
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # --- MAE reconstruction head (only used during pretraining) ---
        if strategy == "mae_mask_token":
            self.mae_decoder = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, keypoint_dim),
            )

    @staticmethod
    def _compute_velocity(pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Per-frame velocity: pose[t] - pose[t-1], zeroed where either frame is missing.

        Args:
            pose: (B, T, K, D) — should be zero-filled at missing positions
            mask: (B, T, K) bool — True = present
        Returns:
            (B, T, K, D) velocity tensor
        """
        vel = torch.zeros_like(pose)
        vel[:, 1:] = pose[:, 1:] - pose[:, :-1]
        prev_present = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
        vel_valid = (mask & prev_present).unsqueeze(-1)
        return vel * vel_valid.float()

    @staticmethod
    def _compute_bone(
        pose: torch.Tensor, mask: torch.Tensor, parents: torch.Tensor
    ) -> torch.Tensor:
        """
        Bone vectors: pose[j] - pose[parent(j)], zeroed where either joint is missing.
        Root joint bone is always zero.

        Args:
            pose:    (B, T, K, D) — should be zero-filled at missing positions
            mask:    (B, T, K) bool — True = present
            parents: (K,) long — parent index per joint, -1 for root
        Returns:
            (B, T, K, D) bone tensor
        """
        parent_idx = parents.clamp(min=0)          # (K,)  root: -1 → 0
        is_root = (parents < 0)                    # (K,)  bool

        parent_pose = pose[:, :, parent_idx]       # (B, T, K, D)
        bone = pose - parent_pose
        bone[:, :, is_root] = 0.0                  # root has no meaningful bone

        parent_mask = mask[:, :, parent_idx]       # (B, T, K)
        bone_valid = mask & parent_mask & (~is_root.view(1, 1, -1))
        return bone * bone_valid.unsqueeze(-1).float()

    def _build_features(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Concatenate requested feature streams: [coords, bone?, velocity?].

        pose must already be zero-filled / interpolated at missing positions.

        Args:
            pose: (B, T, K, D)
            mask: (B, T, K) bool — True = present (used to zero edges of derived features)
        Returns:
            (B, T, K, D * n_streams)
        """
        parts = [pose]
        if self.use_bone:
            parts.append(self._compute_bone(pose, mask, self.parents))
        if self.use_velocity:
            parts.append(self._compute_velocity(pose, mask))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else pose

    def _tokenize_and_embed(self, pose, mask):
        """
        Convert (B, T, K, D) pose into a sequence of token embeddings
        using the selected missing-data strategy.

        Returns:
            tokens:   (B, S, d_model)  — S varies by strategy
            attn_mask: (B, S) bool     — True = attend, False = ignore
            frame_ids: (B, S) long     — frame index per token (for MAE)
            kp_ids:    (B, S) long     — keypoint index per token (for MAE)
        """
        B, T, K, D = pose.shape
        device = pose.device

        # Frame and keypoint index grids: (T, K)
        frame_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, K)
        kp_idx = torch.arange(K, device=device).unsqueeze(0).expand(T, K)

        flat_fi = frame_idx.reshape(-1)   # (T*K,) frame indices, for PE lookup
        flat_ki = kp_idx.reshape(-1)      # (T*K,) keypoint indices

        if self.strategy == "zero_fill":
            pose = pose.clone()
            pose[~mask.unsqueeze(-1).expand_as(pose)] = 0.0
            feat = self._build_features(pose, mask)
            tokens = self.input_proj(feat.view(B, T * K, -1))
            t_pe = self.temporal_pe_table[flat_fi].unsqueeze(0)
            s_pe = self.spatial_pe(flat_ki).unsqueeze(0)
            tokens = tokens + t_pe + s_pe
            attn_mask = torch.ones(B, T * K, dtype=torch.bool, device=device)
            f_ids = flat_fi.unsqueeze(0).expand(B, -1)
            k_ids = flat_ki.unsqueeze(0).expand(B, -1)

        elif self.strategy == "interpolation":
            # Interpolate first; bone/velocity are then derived from filled positions.
            pose = self._interpolate_missing(pose, mask)
            all_present = torch.ones_like(mask)
            feat = self._build_features(pose, all_present)
            tokens = self.input_proj(feat.view(B, T * K, -1))
            t_pe = self.temporal_pe_table[flat_fi].unsqueeze(0)
            s_pe = self.spatial_pe(flat_ki).unsqueeze(0)
            tokens = tokens + t_pe + s_pe
            attn_mask = torch.ones(B, T * K, dtype=torch.bool, device=device)
            f_ids = flat_fi.unsqueeze(0).expand(B, -1)
            k_ids = flat_ki.unsqueeze(0).expand(B, -1)

        elif self.strategy == "token_drop":
            # Zero-fill so bone/velocity of present joints use consistent coords.
            pose = pose.clone()
            pose[~mask.unsqueeze(-1).expand_as(pose)] = 0.0
            feat = self._build_features(pose, mask)   # (B, T, K, n_streams*D)
            feat_flat = feat.reshape(B, T * K, -1)

            all_tokens, all_attn, all_f, all_k = [], [], [], []
            max_len = 0
            for b in range(B):
                present = mask[b].reshape(-1)  # (T*K,)
                idx = present.nonzero(as_tuple=False).squeeze(-1)
                n = idx.shape[0]
                max_len = max(max_len, n)

                tok = self.input_proj(feat_flat[b][idx])  # (n, d_model)
                f_id = flat_fi[idx]
                k_id = flat_ki[idx]
                tok = tok + self.temporal_pe_table[f_id] + self.spatial_pe(k_id)

                all_tokens.append(tok)
                all_attn.append(torch.ones(n, dtype=torch.bool, device=device))
                all_f.append(f_id)
                all_k.append(k_id)

            # Pad to max_len
            tokens = torch.zeros(B, max_len, self.d_model, device=device)
            attn_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
            f_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
            k_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
            for b in range(B):
                n = all_tokens[b].shape[0]
                tokens[b, :n] = all_tokens[b]
                attn_mask[b, :n] = True
                f_ids[b, :n] = all_f[b]
                k_ids[b, :n] = all_k[b]

        elif self.strategy in ("mask_token", "mae_mask_token"):
            # Zero-fill so bone/velocity are consistent; missing tokens replaced by mask_emb.
            pose_filled = pose.clone()
            pose_filled[~mask.unsqueeze(-1).expand_as(pose_filled)] = 0.0
            feat = self._build_features(pose_filled, mask)
            flat_feat = feat.view(B, T * K, -1)
            tokens = self.input_proj(flat_feat)  # (B, T*K, d_model)

            flat_mask = mask.view(B, T * K).unsqueeze(-1)  # (B, T*K, 1)
            mask_emb = self.mask_embedding.expand(B, T * K, -1)
            tokens = torch.where(flat_mask, tokens, mask_emb)

            t_pe = self.temporal_pe_table[flat_fi].unsqueeze(0)
            s_pe = self.spatial_pe(flat_ki).unsqueeze(0)
            tokens = tokens + t_pe + s_pe
            attn_mask = torch.ones(B, T * K, dtype=torch.bool, device=device)
            f_ids = flat_fi.unsqueeze(0).expand(B, -1)
            k_ids = flat_ki.unsqueeze(0).expand(B, -1)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return tokens, attn_mask, f_ids, k_ids

    @staticmethod
    def _interpolate_missing(pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Vectorized linear interpolation of missing keypoints along the time axis.

        Replaces the original B×K×D×T nested Python loop with tensor ops:
          - cummax scan  → left present-frame index for every position
          - cummin scan  → right present-frame index for every position
          - gather       → values at those frames
          - one lerp     → interpolated result

        Complexity is still O(B·K·T) but fully parallelised (no Python loops).
        """
        pose = pose.clone()
        B, T, K, D = pose.shape
        device = pose.device

        # Work in (B, K, T, D) so the time axis is dim 2
        p = pose.permute(0, 2, 1, 3)   # (B, K, T, D)
        m = mask.permute(0, 2, 1)       # (B, K, T)

        t_idx = torch.arange(T, dtype=torch.float32, device=device)  # (T,)

        # left_t[b,k,t]  = most recent present-frame index ≤ t  (-inf if none)
        left_t = t_idx.view(1, 1, T).expand(B, K, T).clone()
        left_t[~m] = float("-inf")
        left_t = left_t.cummax(dim=2).values   # (B, K, T)

        # right_t[b,k,t] = nearest upcoming present-frame index ≥ t  (+inf if none)
        right_t = t_idx.view(1, 1, T).expand(B, K, T).clone()
        right_t[~m] = float("inf")
        right_t = right_t.flip(2).cummin(dim=2).values.flip(2)   # (B, K, T)

        # Extrapolation: if no left neighbour use the right, if no right use the left
        has_left  = left_t.isfinite()
        has_right = right_t.isfinite()
        lf = torch.where(has_left,  left_t,  right_t)
        rf = torch.where(has_right, right_t, left_t)

        # Integer indices — clamp handles all-missing's ±inf safely
        lf_int = lf.clamp(0, T - 1).long()   # (B, K, T)
        rf_int = rf.clamp(0, T - 1).long()   # (B, K, T)

        # Gather keypoint values at the left / right anchor frames
        lf_int_d = lf_int.unsqueeze(-1).expand(B, K, T, D)
        rf_int_d = rf_int.unsqueeze(-1).expand(B, K, T, D)
        val_l = p.gather(2, lf_int_d)   # (B, K, T, D)
        val_r = p.gather(2, rf_int_d)   # (B, K, T, D)

        # Interpolation weight ∈ [0, 1]; clamp(min=1) avoids /0 when lf == rf
        denom = (rf - lf).clamp(min=1.0)
        alpha = ((t_idx.view(1, 1, T) - lf) / denom).clamp(0.0, 1.0)   # (B, K, T)

        interp = (1.0 - alpha.unsqueeze(-1)) * val_l + alpha.unsqueeze(-1) * val_r

        # Completely absent keypoints → zero (matches original behaviour)
        all_absent = ~m.any(dim=2, keepdim=True).unsqueeze(-1)   # (B, K, 1, 1)
        interp = interp.masked_fill(all_absent, 0.0)

        # Overwrite only the missing frames; present frames keep their original values
        p = torch.where((~m).unsqueeze(-1).expand_as(p), interp, p)

        return p.permute(0, 2, 1, 3)   # → (B, T, K, D)

    def _embed_grid(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Embed the full (B, T, K) grid into (B, T, K, d_model).
        Used by the factored space-time attention path.

        Missing positions are handled per strategy:
          zero_fill / token_drop  — zeroed coordinates
          interpolation           — linearly interpolated coordinates
          mask_token / mae_mask_token — learned [MASK] embedding
        """
        B, T, K, D = pose.shape
        device = pose.device

        frame_idx = torch.arange(T, device=device)   # (T,)
        kp_idx   = torch.arange(K, device=device)    # (K,)

        if self.strategy in ("zero_fill", "token_drop"):
            pose = pose.clone()
            pose[~mask.unsqueeze(-1).expand_as(pose)] = 0.0
            tokens = self.input_proj(self._build_features(pose, mask))

        elif self.strategy == "interpolation":
            pose = self._interpolate_missing(pose, mask)
            tokens = self.input_proj(self._build_features(pose, torch.ones_like(mask)))

        elif self.strategy in ("mask_token", "mae_mask_token"):
            pose_filled = pose.clone()
            pose_filled[~mask.unsqueeze(-1).expand_as(pose_filled)] = 0.0
            tokens = self.input_proj(self._build_features(pose_filled, mask))
            mask_emb = self.mask_embedding.view(1, 1, 1, self.d_model).expand(B, T, K, -1)
            tokens = torch.where(mask.unsqueeze(-1), tokens, mask_emb)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Positional encodings — broadcast over batch and the orthogonal axis
        t_pe = self.temporal_pe_table[frame_idx].view(1, T, 1, self.d_model)  # (1,T,1,d)
        s_pe = self.spatial_pe(kp_idx).view(1, 1, K, self.d_model)            # (1,1,K,d)
        return tokens + t_pe + s_pe                          # (B, T, K, d_model)

    def _encode(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder and return globally-pooled features before the head.

        Args:
            pose: (B, T, K, D)
            mask: (B, T, K) bool — True = present
        Returns:
            (B, d_model) normalised pooled representation
        """
        B = pose.shape[0]
        device = pose.device

        if self.use_factored_attention:
            tokens = self._embed_grid(pose, mask)             # (B, T, K, d_model)
            attn_mask = mask if self.strategy == "token_drop" else None
            pool = (self.strategy != "mae_mask_token")
            encoded, final_mask = self.encoder(tokens, mask=attn_mask, pool=pool)

            if self.strategy == "token_drop" and final_mask is not None:
                present = final_mask.unsqueeze(-1).float()
                pooled = (encoded * present).sum(dim=(1, 2)) / present.sum(dim=(1, 2)).clamp(min=1.0)
            else:
                pooled = encoded.mean(dim=(1, 2))

        else:
            tokens, attn_mask, f_ids, k_ids = self._tokenize_and_embed(pose, mask)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            cls_attn = torch.ones(B, 1, dtype=torch.bool, device=device)
            attn_mask = torch.cat([cls_attn, attn_mask], dim=1)
            key_padding_mask = ~attn_mask
            encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
            pooled = encoded[:, 0]

        return self.norm(pooled)  # (B, d_model)

    def forward(self, pose, mask):
        """
        Args:
            pose: (B, T, K, D)
            mask: (B, T, K) bool — True = present
        Returns:
            logits: (B, num_classes)
        """
        return self.head(self._encode(pose, mask))

    def mae_forward(self, pose, mask, mae_mask_ratio=0.5):
        """
        Masked autoencoder forward pass for pretraining.

        Randomly masks `mae_mask_ratio` of PRESENT tokens and tries to
        reconstruct their keypoint values.

        Returns: reconstruction loss (MSE on masked present tokens)
        """
        assert self.strategy == "mae_mask_token", "MAE only for mae_mask_token strategy"

        B, T, K, D = pose.shape
        device = pose.device

        # Decide which present tokens to hide from the encoder
        mae_hide = (torch.rand(B, T, K, device=device) < mae_mask_ratio) & mask  # (B,T,K)
        visible_mask = mask & ~mae_hide                                            # (B,T,K)

        # Zero-fill missing coords so bone/velocity features are consistent.
        pose_filled = pose.clone()
        pose_filled[~mask.unsqueeze(-1).expand_as(pose_filled)] = 0.0
        pose_feat = self._build_features(pose_filled, mask)  # (B, T, K, n_streams*D)

        if self.use_factored_attention:
            tokens = self.input_proj(pose_feat)               # (B, T, K, d_model)
            mask_emb = self.mask_embedding.view(1, 1, 1, self.d_model).expand(B, T, K, -1)
            tokens = torch.where(visible_mask.unsqueeze(-1), tokens, mask_emb)

            frame_idx = torch.arange(T, device=device)
            kp_idx   = torch.arange(K, device=device)
            t_pe = self.temporal_pe_table[frame_idx].view(1, T, 1, self.d_model)
            s_pe = self.spatial_pe(kp_idx).view(1, 1, K, self.d_model)
            tokens = tokens + t_pe + s_pe

            # pool=False keeps T intact so the decoder can map back to all T*K positions
            encoded, _ = self.encoder(tokens, mask=None, pool=False)  # (B, T, K, d_model)
            encoded_flat = encoded.reshape(B, T * K, self.d_model)

        else:
            # Flat-attention path: pack into (B, T*K, d) then encode
            flat_pose = pose_feat.view(B, T * K, -1)
            tokens = self.input_proj(flat_pose)

            vis_flat = visible_mask.view(B, T * K).unsqueeze(-1)
            mask_emb = self.mask_embedding.expand(B, T * K, -1)
            tokens = torch.where(vis_flat, tokens, mask_emb)

            fi = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)
            ki = torch.arange(K, device=device).unsqueeze(0).expand(T, K).reshape(-1)
            tokens = tokens + self.temporal_pe_table[fi].unsqueeze(0) + self.spatial_pe(ki).unsqueeze(0)

            encoded_flat = self.encoder(tokens)               # (B, T*K, d_model)

        # Reconstruct keypoint values for MAE-hidden tokens
        pred = self.mae_decoder(encoded_flat)                 # (B, T*K, D)

        target   = pose.view(B, T * K, D).detach()
        loss_mask = mae_hide.view(B, T * K).unsqueeze(-1).float()   # (B, T*K, 1)
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = ((pred - target) ** 2 * loss_mask).sum() / loss_mask.sum() / D
        return loss


# ---------------------------------------------------------------------------
# pyskl backbone wrappers
# ---------------------------------------------------------------------------

@BACKBONES.register_module()
class PoseTransformerBackbone(nn.Module):
    """
    pyskl backbone wrapping PoseTransformer.

    Converts pyskl's (N, M, T, V, C) input format to PoseTransformer's
    (B, T, K, D) format, derives a presence mask from zero-feature detection,
    runs the encoder, and returns globally-pooled features shaped as
    (N, M, d_model, 1, 1) for compatibility with GCNHead.

    Input features are expected to come from GenSkeFeat (e.g. feats=['j','b',
    'jm','bm']); bone vectors and velocity should be produced there, not
    computed internally.  Set in_channels to match the total channel count:
    len(feats) * channels_per_feat  (e.g. 4 × 3 = 12 for HRNet x/y/score).

    Args:
        num_keypoints (int): Number of skeleton joints (V). Default: 17.
        in_channels (int): Total input channels per joint, i.e. the C
            dimension produced by GenSkeFeat. Default: 3.
        num_frames (int): Expected clip length (used by sinusoidal PE table).
            Default: 64.
        d_model (int): Transformer embedding dimension. Default: 128.
        nhead (int): Number of attention heads. Default: 4.
        num_layers (int): Number of transformer blocks. Default: 4.
        dim_feedforward (int): FFN hidden dimension. Default: 256.
        dropout (float): Dropout rate. Default: 0.1.
        strategy (str): Missing-data strategy — one of 'zero_fill',
            'interpolation', 'token_drop', 'mask_token'. Default: 'zero_fill'.
        use_factored_attention (bool): Use factored space-time attention
            (TimeSformer-style) instead of flat attention. Default: True.
        temporal_pool_every (int): Halve T after every N blocks (0 = off).
            Default: 0.
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        in_channels: int = 3,
        num_frames: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        strategy: str = "zero_fill",
        use_factored_attention: bool = True,
        temporal_pool_every: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self._model = PoseTransformer(
            num_classes=1,  # dummy — classification head is not used
            num_keypoints=num_keypoints,
            keypoint_dim=in_channels,  # full GenSkeFeat channel count
            num_frames=num_frames,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            strategy=strategy,
            use_factored_attention=use_factored_attention,
            temporal_pool_every=temporal_pool_every,
            use_velocity=False,  # feature streams come from GenSkeFeat
            use_bone=False,
        )

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, M, T, V, C) — pyskl backbone input
        Returns:
            (N, M, d_model, 1, 1) — compatible with GCNHead (mode='GCN')
        """
        N, M, T, V, C = x.shape
        x_flat = x.reshape(N * M, T, V, C)          # (N*M, T, V, C)

        # Derive presence mask: a joint is present if any of its feature
        # channels is non-zero.  Zero-padding from the pipeline zeroes all C
        # channels for absent joints, so this reliably identifies missing data.
        mask = x_flat.abs().sum(-1) > 0              # (N*M, T, V) bool

        features = self._model._encode(x_flat, mask) # (N*M, d_model)
        return features.reshape(N, M, self.d_model, 1, 1)


@BACKBONES.register_module()
class PoseSTGCNBackbone(nn.Module):
    """
    pyskl backbone wrapping PoseSTGCN.

    Converts pyskl's (N, M, T, V, C) input format, runs the ST-GCN blocks,
    and returns feature maps shaped as (N, M, C_last, T, V) for compatibility
    with GCNHead.

    Input features are expected to come from GenSkeFeat (e.g. feats=['j','b',
    'jm','bm']).  Set in_channels to match the total channel count:
    len(feats) * channels_per_feat  (e.g. 4 × 3 = 12 for HRNet x/y/score).

    Args:
        num_keypoints (int): Number of skeleton joints. Default: 17.
        in_channels (int): Total input channels per joint from GenSkeFeat.
            Default: 3.
        base_channels (int): Base channel count; doubled every 2 blocks.
            Default: 64.
        num_layers (int): Number of ST-GCN blocks. Default: 5.
        dropout (float): Dropout rate inside temporal conv. Default: 0.1.
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        _mults = [1, 1, 2, 2, 4, 4]
        if num_layers <= len(_mults):
            mults = _mults[:num_layers]
        else:
            mults = _mults + [4] * (num_layers - len(_mults))
        self.out_channels = base_channels * mults[-1]

        self._model = PoseSTGCN(
            num_classes=1,  # dummy — classification head is not used
            num_keypoints=num_keypoints,
            keypoint_dim=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, M, T, V, C) — pyskl backbone input
        Returns:
            (N, M, C_last, T, V) — compatible with GCNHead (mode='GCN')
        """
        N, M, T, V, C = x.shape
        x_flat = x.reshape(N * M, T, V, C)

        mask = x_flat.abs().sum(-1) > 0              # (N*M, T, V) bool
        feat = self._model.forward_features(x_flat, mask)  # (N*M, C_last, T, V)

        _, C_out, T_out, V_out = feat.shape
        return feat.reshape(N, M, C_out, T_out, V_out)


__all__ = [
    'PoseTransformer',
    'PoseSTGCN',
    'PoseTransformerBackbone',
    'PoseSTGCNBackbone',
]
