import torch
import torch.nn as nn

from ..builder import RECOGNIZERS, build_backbone, build_head
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class SkeletonMAE(BaseRecognizer):
    """MAE-style pre-training for skeleton transformers.

    Masks a random subset of (t, v) tokens, runs the full encoder,
    then reconstructs masked positions with a lightweight decoder.

    Args:
        backbone (dict): Config for the SkeletonTransformer encoder.
        decoder_head (dict): Config for the MAEHead decoder.
        mask_ratio (float): Fraction of tokens to mask. Default: 0.75.
        norm_target (bool): Normalize reconstruction targets per-sample.
        train_cfg (dict): Training config.
        test_cfg (dict): Testing config.
    """

    def __init__(self,
                 backbone,
                 decoder_head,
                 mask_ratio=0.75,
                 norm_target=True,
                 train_cfg=dict(),
                 test_cfg=dict()):
        # Skip BaseRecognizer.__init__ to avoid premature init_weights call.
        # Manually replicate the setup.
        nn.Module.__init__(self)
        self.backbone = build_backbone(backbone)
        self.cls_head = None
        self.decoder_head = build_head(decoder_head)

        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.max_testing_views = self.test_cfg.get('max_testing_views', None)

        self.mask_ratio = mask_ratio
        self.norm_target = norm_target

        # Learnable mask token in embedding space
        embed_dim = self.backbone.input_proj.out_features
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self.decoder_head, 'init_weights'):
            self.decoder_head.init_weights()

    def _random_mask(self, N, M, T, V, device):
        """Generate random binary mask. True = masked (to be predicted)."""
        num_tokens = T * V
        num_mask = int(self.mask_ratio * num_tokens)
        # Per-sample, per-person independent masking
        mask = torch.zeros(N, M, num_tokens, dtype=torch.bool, device=device)
        for i in range(N):
            for j in range(M):
                idx = torch.randperm(num_tokens, device=device)[:num_mask]
                mask[i, j, idx] = True
        return mask.reshape(N, M, T, V)

    def forward_train(self, keypoint, label=None, **kwargs):
        assert keypoint.shape[1] == 1
        x = keypoint[:, 0]  # (N, M, T, V, C)
        N, M, T, V, C = x.shape

        # Detect valid (non-padding) persons: a person is valid if any
        # coordinate is non-zero across all frames and joints
        valid_person = (x.abs().sum(dim=(2, 3, 4)) > 0)  # (N, M)

        # Temporal patching for target: group frames into patches
        p = self.backbone.temporal_patch_size
        T_tok = T // p  # token-level temporal resolution
        if p > 1:
            # (N, M, T, V, C) -> (N, M, T_tok, V, p*C)
            target = x.reshape(N, M, T_tok, p, V, C).permute(0, 1, 2, 4, 3, 5)
            target = target.reshape(N, M, T_tok, V, p * C)
        else:
            target = x.clone()

        if self.norm_target:
            mean = target.mean(dim=(2, 3), keepdim=True)
            std = target.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            target = (target - mean) / std

        # Generate mask at token resolution — only mask valid persons
        mask = self._random_mask(N, M, T_tok, V, x.device)  # (N, M, T_tok, V)
        mask = mask & valid_person[:, :, None, None]

        # Encode with mask token injection (backbone handles patching internally)
        feat = self.backbone(x, mask=mask, mask_token=self.mask_token)

        # Decode and compute reconstruction loss (only on masked valid tokens)
        loss_rec = self.decoder_head(feat, target, mask)
        return dict(loss_rec=loss_rec)

    def forward_test(self, keypoint, **kwargs):
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        return self.backbone(keypoint)

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(keypoint, label, **kwargs)
        return self.forward_test(keypoint, **kwargs)
