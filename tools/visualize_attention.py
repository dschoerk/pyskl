#!/usr/bin/env python3
"""Visualize per-joint attention weights from unit_transformer GCN layers.

Usage:
    python tools/visualize_attention.py configs/.../j.py -C checkpoint.pth
    python tools/visualize_attention.py configs/.../j.py -C checkpoint.pth --sample-idx 42 --out attn.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from pyskl.datasets import build_dataset
from pyskl.models import build_model

# Joint names for NTU RGB+D (25 joints, 0-indexed)
NTU_JOINT_NAMES = [
    'Pelvis', 'SpineMid', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'LHand',
    'RShoulder', 'RElbow', 'RWrist', 'RHand',
    'LHip', 'LKnee', 'LAnkle', 'LFoot',
    'RHip', 'RKnee', 'RAnkle', 'RFoot',
    'Chest',
    'LHandTip', 'LThumb', 'RHandTip', 'RThumb',
]

# 0-indexed edges matching graph.py nturgb+d layout
NTU_EDGES = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18), (21, 7), (22, 7), (23, 11), (24, 11),
]

# Rough 2-D frontal-view positions for NTU skeleton
NTU_POS = {
    0:  ( 0.0,  0.0), 1:  ( 0.0,  1.0), 20: ( 0.0,  2.0),
    2:  ( 0.0,  3.0), 3:  ( 0.0,  4.0),
    4:  (-1.5,  2.5), 5:  (-2.5,  1.5), 6:  (-3.0,  0.5), 7:  (-3.5,  0.0),
    8:  ( 1.5,  2.5), 9:  ( 2.5,  1.5), 10: ( 3.0,  0.5), 11: ( 3.5,  0.0),
    12: (-0.7, -1.0), 13: (-0.7, -2.0), 14: (-0.7, -3.0), 15: (-0.7, -3.7),
    16: ( 0.7, -1.0), 17: ( 0.7, -2.0), 18: ( 0.7, -3.0), 19: ( 0.7, -3.7),
    21: (-4.0, -0.3), 22: (-3.5, -0.5),
    23: ( 4.0, -0.3), 24: ( 3.5, -0.5),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GCN joint attention')
    parser.add_argument('config', help='config file path')
    parser.add_argument('-C', '--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='index of the sample to visualize from the val set')
    parser.add_argument('--label-idx', type=int, default=None,
                        help='if set, average attention over all val samples with this label')
    parser.add_argument('--label-map', default='tools/data/label_map/nturgbd_120.txt',
                        help='text file with one class name per line (0-indexed)')
    parser.add_argument('--out', default='attention.png', help='output image path')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def collect_attn(model):
    """Return list of attn_weights tensors, one per unit_transformer GCN layer."""
    weights = []
    for block in model.backbone.gcn:
        gcn = block.gcn
        if hasattr(gcn, 'attn_weights') and gcn.attn_weights is not None:
            weights.append(gcn.attn_weights)   # (N*T, h, V, V)
    return weights


def importance(attn):
    """Attention received per joint: mean over batch*time, heads, and query joints -> (V,)."""
    return attn.float().mean(dim=(0, 1, 2)).numpy()


def plot_bars(ax, imp, joint_names, title):
    norm = plt.Normalize(imp.min(), imp.max())
    colors = cm.YlOrRd(norm(imp))
    ax.bar(range(len(imp)), imp, color=colors)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels(joint_names, rotation=90, fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel('Avg attention received', fontsize=7)


def plot_skeleton(ax, imp, joint_names, edges, pos, title):
    norm = plt.Normalize(imp.min(), imp.max())
    for u, v in edges:
        if u < len(joint_names) and v < len(joint_names):
            xu, yu = pos[u]
            xv, yv = pos[v]
            ax.plot([xu, xv], [yu, yv], color='#aaaaaa', lw=1.5, zorder=1)
    xs = [pos[i][0] for i in range(len(joint_names))]
    ys = [pos[i][1] for i in range(len(joint_names))]
    sc = ax.scatter(xs, ys, c=imp, cmap='YlOrRd', s=250, zorder=2,
                    norm=norm, edgecolors='black', linewidths=0.5)
    for i, name in enumerate(joint_names):
        ax.annotate(name, (pos[i][0], pos[i][1]),
                    textcoords='offset points', xytext=(5, 3), fontsize=5)
    plt.colorbar(sc, ax=ax, label='Attention received', shrink=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = Config.fromfile(args.config)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(device).eval()

    label_map = None
    try:
        with open(args.label_map) as f:
            label_map = [line.strip() for line in f]
    except FileNotFoundError:
        pass

    def label_name(idx):
        if label_map is not None and 0 <= idx < len(label_map):
            return f'{idx} — {label_map[idx]}'
        return str(idx)

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    if args.label_idx is not None:
        # Collect indices of all samples with the requested label
        indices = [i for i, s in enumerate(dataset)
                   if s.get('label', None) == args.label_idx]
        if not indices:
            print(f'No samples found with label {args.label_idx}')
            return
        print(f'Averaging over {len(indices)} samples with label {label_name(args.label_idx)}')
        title_str = f'label {label_name(args.label_idx)} ({len(indices)} samples)'

        # Accumulate per-layer importance sums
        layer_acc = None
        for idx in indices:
            sample = dataset[idx]
            keypoint = sample['keypoint'].unsqueeze(0).to(device)
            with torch.no_grad():
                model(keypoint=keypoint, label=None, return_loss=False)
            weights = collect_attn(model)
            if not weights:
                print('No attention weights collected — is gcn_type=unit_transformer?')
                return
            imps = np.stack([importance(w) for w in weights])  # (L, V)
            if layer_acc is None:
                layer_acc = imps
            else:
                layer_acc += imps
        importances = list(layer_acc / len(indices))
        num_layers = len(importances)
    else:
        sample = dataset[args.sample_idx]
        keypoint = sample['keypoint'].unsqueeze(0).to(device)   # (1, M, T, V, C)
        label = sample.get('label', '?')
        label_str = label_name(label) if isinstance(label, int) else str(label)
        print(f'Sample {args.sample_idx}  label={label_str}  shape={tuple(keypoint.shape)}')
        title_str = f'sample {args.sample_idx} (label {label_str})'

        with torch.no_grad():
            model(keypoint=keypoint, label=None, return_loss=False)

        layer_weights = collect_attn(model)
        if not layer_weights:
            print('No attention weights collected — is gcn_type=unit_transformer?')
            return
        importances = [importance(w) for w in layer_weights]
        num_layers = len(importances)

    V = importances[0].shape[-1]
    joint_names = NTU_JOINT_NAMES[:V]
    agg = np.stack(importances).mean(0)

    # Layout: one column per layer bar chart, then aggregate bar + skeleton
    ncols = min(num_layers, 5)
    nrows_bars = (num_layers + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * (nrows_bars + 1)))
    fig.suptitle(f'Joint attention — {title_str}', fontsize=10)
    gs = GridSpec(nrows_bars + 1, ncols, figure=fig)

    for i, imp in enumerate(importances):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        plot_bars(ax, imp, joint_names, f'Layer {i}')

    ax_agg = fig.add_subplot(gs[nrows_bars, 0])
    plot_bars(ax_agg, agg, joint_names, 'Aggregate (all layers)')

    ax_skel = fig.add_subplot(gs[nrows_bars, 1:])
    plot_skeleton(ax_skel, agg, joint_names, NTU_EDGES, NTU_POS,
                  'Skeleton — aggregate importance')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
