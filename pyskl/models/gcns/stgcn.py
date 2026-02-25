import copy as cp
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import JointPool, mstcn, tcn_transformer, unit_gcn, unit_tcn, unit_transformer

EPS = 1e-4

# NTU RGB+D 25-joint → 10-group body-part hierarchy (0-indexed joints).
# Groups: Torso, Head, L-upper-arm, L-forearm+hand,
#         R-upper-arm, R-forearm+hand, L-thigh, L-shin+foot, R-thigh, R-shin+foot
NTU_POOL_GROUPS_25_10 = [
    [0, 1, 20],        # 0: Torso      (SpineBase, SpineMid, Chest)
    [2, 3],            # 1: Head       (Neck, Head)
    [4, 5],            # 2: L-up-arm   (LShoulder, LElbow)
    [6, 7, 21, 22],    # 3: L-forearm  (LWrist, LHand, LHandTip, LThumb)
    [8, 9],            # 4: R-up-arm   (RShoulder, RElbow)
    [10, 11, 23, 24],  # 5: R-forearm  (RWrist, RHand, RHandTip, RThumb)
    [12, 13],          # 6: L-thigh    (LHip, LKnee)
    [14, 15],          # 7: L-shin     (LAnkle, LFoot)
    [16, 17],          # 8: R-thigh    (RHip, RKnee)
    [18, 19],          # 9: R-shin     (RAnkle, RFoot)
]


def _build_pooled_A(A, groups):
    """Derive a pooled adjacency (K, V', V') from the original (K, V, V).

    Two super-joints are connected when any member of one is connected to any
    member of the other in the original graph.
    """
    V_in = A.size(1)
    V_out = len(groups)
    assignment = torch.zeros(V_in, V_out, dtype=A.dtype, device=A.device)
    for j, group in enumerate(groups):
        for i in group:
            assignment[i, j] = 1.0
    A_pool = torch.einsum('vi,kvw,wj->kij', assignment, A.float(), assignment)
    return (A_pool > 0).float()


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'tcn_transformer']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn', 'unit_transformer']

        if gcn_type == 'unit_gcn':
            self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        elif gcn_type == 'unit_transformer':
            # Drop unit_gcn-specific kwargs that don't apply to transformers
            transformer_kwargs = {k: v for k, v in gcn_kwargs.items()
                                  if k not in ['adaptive', 'conv_pos']}
            self.gcn = unit_transformer(in_channels, out_channels, A, **transformer_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type == 'tcn_transformer':
            self.tcn = tcn_transformer(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


@BACKBONES.register_module()
class STGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pool_stages=[],
                 pool_groups=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
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

        # --- joint pooling setup ---
        pool_stages = set(pool_stages)
        if pool_stages:
            pool_groups = pool_groups or NTU_POOL_GROUPS_25_10
            A_pooled = _build_pooled_A(A, pool_groups)
        else:
            pool_groups = None
            A_pooled = None
        current_A = A          # active adjacency matrix (may switch after pooling)
        pool_layer_dict = {}   # module_idx -> JointPool

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, current_A.clone(), 1, residual=False, **lw_kwargs[0])]
            if 1 in pool_stages:
                pool_layer_dict[0] = JointPool(pool_groups, base_channels)
                current_A = A_pooled

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, current_A.clone(), stride, **lw_kwargs[i - 1]))
            if i in pool_stages:
                module_idx = len(modules) - 1
                pool_layer_dict[module_idx] = JointPool(pool_groups, out_channels)
                current_A = A_pooled

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pool_layers = nn.ModuleDict({str(k): v for k, v in pool_layer_dict.items()})
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)
            if str(i) in self.pool_layers:
                x = self.pool_layers[str(i)](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x
