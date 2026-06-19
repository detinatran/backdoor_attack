"""
CTR-GCN (Channel-wise Topology Refinement Graph Convolution Network)
Simplified implementation for NTU RGB+D skeleton action recognition.
Based on: Chen et al., "Channel-wise Topology Refinement Graph Convolution
for Skeleton-Based Action Recognition", ICCV 2021.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# NTU RGB+D adjacency matrix (25 joints)
NTU_EDGES = [
    (0,1),(1,20),(2,20),(3,2),(4,20),(5,4),(6,5),(7,6),
    (8,20),(9,8),(10,9),(11,10),(12,0),(13,12),(14,13),(15,14),
    (16,0),(17,16),(18,17),(19,18),(21,7),(22,7),(23,11),(24,11),
]

def get_adjacency(n_joints=25):
    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in NTU_EDGES:
        A[i, j] = 1
        A[j, i] = 1
    A += np.eye(n_joints, dtype=np.float32)
    D = np.diag(A.sum(1) ** -0.5)
    return D @ A @ D


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T, J)
        w = x.mean(dim=[2, 3])          # (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class CTRGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, residual=True):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A))
        J = A.shape[0]

        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1),
                      padding=(1, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
        )
        self.ca = ChannelAttention(out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if not residual or in_ch != out_ch or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.res = nn.Identity()

    def forward(self, x):
        # x: (B, C, T, J)
        res = self.res(x)
        # graph conv: matmul with adjacency
        y = torch.einsum('bctj,jk->bctk', x, self.A)
        y = self.gcn(y)
        y = self.tcn(y)
        y = self.ca(y)
        return self.relu(self.bn(y) + res)


class CTRGCN(nn.Module):
    def __init__(self, n_classes=60, n_joints=25, in_channels=3):
        super().__init__()
        A = get_adjacency(n_joints)
        self.data_bn = nn.BatchNorm1d(in_channels * n_joints)

        def _block(ic, oc, stride=1, res=True):
            return CTRGCNBlock(ic, oc, A, stride=stride, residual=res)

        self.layers = nn.Sequential(
            _block(in_channels, 64,  res=False),
            _block(64,  64),
            _block(64,  64),
            _block(64,  64),
            _block(64,  128, stride=2),
            _block(128, 128),
            _block(128, 128),
            _block(128, 256, stride=2),
            _block(256, 256),
            _block(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (B, C, T, J, M)  M=number of persons
        B, C, T, J, M = x.shape
        # average over persons
        x = x.mean(dim=-1)           # (B, C, T, J)

        # data BN over joint×channel
        xbn = x.permute(0, 1, 3, 2).contiguous().view(B, C*J, T)
        xbn = self.data_bn(xbn)
        x = xbn.view(B, C, J, T).permute(0, 1, 3, 2)  # (B, C, T, J)

        x = self.layers(x)
        x = self.pool(x).view(B, -1)
        return self.fc(x)
