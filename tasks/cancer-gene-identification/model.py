"""MTGCN — a multi-task Chebyshev-spectral GNN for cancer-gene node classification.

Two ChebConv layers encode each node into a 100-d embedding; a ChebConv head
produces the cancer-gene logit. A second, self-supervised link-prediction head
(inner product of embeddings on real vs random edges) shares the encoder, and
the two losses are combined with learned homoscedastic uncertainty weights
(Kendall et al.) — the `log_var_*` parameters. Shared by train and inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

import config as C


class MTGCN(nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(in_dim, C.HID1, K=C.K_ORDER)
        self.conv2 = ChebConv(C.HID1, C.HID2, K=C.K_ORDER)
        self.cls = ChebConv(C.HID2, 1, K=C.K_ORDER)
        # learned multi-task uncertainty weights (classification vs link)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_link = nn.Parameter(torch.zeros(1))

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)

    def forward(self, x, edge_index, edge_index_cls):
        z = self.encode(x, edge_index)
        logit = self.cls(F.relu(z), edge_index_cls).squeeze(-1)
        return logit, z


def dropedge(edge_index, p):
    """Randomly drop a fraction p of edges (regularization)."""
    if p <= 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= p
    return edge_index[:, keep]


def sample_negatives(N, num, device):
    """Random node pairs as negative edges for the link-prediction head."""
    a = torch.randint(0, N, (num,), device=device)
    b = torch.randint(0, N, (num,), device=device)
    return a, b
