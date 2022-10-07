import torch

from data.graph import Layout
from .LossFunction import LossFunction


class AreaLoss(LossFunction):
    def __init__(self):
        super(AreaLoss, self).__init__()

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        limit = kwargs['limit']
        cell_span = layout.cell_span
        cell_span_excess = torch.relu(torch.cat([
            torch.tensor(limit[:2], dtype=torch.float32) - cell_span[:, :2],
            cell_span[:, 2:] - torch.tensor(limit[2:], dtype=torch.float32),
        ], dim=-1))
        return torch.mean(cell_span_excess ** 2)
