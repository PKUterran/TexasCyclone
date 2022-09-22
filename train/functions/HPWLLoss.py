import torch

from data.Layout import Layout
from .LossFunction import LossFunction


class HPWLLoss(LossFunction):
    def __init__(self):
        super(HPWLLoss, self).__init__()
        self.cal_vector = torch.tensor([-1, -1, 1, 1], dtype=torch.float32)

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        net_span = layout.net_span
        net_wl = net_span @ self.cal_vector
        return torch.mean(net_wl)
