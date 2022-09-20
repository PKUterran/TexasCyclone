import numpy as np
import torch
from typing import Tuple, List

from data.Layout import Layout
from .LossFunction import LossFunction


def greedy_sample(layout: Layout, span) -> Tuple[List[int], List[int]]:
    cell_pos = layout.cell_pos.cpu().detach().numpy()
    sorted_x = np.argsort(cell_pos[:, 0])
    sorted_y = np.argsort(cell_pos[:, 1])
    sample_i, sample_j = [], []
    for sorted_indices in [sorted_x, sorted_y]:
        n = len(sorted_indices)
        for i in range(n):
            for d in range(1, span + 1):
                if i + d < n:
                    sample_i.append(sorted_indices[i])
                    sample_j.append(sorted_indices[i + d])
    return sample_i, sample_j


class SampleOverlapLoss(LossFunction):
    def __init__(self, span=4):
        super(SampleOverlapLoss, self).__init__()
        self.span = span

    def forward(self, layout: Layout, *args, **kwargs) -> torch.Tensor:
        sample_i, sample_j = greedy_sample(layout, self.span)
        cell_size = layout.netlist.cell_prop_dict['size']
        cell_pos = layout.cell_pos
        sample_cell_size_i = cell_size[sample_i, :]
        sample_cell_size_j = cell_size[sample_j, :]
        sample_cell_pos_i = cell_pos[sample_i, :]
        sample_cell_pos_j = cell_pos[sample_j, :]
        overlap_x = torch.relu((sample_cell_size_i[:, 0] + sample_cell_size_j[:, 0]) / 2 -
                               torch.abs(sample_cell_pos_i[:, 0] - sample_cell_pos_j[:, 0]))
        overlap_y = torch.relu((sample_cell_size_i[:, 1] + sample_cell_size_j[:, 1]) / 2 -
                               torch.abs(sample_cell_pos_i[:, 1] - sample_cell_pos_j[:, 1]))
        return torch.mean(torch.sqrt(overlap_x * overlap_y))
