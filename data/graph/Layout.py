import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
from typing import Dict, List, Tuple, Optional

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.Netlist import Netlist


class Layout:
    def __init__(
            self, netlist: Netlist,
            cell_pos: torch.Tensor = None,
    ):
        self.netlist = netlist
        self._cell_pos = cell_pos
        self._cell_span = None  # (x1, y1, x2, y2)
        self._net_span = None  # (x1, y1, x2, y2)

    @property
    def cell_pos(self) -> Optional[torch.Tensor]:
        assert self._cell_pos is not None
        return self._cell_pos

    @property
    def cell_span(self) -> Optional[torch.Tensor]:
        if self._cell_span is None:
            cell_pos = self.cell_pos
            cell_size = self.netlist.cell_prop_dict['size']
            x1_y1 = cell_pos - cell_size / 2
            x2_y2 = cell_pos + cell_size / 2
            self._cell_span = torch.cat([x1_y1, x2_y2], dim=-1)
        return self._cell_span

    @property
    def net_span(self) -> Optional[torch.Tensor]:
        if self._net_span is None:
            net_cell_indices_matrix = self.netlist.net_cell_indices_matrix
            cell_span = self.cell_span
            net_cell_span = cell_span[net_cell_indices_matrix, :]
            self._net_span = torch.cat([
                torch.min(net_cell_span[:, :, :2], dim=1)[0],
                torch.max(net_cell_span[:, :, 2:], dim=1)[0]
            ], dim=-1)
        return self._net_span
