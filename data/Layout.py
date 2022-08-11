import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
from typing import Dict, List, Tuple, Optional

import os, sys
sys.path.append(os.path.abspath('.'))
from data.Netlist import Netlist, netlist_from_numpy_directory_old


class Layout:
    def __init__(
            self, netlist: Netlist,
            net_pos: torch.Tensor = None,
            cell_pos: torch.Tensor = None,
    ):
        self.netlist = netlist
        self._net_pos = net_pos
        self._cell_pos = cell_pos

    @property
    def net_pos(self) -> Optional[torch.Tensor]:
        return self._net_pos

    @property
    def cell_pos(self) -> Optional[torch.Tensor]:
        return self._cell_pos


def layout_from_netlist_dis_angle(
        netlist: Netlist,
        nets_dis: torch.Tensor, nets_angle: torch.Tensor,
        pins_dis: torch.Tensor, pins_angle: torch.Tensor,
) -> Tuple[Layout, torch.Tensor]:
    nets_offset = torch.stack([nets_dis * torch.cos(nets_angle * np.pi), nets_dis * torch.sin(nets_angle * np.pi)]).t()
    net_pos = netlist.net_offset_pos_matrix @ nets_offset
    pins_offset = torch.stack([pins_dis * torch.cos(pins_angle * np.pi), pins_dis * torch.sin(pins_angle * np.pi)]).t()
    pinned_cells_pos = pins_offset + netlist.pin_net_matrix @ net_pos
    cell_pos = netlist.norm_pin_cell_matrix.t() @ pinned_cells_pos
    cell_pos_discrepancy = pinned_cells_pos - netlist.pin_cell_matrix @ cell_pos
    return Layout(netlist, net_pos, cell_pos), torch.norm(cell_pos_discrepancy)


if __name__ == '__main__':
    layout, d_loss = layout_from_netlist_dis_angle(
        netlist_from_numpy_directory_old('test-old', 900),
        torch.tensor([400, 400, 400], dtype=torch.float32),
        torch.tensor([0, 0, -0.25], dtype=torch.float32),
        torch.tensor([200, 200, 200, 400, 200, 200, 200], dtype=torch.float32),
        torch.tensor([-1, 0.5, -0.5, -0.8, 1.5, 1.3, 1.7], dtype=torch.float32),
    )
    print(layout.net_pos)
    print(layout.cell_pos)
    print(d_loss)
