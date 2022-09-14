import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
from typing import Dict, List, Tuple, Optional

import os, sys
sys.path.append(os.path.abspath('.'))
from data.Netlist import Netlist, netlist_from_numpy_directory_old, netlist_from_numpy_directory


class Layout:
    def __init__(
            self, netlist: Netlist,
            net_pos: torch.Tensor = None,
            cell_pos: torch.Tensor = None,
    ):
        self.netlist = netlist
        self._net_pos = net_pos
        self._cell_pos = cell_pos
        self._cell_span = None  # (x1, y1, x2, y2)
        self._net_span = None  # (x1, y1, x2, y2)

    @property
    def net_pos(self) -> Optional[torch.Tensor]:
        if self._net_pos is None:
            self._net_pos = self.netlist.norm_net_cell_matrix @ self.cell_pos
        return self._net_pos

    @property
    def cell_pos(self) -> Optional[torch.Tensor]:
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


def layout_from_directory(dir_name: str) -> Layout:
    netlist = netlist_from_numpy_directory(dir_name)
    return Layout(netlist, None, torch.tensor(netlist.cell_prop_dict['pos'], dtype=torch.float32))


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
