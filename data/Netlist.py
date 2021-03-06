import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
from typing import Dict, List, Tuple, Union

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph import Tree, generate_net_tree_from_netlist_graph

POS_FAC = 1000


class Netlist:
    def __init__(
            self, graph: dgl.DGLHeteroGraph,
            cell_prop_dict: Dict[str, torch.Tensor],
            net_prop_dict: Dict[str, torch.Tensor],
            pin_prop_dict: Dict[str, torch.Tensor],
    ):
        self.graph = graph
        self.cell_prop_dict = cell_prop_dict
        self.net_prop_dict = net_prop_dict
        self.pin_prop_dict = pin_prop_dict

        self.n_cell = graph.num_nodes(ntype='cell')
        self.n_net = graph.num_nodes(ntype='net')
        self.n_pin = graph.num_edges(etype='pinned')

        self._net_tree = None
        self._net_offset_pos_matrix = None

    @property
    def net_tree(self) -> Tree:
        if self._net_tree is None:
            self._net_tree = generate_net_tree_from_netlist_graph(self.graph)
        return self._net_tree

    @property
    def net_offset_pos_matrix(self) -> sparse.Tensor:
        if self._net_offset_pos_matrix is None:
            indices = [[], []]
            values = []
            for k, path in self.net_tree.path_dict.items():
                for p in path:
                    indices[0].append(k)
                    indices[1].append(p)
                    values.append(1)
            self._net_offset_pos_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(indices, dtype=torch.int32),
                values=values,
                size=[self.n_net, self.n_net]
            )
        return self._net_offset_pos_matrix


def netlist_from_numpy_directory(dir_name: str, given_iter=None) -> Netlist:
    # 1. load data
    with open(f'{dir_name}/edge.pkl', 'rb') as fp:
        edge: Dict[int, Tuple[int, float, float, int]] = pickle.load(fp)
    size_x = np.load(f'{dir_name}/sizdata_x.npy')
    size_y = np.load(f'{dir_name}/sizdata_y.npy')
    pos_x = np.load(f'{dir_name}/xdata_{given_iter}.npy')
    pos_y = np.load(f'{dir_name}/ydata_{given_iter}.npy')
    try:
        cells_data = np.load(f'{dir_name}/pdata.npy')
    except FileNotFoundError:
        cells_data = np.zeros([size_x.shape[0], 0])

    n_cell = size_x.shape[0]
    n_net = len(edge.keys())
    cells_size = torch.tensor(np.vstack((size_x, size_y)), dtype=torch.float32).t()
    cells_pos = torch.tensor(np.vstack((pos_x, pos_y)), dtype=torch.float32).t()
    cells_data = torch.tensor(cells_data, dtype=torch.float32)
    if len(cells_data.shape) == 1:
        cells_data = cells_data.unsqueeze(-1)

    # 2. collect features
    cells, nets, pins_pos, pins_io, pins_data = [], [], [], [], []
    nets_degree = []

    for net, list_cell_feats in edge.items():
        nets_degree.append(len(list_cell_feats))
        for cell, pin_px, pin_py, pin_io in list_cell_feats:
            cells.append(cell)
            nets.append(net)
            pins_pos.append([pin_px, pin_py])
            pins_io.append([pin_io])
            pins_data.append([])

    nets_degree = torch.tensor(nets_degree, dtype=torch.float32).unsqueeze(-1)
    pins_pos = torch.tensor(pins_pos, dtype=torch.float32)
    pins_io = torch.tensor(pins_io, dtype=torch.float32)
    pins_data = torch.tensor(pins_data, dtype=torch.float32)

    # 3. construct graph
    graph = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells, nets),
        ('net', 'pinned', 'cell'): (nets, cells),
    }, num_nodes_dict={'cell': n_cell, 'net': n_net})

    cells_feat = torch.cat([cells_data, cells_size / POS_FAC, cells_pos / POS_FAC], dim=-1)
    nets_feat = torch.cat([nets_degree], dim=-1)
    pins_feat = torch.cat([pins_pos / POS_FAC, pins_io, pins_data], dim=-1)

    cell_prop_dict = {
        'size': cells_size,
        'pos': cells_pos,
        'feat': cells_feat,
    }
    net_prop_dict = {
        'degree': nets_degree,
        'feat': nets_feat,
    }
    pin_prop_dict = {
        'pos': pins_pos,
        'io': pins_io,
        'feat': pins_feat,
    }

    return Netlist(
        graph=graph,
        cell_prop_dict=cell_prop_dict,
        net_prop_dict=net_prop_dict,
        pin_prop_dict=pin_prop_dict
    )


if __name__ == '__main__':
    netlist = netlist_from_numpy_directory('test', 900)
    print(netlist.graph)
    print(netlist.cell_prop_dict)
    print(netlist.net_prop_dict)
    print(netlist.pin_prop_dict)
    print(netlist.n_cell, netlist.n_net, netlist.n_pin)
    print(netlist.net_tree.children_dict)
    print(netlist.net_tree.path_dict)
    print(netlist.net_offset_pos_matrix.to_dense())
