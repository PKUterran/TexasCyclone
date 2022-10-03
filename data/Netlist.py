import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
import tqdm
from typing import Dict, List, Tuple, Union, Optional

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph import Tree, generate_net_tree_from_netlist_graph
from data.utils import pad_net_cell_list

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

        self._macro_cell_indices = None
        self._net_tree = None
        self._net_net_pair_matrix = None
        self._net_cell_pair_matrix = None
        assert self.net_tree is not None

        self._net_offset_pos_matrix = None
        self._net_cell_matrix = None
        self._norm_net_cell_matrix = None
        self._pin_net_matrix = None
        self._pin_cell_matrix = None
        self._norm_pin_cell_matrix = None

        self._net_cell_indices_matrix = None

    @property
    def macro_cell_indices(self) -> List[int]:
        if self._macro_cell_indices is None:
            assert 'type' in self.cell_prop_dict.keys()
            macro_cell_indices = list(np.argwhere(self.cell_prop_dict['type'][:, 0] > 0).reshape([-1]))
            self._macro_cell_indices = list(sorted(macro_cell_indices, key=lambda x: self.cell_prop_dict['size'][x, 0], reverse=True))
        return self._macro_cell_indices

    @property
    def net_tree(self) -> Tree:
        if self._net_tree is None:
            self._net_tree = generate_net_tree_from_netlist_graph(self.graph)
            u, v = [], []
            for i, f in enumerate(self._net_tree.father_list):
                if f != -1:
                    u.append(f)
                    v.append(i)
            self.graph.add_edges(u, v, etype='father')
            self.graph.add_edges(v, u, etype='son')
        return self._net_tree

    @property
    def net_net_pair_matrix(self) -> torch.Tensor:
        if self._net_net_pair_matrix is None:
            root_loop_father_list = [(i, f) if f != -1 else (i, i) for i, f in enumerate(self.net_tree.father_list)]
            self._net_net_pair_matrix = torch.tensor(root_loop_father_list, dtype=torch.int64)
        return self._net_net_pair_matrix

    @property
    def net_cell_pair_matrix(self) -> torch.Tensor:
        if self._net_cell_pair_matrix is None:
            nets, cells = self.graph.edges(etype='pinned')
            self._net_cell_pair_matrix = torch.vstack([nets, cells]).t().to(torch.int64)
        return self._net_cell_pair_matrix

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
                indices=torch.tensor(indices, dtype=torch.int64),
                values=values,
                size=[self.n_net, self.n_net], dtype=torch.float32
            )
        return self._net_offset_pos_matrix

    @property
    def net_cell_matrix(self) -> sparse.Tensor:
        if self._net_cell_matrix is None:
            nets, cells = self.graph.edges(etype='pinned')
            net_cell_tuples = list(zip(nets, cells))
            self._net_cell_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(net_cell_tuples, dtype=torch.int64).t(),
                values=torch.ones(size=[self.n_pin]),
                size=[self.n_net, self.n_cell], dtype=torch.float32
            )
        return self._net_cell_matrix

    @property
    def norm_net_cell_matrix(self) -> sparse.Tensor:
        if self._norm_net_cell_matrix is None:
            nets, cells = self.graph.edges(etype='pinned')
            net_cell_tuples = list(zip(nets, cells))
            dict_net_cell_cnt = {}
            for n, _ in net_cell_tuples:
                dict_net_cell_cnt[int(n)] = dict_net_cell_cnt.setdefault(int(n), 0) + 1
            self._norm_net_cell_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(net_cell_tuples, dtype=torch.int64).t(),
                values=[1 / dict_net_cell_cnt[int(n)] for n in nets],
                size=[self.n_net, self.n_cell], dtype=torch.float32
            )
        return self._norm_net_cell_matrix

    @property
    def pin_net_matrix(self) -> sparse.Tensor:
        if self._pin_net_matrix is None:
            nets, _ = self.graph.edges(etype='pinned')
            pin_net_tuples = list(enumerate(nets))
            self._pin_net_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(pin_net_tuples, dtype=torch.int64).t(),
                values=torch.ones(size=[self.n_pin]),
                size=[self.n_pin, self.n_net], dtype=torch.float32
            )
        return self._pin_net_matrix

    @property
    def pin_cell_matrix(self) -> sparse.Tensor:
        if self._pin_cell_matrix is None:
            _, cells = self.graph.edges(etype='pinned')
            self._pin_cell_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(list(enumerate(cells)), dtype=torch.int64).t(),
                values=torch.ones(size=[self.n_pin]),
                size=[self.n_pin, self.n_cell], dtype=torch.float32
            )
        return self._pin_cell_matrix

    @property
    def norm_pin_cell_matrix(self) -> sparse.Tensor:
        if self._norm_pin_cell_matrix is None:
            _, cells = self.graph.edges(etype='pinned')
            cells = cells.tolist()
            dict_cell_cnt = {}
            for c in cells:
                dict_cell_cnt[c] = dict_cell_cnt.setdefault(c, 0) + 1
            self._norm_pin_cell_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(list(enumerate(cells)), dtype=torch.int64).t(),
                values=[1 / dict_cell_cnt[c] for c in cells],
                size=[self.n_pin, self.n_cell], dtype=torch.float32
            )
        return self._norm_pin_cell_matrix

    @property
    def net_cell_indices_matrix(self) -> Optional[torch.Tensor]:
        if self._net_cell_indices_matrix is None:
            ncl = [[] for _ in range(self.n_net)]
            nets, cells = self.graph.edges(etype='pinned')
            net_cell_tuples = list(zip(nets, cells))
            for net, cell in net_cell_tuples:
                ncl[int(net)].append(int(cell))
            self._net_cell_indices_matrix = pad_net_cell_list(ncl, 30)
        return self._net_cell_indices_matrix


def netlist_from_numpy_directory_old(dir_name: str, given_iter=None) -> Netlist:
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
        ('net', 'father', 'net'): ([], []),
        ('net', 'son', 'net'): ([], []),
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


def netlist_from_numpy_directory(dir_name: str, save_type: int = 1) -> Netlist:
    # 0: ignore cache; 1: use and dump cache; 2: force dump cache
    print(f'\tLoading {dir_name}')
    netlist_pickle_path = f'{dir_name}/netlist.pkl'
    if save_type == 1 and os.path.exists(netlist_pickle_path):
        with open(netlist_pickle_path, 'rb') as fp:
            netlist = pickle.load(fp)
        return netlist
    pin_net_cell = np.load(f'{dir_name}/pin_net_cell.npy')
    cell_data = np.load(f'{dir_name}/cell_data.npy')
    net_data = np.load(f'{dir_name}/net_data.npy')
    pin_data = np.load(f'{dir_name}/pin_data.npy')

    n_cell, n_net = cell_data.shape[0], net_data.shape[0]
    if os.path.exists(f'{dir_name}/cell_pos.npy'):
        cells_pos = np.load(f'{dir_name}/cell_pos.npy')
    else:
        cells_pos = np.zeros(shape=[n_cell, 2], dtype=np.float)
    cells = list(pin_net_cell[:, 1])
    nets = list(pin_net_cell[:, 0])
    cells_size = torch.tensor(cell_data[:, [0, 1]], dtype=torch.float32)
    cells_degree = torch.tensor(cell_data[:, 2], dtype=torch.float32).unsqueeze(-1)
    cells_type = torch.tensor(cell_data[:, 3], dtype=torch.float32).unsqueeze(-1)
    nets_degree = torch.tensor(net_data, dtype=torch.float32).unsqueeze(-1)
    pins_pos = torch.tensor(pin_data[:, [0, 1]], dtype=torch.float32)
    pins_io = torch.tensor(pin_data[:, 2], dtype=torch.float32).unsqueeze(-1)

    graph = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells, nets),
        ('net', 'pinned', 'cell'): (nets, cells),
        ('net', 'father', 'net'): ([], []),
        ('net', 'son', 'net'): ([], []),
    }, num_nodes_dict={'cell': n_cell, 'net': n_net})

    cells_feat = torch.cat([cells_size / POS_FAC, cells_degree], dim=-1)
    nets_feat = torch.cat([nets_degree], dim=-1)
    pins_feat = torch.cat([pins_pos / POS_FAC, pins_io], dim=-1)

    cell_prop_dict = {
        'size': cells_size,
        'pos': cells_pos,
        'feat': cells_feat,
        'type': cells_type,
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

    netlist = Netlist(
        graph=graph,
        cell_prop_dict=cell_prop_dict,
        net_prop_dict=net_prop_dict,
        pin_prop_dict=pin_prop_dict
    )
    if save_type != 0:
        with open(netlist_pickle_path, 'wb+') as fp:
            pickle.dump(netlist, fp)
    return netlist


# if __name__ == '__main__':
#     netlist = netlist_from_numpy_directory('test/dataset1/small')
#     print(netlist.graph)
#     print(netlist.cell_prop_dict)
#     print(netlist.net_prop_dict)
#     print(netlist.pin_prop_dict)
#     print(netlist.n_cell, netlist.n_net, netlist.n_pin)
#     print(netlist.net_tree.children_dict)
#     print(netlist.net_tree.path_dict)
#     print(netlist.net_offset_pos_matrix.to_dense())
#     print(netlist.pin_net_matrix.to_dense())
#     print(netlist.pin_cell_matrix.to_dense())
