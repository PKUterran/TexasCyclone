import numpy as np
import torch
import pickle
import dgl

from typing import Dict, List, Tuple
# import os, sys
# sys.path.append(os.path.abspath('.'))


class Netlist:
    def __init__(self, graph: dgl.DGLHeteroGraph):
        self.graph = graph

    @staticmethod
    def from_numpy_directory(dir_name: str, given_iter=None):
        # 1. load data
        with open(f'{dir_name}/edge.pkl', 'rb') as fp:
            edge: Dict[int, Tuple[int, float, float, int]] = pickle.load(fp)
        size_x = np.load(f'{dir_name}/sizdata_x.npy')
        size_y = np.load(f'{dir_name}/sizdata_y.npy')
        pos_x = np.load(f'{dir_name}/xdata_{given_iter}.npy')
        pos_y = np.load(f'{dir_name}/ydata_{given_iter}.npy')
        cells_data = np.load(f'{dir_name}/pdata.npy')

        n_node = size_x.shape[0]
        n_net = len(edge.keys())
        cells_size = torch.tensor(np.vstack((size_x, size_y)), dtype=torch.float32).t()
        cells_pos = torch.tensor(np.vstack((pos_x, pos_y)), dtype=torch.float32).t()
        cells_data = torch.tensor(cells_data, dtype=torch.float32)
        if len(cells_data.shape) == 1:
            cells_data = cells_data.unsqueeze(-1)

        # 2. collect features
        cells, nets, pins_pos, pins_io, pins_data = [], [], [], [], []
        net_degree = []

        for net, list_cell_feats in edge.items():
            net_degree.append(len(list_cell_feats))
            for cell, pin_px, pin_py, pin_io in list_cell_feats:
                cells.append(cell)
                nets.append(net)
                pins_pos.append([pin_px, pin_py])
                pins_io.append([pin_io])
                pins_data.append([])

        net_degree = torch.tensor(net_degree, dtype=torch.float32).unsqueeze(-1)
        pins_pos = torch.tensor(pins_pos, dtype=torch.float32)
        pins_io = torch.tensor(pins_io, dtype=torch.float32)
        pins_data = torch.tensor(pins_data, dtype=torch.float32)

        # 3. construct graph
        graph = dgl.heterograph({
            ('cell', 'pins', 'net'): (cells, nets),
            ('net', 'pinned', 'cell'): (nets, cells),
        }, num_nodes_dict={'node': n_node, 'net': n_net})

        cells_feat = torch.cat([cells_data, cells_size, cells_pos], dim=-1)
        nets_feat = torch.cat([net_degree], dim=-1)
        pins_feat = torch.cat([pins_pos, pins_io, pins_data], dim=-1)

        graph.nodes['node'].data['size'] = cells_size
        graph.nodes['node'].data['pos'] = cells_pos
        graph.nodes['node'].data['feat'] = cells_feat
        graph.nodes['net'].data['feat'] = nets_feat
        # same for graph.edges['pins']
        graph.edges['pinned'].data['pos'] = pins_pos
        graph.edges['pinned'].data['io'] = pins_io
        graph.edges['pinned'].data['feat'] = pins_feat

        return Netlist(graph=graph)
