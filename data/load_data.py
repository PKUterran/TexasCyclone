import numpy as np
import torch
import pickle
import dgl

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.Netlist import Netlist


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
    cells_pos = torch.tensor(cells_pos, dtype=torch.float32)
    cells_size = torch.tensor(cell_data[:, [0, 1]], dtype=torch.float32)
    cells_degree = torch.tensor(cell_data[:, 2], dtype=torch.float32).unsqueeze(-1)
    cells_type = torch.tensor(cell_data[:, 3], dtype=torch.float32).unsqueeze(-1)
    nets_degree = torch.tensor(net_data[:, 0], dtype=torch.float32).unsqueeze(-1)
    pins_pos = torch.tensor(pin_data[:, [0, 1]], dtype=torch.float32)
    pins_io = torch.tensor(pin_data[:, 2], dtype=torch.float32).unsqueeze(-1)

    graph = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells, nets),
        ('net', 'pinned', 'cell'): (nets, cells),
        ('net', 'father', 'net'): ([], []),
        ('net', 'son', 'net'): ([], []),
    }, num_nodes_dict={'cell': n_cell, 'net': n_net})

    cells_feat = torch.cat([torch.log(cells_size), cells_degree], dim=-1)
    nets_feat = torch.cat([nets_degree], dim=-1)
    pins_feat = torch.cat([pins_pos / 1000, pins_io], dim=-1)

    cell_prop_dict = {
        'pos': cells_pos,
        'size': cells_size,
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
    # for k, v in cell_prop_dict.items():
    #     print(f'{k}: {v.shape}')
    # for k, v in net_prop_dict.items():
    #     print(f'{k}: {v.shape}')
    # for k, v in pin_prop_dict.items():
    #     print(f'{k}: {v.shape}')

    netlist = Netlist(
        graph=graph,
        cell_prop_dict=cell_prop_dict,
        net_prop_dict=net_prop_dict,
        pin_prop_dict=pin_prop_dict,
        hierarchical=True,
        cell_clusters=[[2, 3]]
    )
    if save_type != 0:
        with open(netlist_pickle_path, 'wb+') as fp:
            pickle.dump(netlist, fp)
    return netlist


if __name__ == '__main__':
    netlist_ = netlist_from_numpy_directory('test/dataset1/medium', save_type=2)
    print(netlist_.graph)
    print(netlist_.cell_prop_dict)
    print(netlist_.net_prop_dict)
    print(netlist_.pin_prop_dict)
    print(netlist_.n_cell, netlist_.n_net, netlist_.n_pin)
    print(netlist_.cell_flow.fathers_list)
    for _, edges in enumerate(netlist_.cell_flow.flow_edges):
        print(f'{_}: {edges}')
    print(netlist_.cell_flow.cell_paths)
    for k, v in netlist_.dict_sub_netlist.items():
        print(f'{k}:')
        print('\t', v.graph)
        print('\t', v.cell_prop_dict)
        print('\t', v.net_prop_dict)
        print('\t', v.pin_prop_dict)
        print('\t', v.n_cell, v.n_net, v.n_pin)
        print('\t', v.cell_flow.fathers_list)
        for _, edges in enumerate(v.cell_flow.flow_edges):
            print('\t', f'{_}: {edges}')
        print('\t', v.cell_flow.cell_paths)
