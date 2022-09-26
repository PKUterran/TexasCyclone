import pickle
import torch
from typing import Tuple

import os, sys
sys.path.append(os.path.abspath('.'))
from data.Netlist import Netlist, netlist_from_numpy_directory
from data.Layout import layout_from_directory

TOKEN = 'dis_angle'
DIS_ANGLE_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def dump_pretrain_data(dir_name: str):
    layout = layout_from_directory(dir_name, save_type=1)
    net_net_pair_matrix = layout.netlist.net_net_pair_matrix
    net_rel_pos = layout.net_pos - layout.net_pos[net_net_pair_matrix[:, 1], :]

    nets, cells = layout.netlist.graph.edges(etype='pinned')
    pin_rel_pos = layout.cell_pos[cells, :] - layout.net_pos[nets, :]

    net_dis = torch.norm(net_rel_pos, dim=1)
    net_angle = torch.arctan(net_rel_pos[:, 1] / net_rel_pos[:, 0])
    pin_dis = torch.norm(pin_rel_pos, dim=1)
    pin_angle = torch.arctan(pin_rel_pos[:, 1] / pin_rel_pos[:, 0])
    with open(f'{dir_name}/{TOKEN}.pkl', 'wb+') as fp:
        pickle.dump((net_dis, net_angle, pin_dis, pin_angle), fp)


def load_pretrain_data(dir_name: str
                       ) -> Tuple[Netlist, DIS_ANGLE_TYPE]:
    pretrain_pickle_path = f'{dir_name}/{TOKEN}.pkl'
    if not os.path.exists(pretrain_pickle_path):
        dump_pretrain_data(dir_name)
    with open(pretrain_pickle_path, 'rb') as fp:
        net_dis, net_angle, pin_dis, pin_angle = pickle.load(fp)
    netlist = netlist_from_numpy_directory(dir_name, save_type=1)
    return netlist, (net_dis, net_angle, pin_dis, pin_angle)
