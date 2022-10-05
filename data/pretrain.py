import pickle
import torch
import numpy as np
from typing import Tuple, Dict

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph import Netlist, expand_netlist
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_ref

TOKEN = 'dis_angle'
DIS_ANGLE_TYPE = Tuple[torch.Tensor, torch.Tensor]


def dump_pretrain_data(dir_name: str, save_type=1):
    netlist = netlist_from_numpy_directory(dir_name, save_type=save_type)
    dict_netlist = expand_netlist(netlist)

    dict_nid_dis_angle = {}
    for nid, sub_netlist in dict_netlist.items():
        layout = layout_from_netlist_ref(netlist)
        fathers, sons = layout.netlist.graph.edges(etype='points-to')
        edge_rel_pos = layout.cell_pos[sons, :] - layout.cell_pos[fathers, :]
        edge_dis = torch.norm(edge_rel_pos, dim=1)
        edge_angle = torch.tensor(np.arctan2(edge_rel_pos[:, 1], edge_rel_pos[:, 0]) / np.pi, dtype=torch.float32)
        dict_nid_dis_angle[nid] = (edge_dis, edge_angle)

    with open(f'{dir_name}/{TOKEN}.pkl', 'wb+') as fp:
        pickle.dump(dict_nid_dis_angle, fp)


def load_pretrain_data(dir_name: str, save_type=1) -> Tuple[Netlist, Dict[int, DIS_ANGLE_TYPE]]:
    # save_type 2: force save
    pretrain_pickle_path = f'{dir_name}/{TOKEN}.pkl'
    if not os.path.exists(pretrain_pickle_path) or save_type != 1:
        dump_pretrain_data(dir_name, save_type=save_type)
    print(f'\tLoading {pretrain_pickle_path}...')
    with open(pretrain_pickle_path, 'rb') as fp:
        dict_nid_dis_angle = pickle.load(fp)
    netlist = netlist_from_numpy_directory(dir_name, save_type=1)
    return netlist, dict_nid_dis_angle
