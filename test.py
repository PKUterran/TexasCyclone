import numpy as np
import torch.multiprocessing as mp
import time

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph import Netlist, expand_netlist
from data.load_data import netlist_from_numpy_directory, layout_from_netlist_ref
from data.pretrain import DIS_ANGLE_TYPE, load_pretrain_data
import torch
import itertools
import json
from math import sqrt



def get_gt_dis_mean_std(layout):
    nets,cells = layout.netlist.graph.edges(etype='pinned')
    n_net = layout.netlist.graph.num_nodes(ntype='net')
    n_cell = layout.netlist.graph.num_nodes(ntype='cell')

    list_net_cells = [[] for _ in range(n_net)]
    list_cell_nets = [[] for _ in range(n_cell)]
    for net,cell in zip(nets,cells):
        list_net_cells[net].append(cell)
        list_cell_nets[cell].append(net)
    
    list_dis = []
    for net in range(n_net):
        for pair in itertools.product(list_net_cells[net],list_net_cells[net]):
            cell_a,cell_b = pair
            if(cell_a >= cell_b):
                continue
            dis = torch.norm(layout.cell_pos[cell_a, :] - layout.cell_pos[cell_b, :],dim=0)
            list_dis.append(dis)
    tensor_dis = torch.tensor(list_dis)
    return torch.std_mean(tensor_dis)

if __name__ == '__main__':
    # w, h = 100, 80
    # net_span = np.array([
    #     [0, 0, 250, 250],
    #     [150, 150, 400, 500],
    #     [600, 600, 1500, 1001],
    # ], dtype=np.float32)
    # net_degree = np.array([[2], [3], [4]], dtype=np.float32)
    # layout_size = (1000, 900)
    # shape = (int(layout_size[0] / w) + 1, int(layout_size[1] / h) + 1)
    # print(shape)
    # cong_map = np.zeros(shape=shape, dtype=np.float32)

    # for span, (degree,) in zip(net_span, net_degree):
    #     w1, w2 = map(int, span[[0, 2]] / w)
    #     h1, h2 = map(int, span[[0, 2]] / h)
    #     print(w1, w2, h1, h2)
    #     density = degree / (w2 - w1 + 1) / (h2 - h1 + 1)
    #     for i in range(w1, min(w2 + 1, shape[0])):
    #         for j in range(h1, min(h2 + 1, shape[1])):
    #             cong_map[i, j] += density

    # print(cong_map)
    train_datasets = [
    '../Placement-datasets/dac2012/superblue2',
    # '../Placement-datasets/dac2012/superblue3',
    # '../Placement-datasets/dac2012/superblue6',
    # 'data/test/dataset1/medium',
]   
    # train_netlists = [netlist_from_numpy_directory(dataset,True,2) for dataset in train_datasets]
    train_netlists = [load_pretrain_data(dataset,2) for dataset in train_datasets]
    # dict_netlist = expand_netlist(train_netlists[0])

    # dict_nid_dis_angle = {}
    # for nid, sub_netlist in dict_netlist.items():
    #     layout = layout_from_netlist_ref(sub_netlist)
    #     fathers, sons = layout.netlist.graph.edges(etype='points-to')
    #     edge_rel_pos = layout.cell_pos[sons, :] - layout.cell_pos[fathers, :]
    #     edge_dis = torch.norm(edge_rel_pos, dim=1)
    #     edge_angle = torch.tensor(np.arctan2(edge_rel_pos[:, 1], edge_rel_pos[:, 0]) / np.pi, dtype=torch.float32)
    #     dict_nid_dis_angle[nid] = (edge_dis, edge_angle)
    #     if(layout.netlist.graph.num_nodes(ntype='cell') == 2):
    #         continue
    #     gt_edge_dis_mean,true_edge_dis_std = get_gt_dis_mean_std(layout)
    #     print("------------")
    #     print(layout.netlist.graph.num_nodes(ntype='cell'))
    #     print(torch.std_mean(dict_nid_dis_angle[nid][0]))
    #     print(gt_edge_dis_mean,true_edge_dis_std)
    #     print("------------")
    # with open(f'{train_datasets[0]}/cell_clusters_old.json') as fp:
    #     cell_clusters = json.load(fp)
    # cal_small_group_num_nodes = 0
    # total_group_size = []
    # for cell_group in cell_clusters:
    #     if len(cell_group) <= 10:
    #         cal_small_group_num_nodes += len(cell_group)
    #     total_group_size.append(len(cell_group))
    # want_group_fenweishu = 1-sqrt(sum(total_group_size))/len(cell_clusters)
    # sifenweishu = (25, 50, 75,want_group_fenweishu*100)
    # total_group_size.sort()
    # print(f"四分位数 {sifenweishu} is {np.percentile(total_group_size, sifenweishu)}")
    # print(f"total num nodes {sum(total_group_size)}")
    # print(f"total group is {len(cell_clusters)}")
    # print(f"group node num is {sum(total_group_size[int(len(cell_clusters)*want_group_fenweishu):])}")
