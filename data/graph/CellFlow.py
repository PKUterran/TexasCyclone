import numpy as np
import dgl
import queue
import torch
from typing import List, Dict, Sequence, Set
from copy import deepcopy

import os, sys
sys.path.append(os.path.abspath('.'))


class CellFlow:
    def __init__(self, graph: dgl.DGLHeteroGraph, terminal_indices: Sequence[int]):
        n_net = graph.num_nodes(ntype='net')
        n_cell = graph.num_nodes(ntype='cell')

        # 1. Get cell connection
        list_net_cells: List[Set[int]] = [set() for _ in range(n_net)]
        list_cell_nets: List[Set[int]] = [set() for _ in range(n_cell)]
        for net, cell in zip(*graph.edges(etype='pinned')):
            list_net_cells[int(net)].add(int(cell))
            list_cell_nets[int(cell)].add(int(net))

        # 2. Initialize roots of CellFlow
        fathers_list: List[List[int]] = [[] for _ in range(n_cell)]
        for t in terminal_indices:
            fathers_list[t].append(-1)

        # 3. Expand the flow
        net_flag = [False for _ in range(n_net)]
        cell_queue = queue.Queue()
        for t in terminal_indices:
            cell_queue.put(t)
        while not cell_queue.empty():
            cell: int = cell_queue.get()
            adj_cells = set()
            for net in list_cell_nets[cell]:
                if net_flag[net]:
                    continue
                adj_cells |= list_net_cells[net]
                net_flag[net] = True
            for adj_cell in adj_cells - {cell}:
                if len(fathers_list[adj_cell]) == 0:
                    cell_queue.put(adj_cell)
                fathers_list[adj_cell].append(cell)

        # 4. Collect the flow edges
        dict_cell_children = {}
        for i, fathers in enumerate(fathers_list):
            for f in fathers:
                dict_cell_children.setdefault(f, []).append(i)

        flow_edge_indices = []
        cell_paths = [[] for _ in range(n_cell)]
        edge_cnt = 0
        edge_stack, temp_path = queue.LifoQueue(), []

        ## 4.1 Label the terminals (fixed)
        for t in terminal_indices:
            flow_edge_indices.append((-1, t))
            edge_cnt += 1

        ## 4.2 Find the paths from terminals to movable cells
        set_terminals = set(terminal_indices)
        for i, t in enumerate(terminal_indices):
            assert edge_stack.empty()
            edge_stack.put((-1, t))
            while not edge_stack.empty():
                k = edge_stack.get()
                if k == (-2, -2):
                    temp_path.pop()
                    continue
                if k[0] == -1:
                    temp_path.append(i)
                else:
                    flow_edge_indices.append(k)
                    temp_path.append(edge_cnt)
                    edge_cnt += 1
                cell_paths[k[1]].append(deepcopy(temp_path))
                edge_stack.put((-2, -2))
                # Sample only one path from each of its father to avoid combination explosion
                if k[0] == fathers_list[k[1]][0]:
                    for child in dict_cell_children.get(k[1], []):
                        if child in set_terminals:
                            continue
                        edge_stack.put((k[1], child))

        # time/space complexity: O(D * P)
        # where D is the max depth of CellFlow and P is the # of pins
        assert len(flow_edge_indices) == edge_cnt
        self.fathers_list = fathers_list
        self.flow_edge_indices = flow_edge_indices
        self.cell_paths = cell_paths


if __name__ == '__main__':
    g = dgl.heterograph({
        ('net', 'pinned', 'cell'): (
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [0, 1, 3, 0, 2, 1, 4, 2, 4, 5, 4]
        ),
    }, num_nodes_dict={'cell': 6, 'net': 5})
    rs = [0, 1]
    cell_flow = CellFlow(g, rs)
    print(cell_flow.fathers_list)
    for _, edges in enumerate(cell_flow.flow_edge_indices):
        print(f'{_}: {edges}')
    print(cell_flow.cell_paths)
