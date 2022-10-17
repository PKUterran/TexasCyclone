from tkinter import E
import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
import tqdm
from typing import Dict, List, Tuple, Optional

from pympler import asizeof
import psutil
from memory_profiler import profile, memory_usage
import copy
import ctypes

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.CellFlow import CellFlow
from data.graph.utils import pad_net_cell_list


class Netlist:
    def __init__(
            self, graph: dgl.DGLHeteroGraph,
            cell_prop_dict: Dict[str, torch.Tensor],
            net_prop_dict: Dict[str, torch.Tensor],
            pin_prop_dict: Dict[str, torch.Tensor],
            layout_size: Optional[Tuple[float, float]] = None,
            hierarchical: bool = False,
            cell_clusters: Optional[List[List[int]]] = None,
            original_netlist=None, simple=False,
    ):
        self.graph = graph
        self.cell_prop_dict = cell_prop_dict
        self.net_prop_dict = net_prop_dict
        self.pin_prop_dict = pin_prop_dict
        self.original_netlist = original_netlist
        self.dict_sub_netlist = {}
        if hierarchical:
            self.adapt_hierarchy(cell_clusters, use_tqdm=True)

        self.n_cell = self.graph.num_nodes(ntype='cell')
        self.n_net = self.graph.num_nodes(ntype='net')
        self.n_pin = self.graph.num_edges(etype='pinned')

        self.layout_size = layout_size
        if self.layout_size is None:
            self.adapt_layout_size()
            assert self.layout_size is not None

        self.terminal_indices = list(map(lambda x: int(x),
                                         torch.argwhere(self.cell_prop_dict['type'][:, 0] > 0).view(-1)))
        if len(self.terminal_indices) == 0:
            self.adapt_terminals()
            assert len(self.terminal_indices) > 0

        if simple:
            return
        self._cell_flow = None
        self._cell_path_edge_matrix = None
        self._path_cell_matrix = None
        self._path_edge_matrix = None
        self._net_cell_indices_matrix = None
        self.terminal_edge_pos = self.cell_prop_dict['pos'][self.terminal_indices, :]
        self.n_edge = len(self.cell_flow.flow_edge_indices)
        fathers, sons = zip(*self.cell_flow.flow_edge_indices[len(self.terminal_indices):])
        self.graph.add_edges(fathers, sons, etype='points-to')
        self.graph.add_edges(sons, fathers, etype='pointed-from')

    @property
    def cell_flow(self) -> CellFlow:
        if self._cell_flow is None:
            self.construct_cell_flow()
        return self._cell_flow

    @property
    def cell_path_edge_matrix(self) -> torch.sparse.Tensor:
        if self._cell_path_edge_matrix is None:
            self.construct_cell_path_edge_matrices()
        return self._cell_path_edge_matrix

    @property
    def path_cell_matrix(self) -> torch.sparse.Tensor:
        if self._path_cell_matrix is None:
            self.construct_cell_path_edge_matrices()
        return self._path_cell_matrix

    @property
    def path_edge_matrix(self) -> torch.sparse.Tensor:
        if self._path_edge_matrix is None:
            self.construct_cell_path_edge_matrices()
        return self._path_edge_matrix

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

    def get_cell_clusters(self) -> List[List[int]]:
        raise NotImplementedError

    def adapt_hierarchy(self, cell_clusters: Optional[List[List[int]]], use_tqdm=False):
        if cell_clusters is None:
            cell_clusters = self.get_cell_clusters()

        temp_n_cell = self.graph.num_nodes(ntype='cell')
        parted_cells = set()
        pseudo_cell_ref_pos = []
        pseudo_cell_size = []
        pseudo_cell_degree = []
        pseudo_pin_pos = []

        iter_partition_list = tqdm.tqdm(cell_clusters, total=len(cell_clusters)) if use_tqdm else cell_clusters
        ###########
        '''
        提前预处理每个子图中的net
        '''
        sub_graph_net_degree_dict_list = [dict() for _ in range(len(cell_clusters))]
        belong_node = np.ones(temp_n_cell) * -1
        for i,sub_graph_list in enumerate(cell_clusters):
            for node in sub_graph_list:
                belong_node[int(node)] = i
        for net_id, cell_id in zip(*[ns.tolist() for ns in self.graph.edges(etype='pinned')]):
            sub_graph_id = int(belong_node[int(cell_id)])
            if sub_graph_id == -1:
                continue
            sub_graph_net_degree_dict_list[sub_graph_id].setdefault(int(net_id),0)
            sub_graph_net_degree_dict_list[sub_graph_id][int(net_id)] += 1
        ###########
        for i, partition in enumerate(iter_partition_list):
            if len(partition) <= 1:
                continue
            partition_set = set(partition)
            parted_cells |= partition_set
            #################
            # new_net_degree_dict = {}
            # for net_id, cell_id in zip(*[ns.tolist() for ns in self.graph.edges(etype='pinned')]):
            #     if cell_id in partition_set:
            #         new_net_degree_dict.setdefault(net_id, 0)
            #         new_net_degree_dict[net_id] += 1
            new_net_degree_dict = sub_graph_net_degree_dict_list[i]
            ######################
            keep_nets_id = np.array(list(new_net_degree_dict.keys()))
            keep_nets_degree = np.array(list(new_net_degree_dict.values()))
            good_nets = np.abs(self.net_prop_dict['degree'][keep_nets_id, 0] - keep_nets_degree) < 1e-5
            good_nets_id = torch.tensor(keep_nets_id)[good_nets]  # numpy似乎不支持用TRUE FALSE来筛选数据所以换成tensor
            sub_graph = dgl.node_subgraph(self.graph, nodes={'cell': partition, 'net': keep_nets_id})
            sub_netlist = Netlist(
                graph=sub_graph,
                cell_prop_dict={k: v[sub_graph.nodes['cell'].data[dgl.NID], :] for k, v in self.cell_prop_dict.items()},
                net_prop_dict={k: v[sub_graph.nodes['net'].data[dgl.NID], :] for k, v in self.net_prop_dict.items()},
                pin_prop_dict={k: v[sub_graph.edges['pinned'].data[dgl.EID], :] for k, v in self.pin_prop_dict.items()},
            )
            ######################

            # netlist_size = asizeof.asizeof(sub_netlist) / 1024 / 1024
            # cell_flow_size = asizeof.asizeof(sub_netlist._cell_flow) / 1024 / 1024
            # cell_proc_dict_size = asizeof.asizeof(sub_netlist.cell_prop_dict) / 1024 / 1024
            # flow_edge_indices_size = asizeof.asizeof(sub_netlist._cell_flow.flow_edge_indices) / 1024 / 1024
            # cell_paths_size = asizeof.asizeof(sub_netlist._cell_flow.cell_paths) / 1024 / 1024
            # print(f"netlist {i} cell flow size is {cell_flow_size} MB account for {cell_flow_size / netlist_size}%")
            # print(f"netlist {i} cell proc dict size is {cell_proc_dict_size} MB account for {cell_proc_dict_size / netlist_size}%")
            # print(f"netlist {i} flow edge size is {flow_edge_indices_size} MB account for {flow_edge_indices_size / netlist_size}%")
            # print(f"netlist {i} cell path size is {cell_paths_size} MB account for {cell_paths_size / netlist_size}%")
            # print(f"netlist {i} size is {asizeof.asizeof(sub_netlist) / 1024 / 1024} MB has {len(partition)} node")
            # dict_sub_netlist_size = asizeof.asizeof(self.dict_sub_netlist) / 1024 / 1024
            # total_mem_use = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            # print(f"sub netlist size is {dict_sub_netlist_size} MB account for {dict_sub_netlist_size / total_mem_use} total")
            # print(f"total mem use {total_mem_use} MB")
            #####################
            self.dict_sub_netlist[temp_n_cell] = sub_netlist
            self.graph.add_nodes(1, ntype='cell')
            self.graph.add_edges(keep_nets_id, [temp_n_cell] * len(keep_nets_id), etype='pinned')
            self.graph.add_edges([temp_n_cell] * len(keep_nets_id), keep_nets_id, etype='pins')
            ref_pos = torch.mean(sub_netlist.cell_prop_dict['ref_pos'], dim=0)
            sub_netlist.cell_prop_dict['ref_pos'] -= \
                ref_pos - torch.tensor(sub_netlist.layout_size, dtype=torch.float32)
            #################
            '''
            这个地方感觉append可能会慢所以修改了一下实现
            '''
            # pseudo_cell_ref_pos.append(ref_pos)
            pseudo_cell_size.append(sub_netlist.layout_size)
            pseudo_cell_degree.append(len(keep_nets_id) - len(good_nets_id))
            # pseudo_pin_pos.extend([[0, 0] for _ in range(len(keep_nets_id))])
            if len(pseudo_cell_ref_pos) == 0:
                pseudo_cell_ref_pos = ref_pos
            else:
                pseudo_cell_ref_pos = torch.vstack([pseudo_cell_ref_pos, ref_pos])
            if len(pseudo_pin_pos) == 0:
                pseudo_pin_pos = torch.zeros([len(keep_nets_id), 2])
            else:
                pseudo_pin_pos = torch.zeros([pseudo_pin_pos.size(0)+len(keep_nets_id), 2])
            #######################
            temp_n_cell += 1

        #################
        # pseudo_cell_ref_pos = torch.vstack(pseudo_cell_ref_pos)
        pseudo_cell_size = torch.tensor(pseudo_cell_size, dtype=torch.float32)
        pseudo_cell_pos = torch.full_like(pseudo_cell_size, fill_value=torch.nan)
        ######################
        pseudo_cell_degree = torch.tensor(pseudo_cell_degree, dtype=torch.float32).unsqueeze(-1)
        pseudo_cell_feat = torch.cat([torch.log(pseudo_cell_size), pseudo_cell_degree], dim=-1)
        self.cell_prop_dict['ref_pos'] = torch.vstack([self.cell_prop_dict['ref_pos'], pseudo_cell_ref_pos])
        self.cell_prop_dict['size'] = torch.vstack([self.cell_prop_dict['size'], pseudo_cell_size])
        self.cell_prop_dict['pos'] = torch.vstack([self.cell_prop_dict['pos'], pseudo_cell_pos])
        self.cell_prop_dict['feat'] = torch.vstack([self.cell_prop_dict['feat'], pseudo_cell_feat])
        self.cell_prop_dict['type'] = torch.vstack([self.cell_prop_dict['type'], torch.zeros_like(pseudo_cell_degree)])

        pseudo_pin_pos = torch.tensor(pseudo_pin_pos, dtype=torch.float32)
        pseudo_pin_io = torch.full(size=[pseudo_pin_pos.shape[0], 1], fill_value=2)
        pseudo_pin_feat = torch.cat([pseudo_pin_pos / 1000, pseudo_pin_io], dim=-1)
        self.pin_prop_dict['pos'] = torch.vstack([self.pin_prop_dict['pos'], pseudo_pin_pos])
        self.pin_prop_dict['io'] = torch.vstack([self.pin_prop_dict['io'], pseudo_pin_io])
        self.pin_prop_dict['feat'] = torch.vstack([self.pin_prop_dict['feat'], pseudo_pin_feat])

        left_cells = set(range(temp_n_cell)) - parted_cells
        left_nets = set()
        for net_id, cell_id in zip(*[ns.tolist() for ns in self.graph.edges(etype='pinned')]):
            if cell_id in left_cells:
                left_nets.add(net_id)
        self.graph = dgl.node_subgraph(self.graph, nodes={'cell': list(left_cells), 'net': list(left_nets)})
        self.cell_prop_dict = {k: v[self.graph.nodes['cell'].data[dgl.NID], :] for k, v in self.cell_prop_dict.items()}
        self.net_prop_dict = {k: v[self.graph.nodes['net'].data[dgl.NID], :] for k, v in self.net_prop_dict.items()}
        self.pin_prop_dict = {k: v[self.graph.edges['pinned'].data[dgl.EID], :] for k, v in self.pin_prop_dict.items()}
        dict_reverse_nid = {int(idx): i for i, idx in enumerate(self.graph.nodes['cell'].data[dgl.NID])}
        self.dict_sub_netlist = {dict_reverse_nid[k]: v for k, v in self.dict_sub_netlist.items()}

    def adapt_layout_size(self):
        # TODO: better layout adaption
        cells_size = self.cell_prop_dict['size']
        span = (cells_size[:, 0] * cells_size[:, 1]).sum() * 5
        self.layout_size = (span ** 0.5, span ** 0.5)

    def adapt_terminals(self):
        # TODO: better terminal selection
        biggest_cell_id = int(torch.argmax(torch.sum(self.cell_prop_dict['size'], dim=-1)))
        self.terminal_indices = [biggest_cell_id]
        self.cell_prop_dict['pos'][biggest_cell_id, :] = self.cell_prop_dict['size'][biggest_cell_id, :] / 2

    def construct_cell_flow(self):
        self._cell_flow = CellFlow(self.graph, self.terminal_indices)

    def construct_cell_path_edge_matrices(self):
        cell_path_edge_indices = [[], []]
        path_cell_indices = [[], []]
        path_edge_indices = [[], []]
        cell_path_edge_values = []
        n_paths = 0
        for c, paths in enumerate(self.cell_flow.cell_paths):
            n_path = len(paths)
            dict_edge_weight = {}
            for ip, path in enumerate(paths):
                path_cell_indices[0].append(n_paths + ip)
                path_cell_indices[1].append(c)
                for e in path:
                    dict_edge_weight.setdefault(e, 0)
                    dict_edge_weight[e] += 1 / n_path
                    path_edge_indices[0].append(n_paths + ip)
                    path_edge_indices[1].append(e)
            for k, v in dict_edge_weight.items():
                cell_path_edge_indices[0].append(c)
                cell_path_edge_indices[1].append(k)
                cell_path_edge_values.append(v)
            n_paths += n_path
        self._cell_path_edge_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(cell_path_edge_indices, dtype=torch.int64),
            values=cell_path_edge_values,
            size=[self.n_cell, self.n_edge], dtype=torch.float32
        )
        self._path_cell_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(path_cell_indices, dtype=torch.int64),
            values=torch.ones(size=[len(path_cell_indices[0])]),
            size=[n_paths, self.n_cell], dtype=torch.float32
        )
        self._path_edge_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(path_edge_indices, dtype=torch.int64),
            values=torch.ones(size=[len(path_edge_indices[0])]),
            size=[n_paths, self.n_edge], dtype=torch.float32
        )


def expand_netlist(netlist: Netlist) -> Dict[int, Netlist]:
    # key is the id of pseudo macro in main netlist
    # main netlist with key -1
    dict_netlist = {-1: netlist}
    dict_netlist.update(netlist.dict_sub_netlist)
    return dict_netlist


def sequentialize_netlist(netlist: Netlist) -> List[Netlist]:
    return [netlist] + list(netlist.dict_sub_netlist.values())
