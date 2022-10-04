import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
import tqdm
from typing import Dict, List, Tuple, Optional

import os, sys
sys.path.append(os.path.abspath('.'))
from data.graph.CellFlow import CellFlow


class Netlist:
    def __init__(
            self, graph: dgl.DGLHeteroGraph,
            cell_prop_dict: Dict[str, torch.Tensor],
            net_prop_dict: Dict[str, torch.Tensor],
            pin_prop_dict: Dict[str, torch.Tensor],
            layout_size: Optional[Tuple[float, float]] = None,
            hierarchical: bool = False,
            cell_clusters: Optional[List[List[int]]] = None
    ):
        self.graph = graph
        self.cell_prop_dict = cell_prop_dict
        self.net_prop_dict = net_prop_dict
        self.pin_prop_dict = pin_prop_dict
        self.original_n_cell = graph.num_nodes(ntype='cell')
        self.dict_sub_netlist = {}
        if hierarchical:
            self.adapt_hierarchy(cell_clusters, use_tqdm=True)

        self.n_cell = graph.num_nodes(ntype='cell')
        self.n_net = graph.num_nodes(ntype='net')
        self.n_pin = graph.num_edges(etype='pinned')

        self.layout_size = layout_size
        if self.layout_size is None:
            self.adapt_layout_size()
            assert self.layout_size is not None

        self.terminal_indices = list(map(lambda x: int(x),
                                         torch.argwhere(self.cell_prop_dict['type'][:, 0] > 0).view(-1)))
        if len(self.terminal_indices) == 0:
            self.adapt_terminals()
            assert len(self.terminal_indices) > 0

        self._cell_flow = None

    def get_cell_clusters(self) -> List[List[int]]:
        raise NotImplementedError

    def adapt_hierarchy(self, cell_clusters: Optional[List[List[int]]], use_tqdm=False):
        if cell_clusters is None:
            cell_clusters = self.get_cell_clusters()

        temp_n_cell = self.graph.num_nodes(ntype='cell')
        parted_cells = set()
        pseudo_cells_size = []
        pseudo_cells_degree = []
        pseudo_pins_pos = []

        iter_partition_list = tqdm.tqdm(cell_clusters, total=len(cell_clusters)) if use_tqdm else cell_clusters
        for partition in iter_partition_list:
            if len(partition) <= 1:
                continue
            partition_set = set(partition)
            parted_cells |= partition_set
            new_net_degree_dict = {}
            for net_id, cell_id in zip(*[ns.tolist() for ns in self.graph.edges(etype='pinned')]):
                if cell_id in partition_set:
                    new_net_degree_dict.setdefault(net_id, 0)
                    new_net_degree_dict[net_id] += 1
            keep_nets_id = np.array(list(new_net_degree_dict.keys()))
            keep_nets_degree = np.array(list(new_net_degree_dict.values()))
            good_nets = np.abs(self.net_prop_dict['degree'][keep_nets_id, 0] - keep_nets_degree) < 1e-5
            good_nets_id = keep_nets_id[good_nets]
            sub_graph = dgl.node_subgraph(self.graph, nodes={'cell': partition, 'net': good_nets_id})
            sub_netlist = Netlist(
                graph=sub_graph,
                cell_prop_dict={k: v[sub_graph.nodes['cell'].data[dgl.NID], :] for k, v in self.cell_prop_dict.items()},
                net_prop_dict={k: v[sub_graph.nodes['net'].data[dgl.NID], :] for k, v in self.net_prop_dict.items()},
                pin_prop_dict={k: v[sub_graph.edges['pinned'].data[dgl.EID], :] for k, v in self.pin_prop_dict.items()},
            )
            sub_netlist.construct_cell_flow()
            self.dict_sub_netlist[temp_n_cell] = sub_netlist
            self.graph.add_nodes(temp_n_cell, ntype='cell')
            self.graph.add_edges(keep_nets_id, [temp_n_cell] * len(keep_nets_id), etype='pinned')
            self.graph.add_edges([temp_n_cell] * len(keep_nets_id), keep_nets_id, etype='pins')
            pseudo_cells_size.append(sub_netlist.layout_size)
            pseudo_cells_degree.append(len(keep_nets_id) - len(good_nets_id))
            pseudo_pins_pos.extend([[0, 0] for _ in range(len(keep_nets_id))])
            temp_n_cell += 1

        pseudo_cells_size = torch.tensor(pseudo_cells_size, dtype=torch.float32)
        pseudo_cells_degree = torch.tensor(pseudo_cells_degree, dtype=torch.float32).unsqueeze(-1)
        pseudo_cells_feat = torch.cat([torch.log(pseudo_cells_size), pseudo_cells_degree], dim=-1)
        self.cell_prop_dict['size'] = torch.vstack([self.cell_prop_dict['size'], pseudo_cells_size])
        self.cell_prop_dict['pos'] = torch.vstack([self.cell_prop_dict['pos'], torch.zeros_like(pseudo_cells_size)])
        self.cell_prop_dict['feat'] = torch.vstack([self.cell_prop_dict['feat'], pseudo_cells_feat])
        self.cell_prop_dict['type'] = torch.vstack([self.cell_prop_dict['type'], torch.zeros_like(pseudo_cells_degree)])

        pseudo_pins_pos = torch.tensor(pseudo_pins_pos, dtype=torch.float32)
        pseudo_pins_io = torch.full(size=[pseudo_pins_pos.shape[0], 1], fill_value=2)
        pseudo_pins_feat = torch.cat([pseudo_pins_pos / 1000, pseudo_pins_io], dim=-1)
        self.pin_prop_dict['pos'] = torch.vstack([self.pin_prop_dict['pos'], pseudo_pins_pos])
        self.pin_prop_dict['io'] = torch.vstack([self.pin_prop_dict['io'], pseudo_pins_io])
        self.pin_prop_dict['feat'] = torch.vstack([self.pin_prop_dict['feat'], pseudo_pins_feat])

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
        self.terminal_indices = [0]

    def construct_cell_flow(self):
        self._cell_flow = CellFlow(self.graph, self.terminal_indices)

    @property
    def cell_flow(self) -> CellFlow:
        if self._cell_flow is None:
            self.construct_cell_flow()
        return self._cell_flow
