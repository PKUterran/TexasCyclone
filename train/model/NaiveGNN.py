import torch
import torch.nn as nn
import dgl
from typing import Tuple, Dict, Any
from dgl.nn.pytorch import HeteroGraphConv, CFConv, GraphConv, SAGEConv
import dgl.function as fn
import numpy as np

from data.graph import Netlist


class NaiveGNN(nn.Module):
    def __init__(
            self,
            raw_cell_feats: int,
            raw_net_feats: int,
            raw_pin_feats: int,
            config: Dict[str, Any]
    ):
        super(NaiveGNN, self).__init__()
        self.device = config['DEVICE']
        self.raw_cell_feats = raw_cell_feats
        self.raw_net_feats = raw_net_feats
        self.raw_pin_feats = raw_pin_feats
        self.hidden_cell_feats = config['CELL_FEATS']
        self.hidden_net_feats = config['NET_FEATS']
        self.hidden_pin_feats = config['PIN_FEATS']
        self.pass_type = config['PASS_TYPE']

        self.cell_lin = nn.Linear(self.raw_cell_feats + 2 * self.raw_net_feats, self.hidden_cell_feats)
        self.net_lin = nn.Linear(self.raw_net_feats + 2 * self.raw_cell_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.raw_pin_feats, self.hidden_pin_feats)

        # 这个naive模型只卷一层，所以直接这么写了。如果需要卷多层的话，建议卷积层单独写一个class，看起来更美观。
        if self.pass_type == 'bidirection':
            self.hetero_conv = HeteroGraphConv({
                'pins': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_net_feats),
                'pinned': CFConv(node_in_feats=self.hidden_net_feats, edge_in_feats=self.hidden_pin_feats,
                                 hidden_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                # 'points-to': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                # 'pointed-from': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
            }, aggregate='max')
        elif self.pass_type == 'single':
            self.hetero_conv = HeteroGraphConv({
                'pinned': SAGEConv(in_feats=(self.hidden_net_feats, self.hidden_cell_feats), aggregator_type='mean',
                                   out_feats=self.hidden_cell_feats),
                # 'points-to': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                # 'pointed-from': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
            }, aggregate='max')
            self.edge_weight_lin = nn.Linear(self.hidden_pin_feats, 1)
        else:
            raise NotImplementedError

        self.edge_dis_readout = nn.Linear(2 * self.hidden_cell_feats + self.hidden_net_feats, 1)
        self.edge_deflect_readout = nn.Linear(3 * self.hidden_cell_feats + 2 * self.hidden_net_feats, 1)
        self.to(self.device)

    def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            cell_size: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ####
        # cell_feat = graph.nodes['cell'].data['feat']
        # net_feat = graph.nodes['net'].data['feat']
        # pin_feat = graph.edges['pin'].data['feat']

        ####
        graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'mean'), etype='pins')
        graph.update_all(fn.copy_u('feat', 'm'), fn.max('m', 'max'), etype='pins')
        graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'mean'), etype='pinned')
        graph.update_all(fn.copy_u('feat', 'm'), fn.max('m', 'max'), etype='pinned')
        cell_feat = torch.cat([
            graph.nodes['cell'].data['feat'],
            graph.nodes['cell'].data['mean'],
            graph.nodes['cell'].data['max'],
        ], dim=-1)
        net_feat = torch.cat([
            graph.nodes['net'].data['feat'],
            graph.nodes['net'].data['mean'],
            graph.nodes['net'].data['max'],
        ], dim=-1)
        pin_feat = graph.edges['pin'].data['feat']

        hidden_cell_feat = torch.tanh(self.cell_lin(cell_feat))
        hidden_net_feat = torch.tanh(self.net_lin(net_feat))
        hidden_pin_feat = torch.tanh(self.pin_lin(pin_feat))

        h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
        graph = graph.to(self.device)
        if self.pass_type == 'bidirection':
            h = self.hetero_conv.forward(graph.edge_type_subgraph(['pins']), h)
            h = {'cell': hidden_cell_feat, 'net': h['net']}
            h = self.hetero_conv.forward(graph.edge_type_subgraph(['pinned']), h,
                                         mod_kwargs={'pinned': {'edge_feats': hidden_pin_feat}})
            hidden_cell_feat = h['cell']
        elif self.pass_type == 'single':
            edge_weight = torch.tanh(self.edge_weight_lin(hidden_pin_feat))
            h = self.hetero_conv.forward(graph.edge_type_subgraph(['pinned']), h,
                                         mod_kwargs={'pinned': {'edge_weight': edge_weight}})
            hidden_cell_feat = h['cell']

        fathers, sons = graph.edges(etype='points-to')
        fathers1, grandfathers = graph.edges(etype='pointed-from')
        fathers2, fs_nets = graph.edges(etype='points-to-net')
        fathers3, gf_nets = graph.edges(etype='pointed-from-net')
        assert torch.equal(fathers, fathers1)
        assert torch.equal(fathers, fathers2)
        assert torch.equal(fathers, fathers3)
        hidden_cell_pair_feat = torch.cat([
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :],
            hidden_net_feat[fs_nets, :]
        ], dim=-1)
        hidden_cell_pair_feat_extend = torch.cat([
            hidden_cell_feat[grandfathers, :],
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :],
            hidden_net_feat[gf_nets, :],
            hidden_net_feat[fs_nets, :]
        ], dim=-1)
        # print(torch.max(self.edge_dis_readout(hidden_cell_pair_feat)),torch.min(self.edge_dis_readout(hidden_cell_pair_feat)))
        edge_dis_ = torch.exp(-2 + 15 * torch.tanh(self.edge_dis_readout(hidden_cell_pair_feat))).view(-1)
        edge_deflect = torch.tanh(self.edge_deflect_readout(hidden_cell_pair_feat_extend)).view(-1) * 2 * torch.pi
        cell_size = cell_size.to(self.device)
        bound_size = (cell_size[fathers] + cell_size[sons]).to(self.device) / 2
        edge_dis = edge_dis_ + torch.min(bound_size, dim=1)[0]
        return edge_dis, edge_deflect
