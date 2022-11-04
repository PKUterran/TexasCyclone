import torch
import torch.nn as nn
import dgl
from typing import Tuple, Dict, Any
from dgl.nn.pytorch import HeteroGraphConv, CFConv, GraphConv, SAGEConv
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

        self.cell_lin = nn.Linear(self.raw_cell_feats, self.hidden_cell_feats)
        self.net_lin = nn.Linear(self.raw_net_feats, self.hidden_net_feats)
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
                'pinned': SAGEConv(in_feats=(self.hidden_net_feats,self.hidden_cell_feats),aggregator_type='mean',out_feats=self.hidden_cell_feats),
                # 'points-to': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                # 'pointed-from': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
            }, aggregate='max')
            self.edge_weight_lin = nn.Linear(self.hidden_pin_feats,1)
        else:
            raise NotImplementedError

        self.edge_dis_readout = nn.Linear(2 * self.hidden_cell_feats, 1)
        self.edge_angle_readout = nn.Linear(2 * self.hidden_cell_feats, 1)
        self.to(self.device)

    def forward(
            self,
            graph: dgl.DGLHeteroGraph,
            feature: Tuple[torch.tensor, torch.tensor, torch.tensor],
            cell_size : torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_feat, net_feat, pin_feat = feature
        cell_feat = cell_feat.to(self.device)
        net_feat = net_feat.to(self.device)
        pin_feat = pin_feat.to(self.device)
        
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
        hidden_cell_pair_feat = torch.cat([
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :]
        ], dim=-1)
        # print(torch.max(self.edge_dis_readout(hidden_cell_pair_feat)),torch.min(self.edge_dis_readout(hidden_cell_pair_feat)))
        edge_dis_ = torch.exp(-2+15 * torch.tanh(self.edge_dis_readout(hidden_cell_pair_feat))).view(-1)
        edge_angle = torch.tanh(self.edge_angle_readout(hidden_cell_pair_feat)).view(-1) * 4
        cell_size = cell_size.to(self.device)
        bound_size = (cell_size[fathers] + cell_size[sons]).to(self.device) / 2
        eps = torch.ones_like(edge_angle).to(self.device) * 1e-4
        tmp = torch.min(torch.abs(bound_size[:,0] / (torch.cos(edge_angle*np.pi)+eps)),torch.abs(bound_size[:,1] / (torch.sin(edge_angle*np.pi)+eps)))
        edge_dis = edge_dis_ + tmp
        return edge_dis, edge_angle

    def forward_with_netlist(
            self,
            netlist: Netlist
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph = netlist.graph
        cell_feat = netlist.cell_prop_dict['feat']
        net_feat = netlist.net_prop_dict['feat']
        pin_feat = netlist.pin_prop_dict['feat']
        return self.forward(graph, (cell_feat, net_feat, pin_feat,None))
