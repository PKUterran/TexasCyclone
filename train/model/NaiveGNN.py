import torch
import torch.nn as nn
import dgl
from time import time
from typing import Tuple, Dict, Any
from dgl.nn.pytorch import HeteroGraphConv, CFConv, GraphConv

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

        self.cell_lin = nn.Linear(self.raw_cell_feats, self.hidden_cell_feats)
        self.net_lin = nn.Linear(self.raw_net_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.raw_pin_feats, self.hidden_pin_feats)

        # 这个naive模型只卷一层，所以直接这么写了。如果需要卷多层的话，建议卷积层单独写一个class，看起来更美观。
        self.hetero_conv = HeteroGraphConv({
            'pins': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_net_feats),
            'pinned': CFConv(node_in_feats=self.hidden_net_feats, edge_in_feats=self.hidden_pin_feats,
                             hidden_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
#             'points-to': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
#             'pointed-from': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
        }, aggregate='max')

        self.edge_dis_readout = nn.Linear(2 * self.hidden_cell_feats, 1)
        self.edge_angle_readout = nn.Linear(2 * self.hidden_cell_feats, 1)
        self.to(self.device)

    def forward(
            self,
            netlist: Netlist
    ) -> Tuple[torch.Tensor, torch.Tensor]:
#         t0 = time()
        cell_feat = netlist.cell_prop_dict['feat'].to(self.device)
        net_feat = netlist.net_prop_dict['feat'].to(self.device)
        pin_feat = netlist.pin_prop_dict['feat'].to(self.device)
#         print('in train 1:', time() - t0)
        hidden_cell_feat = torch.tanh(self.cell_lin(cell_feat))
        hidden_net_feat = torch.tanh(self.net_lin(net_feat))
        hidden_pin_feat = torch.tanh(self.pin_lin(pin_feat))
#         print('in train 2:', time() - t0)

        h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
        graph = netlist.graph.to(self.device)
#         print('in train 3:', time() - t0)
#         print('# node:', graph.num_nodes(ntype='cell'))
#         print('# net:', graph.num_nodes(ntype='net'))
#         print('# pins:', graph.num_edges(etype='pins'))
#         print('# points-to:', graph.num_edges(etype='points-to'))
        h = self.hetero_conv.forward(graph.edge_type_subgraph(['pins', 'pinned']), h, mod_kwargs={'pinned': {'edge_feats': hidden_pin_feat}})
        hidden_cell_feat, hidden_net_feat = h['cell'], h['net']
#         print('in train 4:', time() - t0)

        fathers, sons = graph.edges(etype='points-to')
        hidden_cell_pair_feat = torch.cat([
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :]
        ], dim=-1)
        edge_dis = torch.exp(self.edge_dis_readout(hidden_cell_pair_feat)).view(-1)
        edge_angle = torch.tanh(self.edge_angle_readout(hidden_cell_pair_feat)).view(-1) * 4
#         print('in train 5:', time() - t0)
        return edge_dis, edge_angle


    def forward_with_feat(
            self,
            graph: dgl.DGLHeteroGraph,
            cell_feat: torch.Tensor,
            net_feat: torch.Tensor,
            pin_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
#         t0 = time()
        cell_feat = cell_feat.to(self.device)
        net_feat = net_feat.to(self.device)
        pin_feat = pin_feat.to(self.device)
#         print('in train 1:', time() - t0)
        hidden_cell_feat = torch.tanh(self.cell_lin(cell_feat))
        hidden_net_feat = torch.tanh(self.net_lin(net_feat))
        hidden_pin_feat = torch.tanh(self.pin_lin(pin_feat))
#         print('in train 2:', time() - t0)

        h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
        graph = graph.to(self.device)
#         print('in train 3:', time() - t0)
#         print('# node:', graph.num_nodes(ntype='cell'))
#         print('# net:', graph.num_nodes(ntype='net'))
#         print('# pins:', graph.num_edges(etype='pins'))
#         print('# points-to:', graph.num_edges(etype='points-to'))
        h = self.hetero_conv.forward(graph.edge_type_subgraph(['pins', 'pinned']), h, mod_kwargs={'pinned': {'edge_feats': hidden_pin_feat}})
        hidden_cell_feat, hidden_net_feat = h['cell'], h['net']
#         print('in train 4:', time() - t0)

        fathers, sons = graph.edges(etype='points-to')
        hidden_cell_pair_feat = torch.cat([
            hidden_cell_feat[fathers, :],
            hidden_cell_feat[sons, :]
        ], dim=-1)
        edge_dis = torch.exp(self.edge_dis_readout(hidden_cell_pair_feat)).view(-1)
        edge_angle = torch.tanh(self.edge_angle_readout(hidden_cell_pair_feat)).view(-1) * 4
#         print('in train 5:', time() - t0)
        return edge_dis, edge_angle
