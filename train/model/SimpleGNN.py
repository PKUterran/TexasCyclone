import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any,Dict,List,Tuple
from dgl.nn.pytorch import HeteroGraphConv,CFConv,GraphConv

from data.Netlist import Netlist


class SimpleGNN(nn.Module):
    def __init__(
            self,
            raw_cell_feats: int,
            raw_net_feats: int,
            raw_pin_feats: int,
            config: Dict[str,Any]):
        super(SimpleGNN,self).__init__()
        self.device = config['DEVICE']
        
        self.raw_cell_feats = raw_cell_feats
        self.raw_net_feats = raw_net_feats
        self.raw_pin_feats = raw_pin_feats

        self.hidden_cell_feats = config['CELL_FEATS']
        self.hidden_net_feats = config['NET_FEATS']
        self.hidden_pin_feats = config['PIN_FEATS']
        self.num_layers = config['NUM_LAYERS']

        self.cell_lin = nn.Linear(self.raw_cell_feats,self.hidden_cell_feats)
        self.net_lin = nn.Linear(self.raw_net_feats,self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.raw_pin_feats,self.hidden_pin_feats)

        self.hetero_conv = HeteroGraphConvLayers(
                hidden_cell_feats=self.hidden_cell_feats,
                hidden_net_feats=self.hidden_net_feats,
                hidden_pin_feats=self.hidden_pin_feats,
                num_layers=3,
                device=self.device)
        self.net_dis_readout = nn.Linear(2 * self.hidden_net_feats, 1)
        self.net_angle_readout = nn.Linear(2 * self.hidden_net_feats, 1)
        self.pin_dis_readout = nn.Linear(self.hidden_cell_feats + self.hidden_pin_feats + self.hidden_net_feats, 1)
        self.pin_angle_readout = nn.Linear(self.hidden_cell_feats + self.hidden_pin_feats + self.hidden_net_feats, 1)
        
        self.to(self.device)

    def forward(
            self,
            netlist: Netlist,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        cell_feat = netlist.cell_prop_dict['feat'].to(self.device)
        net_feat = netlist.net_prop_dict['feat'].to(self.device)
        pin_feat = netlist.pin_prop_dict['feat'].to(self.device)
        hidden_cell_feat = F.tanh(self.cell_lin(cell_feat))
        hidden_net_feat = F.tanh(self.net_lin(net_feat))
        hidden_pin_feat = F.tanh(self.pin_lin(pin_feat))

        h = self.hetero_conv(
                netlist,
                hidden_cell_feat,
                hidden_net_feat,
                hidden_pin_feat)
        hidden_cell_feat, hidden_net_feat = h['cell'], h['net']

        net_net_pair_matrix = netlist.net_net_pair_matrix.to(self.device)
        net_cell_pair_matrix = netlist.net_cell_pair_matrix.to(self.device)

        hidden_net_pair_feat = torch.cat([
        hidden_net_feat[net_net_pair_matrix[:, 0], :],
        hidden_net_feat[net_net_pair_matrix[:, 1], :]
        ], dim=-1)
        hidden_net_pin_cell_feat = torch.cat([
        hidden_net_feat[net_cell_pair_matrix[:, 0], :],
        hidden_pin_feat,
        hidden_cell_feat[net_cell_pair_matrix[:, 1], :],
        ], dim=-1)
        net_dis = F.softplus(self.net_dis_readout(hidden_net_pair_feat)).view(-1)
        net_angle = 2*F.sigmoid(self.net_angle_readout(hidden_net_pair_feat)).view(-1)
        pin_dis = F.softplus(self.pin_dis_readout(hidden_net_pin_cell_feat)).view(-1)
        pin_angle = 2*F.sigmoid(self.pin_angle_readout(hidden_net_pin_cell_feat)).view(-1)
        
        return net_dis, net_angle, pin_dis, pin_angle
                





class HeteroGraphConvLayers(nn.Module):
    def __init__(
            self,
            hidden_cell_feats: int,
            hidden_net_feats: int,
            hidden_pin_feats: int,
            num_layers: int,
            device: torch.device) -> None:

        super(HeteroGraphConvLayers,self).__init__()

        self.num_layers = num_layers
        self.hidden_cell_feats = hidden_cell_feats
        self.hidden_net_feats = hidden_net_feats
        self.hidden_pin_feats = hidden_pin_feats

        self.heterographconvlayers = []

        for _ in range(num_layers):
                self.heterographconvlayers.append(
                        HeteroGraphConv({
                        'pins': GraphConv(in_feats=self.hidden_cell_feats, out_feats=self.hidden_net_feats),
                        'pinned': CFConv(node_in_feats=self.hidden_net_feats, edge_in_feats=self.hidden_pin_feats,
                                        hidden_feats=self.hidden_cell_feats, out_feats=self.hidden_cell_feats),
                        'father': GraphConv(in_feats=self.hidden_net_feats, out_feats=self.hidden_net_feats),
                        'son': GraphConv(in_feats=self.hidden_net_feats, out_feats=self.hidden_net_feats),
                }, aggregate='max').to(device)
                )
        self.device = device
        self.to(self.device)
    def forward(
            self,
            netlist: Netlist,
            hidden_cell_feat: torch.Tensor,
            hidden_net_feat: torch.Tensor,
            hidden_pin_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        for heterconvlayer in self.heterographconvlayers:
                h = {'cell': hidden_cell_feat, 'net': hidden_net_feat}
                h = heterconvlayer.forward(netlist.graph, h,mod_kwargs={'pinned': {'edge_feats': hidden_pin_feat}})
                hidden_cell_feat, hidden_net_feat = h['cell'], h['net']
        
        return {'cell': hidden_cell_feat, 'net': hidden_net_feat}
