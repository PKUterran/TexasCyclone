import numpy as np
import torch
import pickle
import dgl
from typing import Dict, List, Tuple

import os, sys
sys.path.append(os.path.abspath('.'))
from data.Netlist import Netlist


class Layout:
    def __init__(
            self, graph: dgl.DGLHeteroGraph,
            cell_prop_dict: Dict[str, torch.Tensor],
            net_prop_dict: Dict[str, torch.Tensor],
            pin_prop_dict: Dict[str, torch.Tensor],
    ):
        self.graph = graph
        self.cell_prop_dict = cell_prop_dict
        self.net_prop_dict = net_prop_dict
        self.pin_prop_dict = pin_prop_dict

    @staticmethod
    def from_netlist_dis_angle(netlist: Netlist, pins_dis: torch.Tensor, pins_angle: torch.Tensor):
        pass
