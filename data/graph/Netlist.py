import numpy as np
import torch
import torch.sparse as sparse
import pickle
import dgl
import tqdm
from typing import Dict, List, Tuple, Union, Optional

import os, sys
sys.path.append(os.path.abspath('.'))


class Netlist:
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

        self.n_cell = graph.num_nodes(ntype='cell')
        self.n_net = graph.num_nodes(ntype='net')
        self.n_pin = graph.num_edges(etype='pinned')
