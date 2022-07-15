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
            self, netlist: Netlist,
    ):
        self.netlist = netlist


def layout_from_netlist_dis_angle(netlist: Netlist, pins_dis: torch.Tensor, pins_angle: torch.Tensor
                                  ) -> Tuple[Layout, torch.Tensor]:
    pass
