import numpy as np
import dgl
from typing import List


def generate_net_tree_from_netlist_graph(graph: dgl.DGLHeteroGraph) -> List[int]:
    pass


if __name__ == '__main__':
    cells_ = [1, 2, 0, 0, 3, 5, 4]
    nets_ = [0, 0, 0, 1, 1, 2, 2]
    n_cell_, n_net_ = 6, 3
    graph_ = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells_, nets_),
        ('net', 'pinned', 'cell'): (nets_, cells_),
    }, num_nodes_dict={'cell': n_cell_, 'net': n_net_})
    net_father_list = generate_net_tree_from_netlist_graph(graph_)
    print(net_father_list)
