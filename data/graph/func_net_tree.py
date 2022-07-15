import numpy as np
import dgl
import queue
from typing import List, Dict
from copy import deepcopy


class Tree:
    def __init__(self, father_list: List[int]):
        self.n_node = len(father_list)
        self.father_list = father_list

        self._children_dict = None
        self._path_dict = None

    @property
    def children_dict(self) -> Dict[int, List[int]]:
        if self._children_dict is None:
            self._children_dict = {}
            for i, f in enumerate(self.father_list):
                self._children_dict.setdefault(f, []).append(i)
        return self._children_dict

    @property
    def path_dict(self) -> Dict[int, List[int]]:
        if self._path_dict is None:
            self._path_dict = {}
            stack, path = queue.LifoQueue(), []
            stack.put(-1)
            while not stack.empty():
                k = stack.get()
                if k == -2:
                    path.pop()
                    continue
                path.append(k)
                if k != -1:
                    self._path_dict[k] = deepcopy(path[2:])
                stack.put(-2)
                for child in self.children_dict.get(k, []):
                    stack.put(child)
        return self._path_dict

    def father_of(self, idx: int) -> int:
        return self.father_list[idx]

    def children_of(self, idx: int) -> List[int]:
        return self.children_dict[idx]

    def path_of(self, idx: int) -> List[int]:
        return self.path_dict[idx]


def generate_net_tree_from_netlist_graph(graph: dgl.DGLHeteroGraph) -> Tree:
    n_net = graph.num_nodes(ntype='net')

    # TODO: better tree generate
    # father of root is -1
    father_list = list(range(-1, n_net - 1))

    # TODO end

    return Tree(father_list)


if __name__ == '__main__':
    cells_ = [1, 2, 0, 0, 3, 5, 4]
    nets_ = [0, 0, 0, 1, 1, 2, 2]
    n_cell_, n_net_ = 6, 3
    graph_ = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells_, nets_),
        ('net', 'pinned', 'cell'): (nets_, cells_),
    }, num_nodes_dict={'cell': n_cell_, 'net': n_net_})
    net_tree = generate_net_tree_from_netlist_graph(graph_)
    print(net_tree.n_node)
    print(net_tree.father_list)
    print(net_tree.children_dict)
    print(net_tree.path_dict)
    assert net_tree.n_node == n_net_, f"{net_tree.n_node} != {n_net_}"
