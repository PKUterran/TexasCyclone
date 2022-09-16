import numpy as np
import dgl
import queue
from typing import List, Dict
from copy import deepcopy

# from RL.TexasCyclone.data.Netlist import Netlist
from typing import Tuple
import pickle
import networkx as nx
import itertools
import random
from matplotlib import pyplot as plt
import torch
import dgl.function as fn
import time
# from RL.TexasCyclone.data.Netlist import netlist_from_numpy_directory

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

######通过最短路迭代得到生成树
def generate_net_tree_from_netlist_graph_short_path(graph: dgl.DGLHeteroGraph) -> Tree:
    n_net = graph.num_nodes(ntype='net')

    # TODO: better tree generate
    # father of root is -1
    # father_list = list(range(-1, n_net - 1))
    net_edge_list = set()
    num_cell = graph.num_nodes('cell')
    cell2net_dict_set = {}
    src_,dst_ = graph.edges(etype='pins')
    for i in range(num_cell):
        cell2net_dict_set[i] = set()
    for src,dst in zip(src_,dst_):
        src = int(src)
        cell2net_dict_set[src].add(dst)
    for cell2net_list in cell2net_dict_set.values():
        cell2net_list = list(cell2net_list)
        for src,dst in itertools.product(cell2net_list,cell2net_list):
            if src != dst:
                net_edge_list.add((int(src),int(dst)))
    net_edge_list = list(net_edge_list)
    nx_graph = nx.Graph(net_edge_list)
    ##################
    center_node = random.choice(list(nx_graph.nodes))
    num_net = len(nx_graph.nodes)#
    num_edges = len(nx_graph.edges)
    print(num_net,num_edges)
    max_iter = 20
    min_node_length = 1e9
    min_node_depth = 1e9
    best_center_node = -1
    for _ in range(max_iter):
        
        shortest_path_len = nx.single_source_dijkstra_path_length(G=nx_graph,source=center_node)
        total_node_length = sum(shortest_path_len.values())
        max_tree_depth = max(shortest_path_len.values())
        averange_depth = round(total_node_length / num_net)
        print(total_node_length,max_tree_depth)
        if(max_tree_depth < min_node_depth or 
            (min_node_depth == max_tree_depth and total_node_length < min_node_length) or
            (min_node_depth == max_tree_depth and total_node_length == min_node_length and nx_graph.degree[best_center_node] < nx_graph.degree[center_node])):
            min_node_depth = max_tree_depth
            min_node_length = total_node_length
            best_center_node = center_node
        choose_node_list = []
        p = []
        for node,distance in zip(shortest_path_len.keys(),shortest_path_len.values()):
            if abs(distance - averange_depth)<=2:
                choose_node_list.append(node)
                p.append(nx_graph.degree[node])
        p = np.array(p) / sum(p)
        center_node = np.random.choice(choose_node_list,1,p=p)[0]
        print(center_node,nx_graph.degree[center_node])
    print(f'best center node is {best_center_node} total node length is {min_node_length} max depth is {min_node_depth}')
    pred, _ = nx.dijkstra_predecessor_and_distance(nx_graph, best_center_node)
    tree_edge_list = []
    father_list = list(range(-1, n_net - 1))
    for v,father in pred.items():
        if father == []:
            father_list[v] = -1
            pass
        else:
            father_list[v] = father[0]
            tree_edge_list.append((v,father[0]))
            tree_edge_list.append((father[0],v))
    # father_list = list(range(-1, n_net - 1))
    pass


    # TODO end

    return Tree(father_list)

######通过pagerank得到生成树
def generate_net_tree_from_netlist_graph(graph: dgl.DGLHeteroGraph) -> Tree:

    # TODO: better tree generate
    # father of root is -1
    num_cell = graph.num_nodes('cell')
    num_net = graph.num_nodes('net')
    src_pins,dst_pins = graph.edges(etype='pins')
    src_pinned,dst_pinned = graph.edges(etype='pinned')
    u = torch.cat([src_pins+num_net,src_pinned])
    v = torch.cat([dst_pins,dst_pinned+num_net])
    g = dgl.graph((u,v))


    edge_list = [(int(a),int(b)) for a,b in zip(u,v)]
    nx_graph = nx.Graph(edge_list)


    g.ndata['pagerank'] = torch.ones(g.num_nodes()) / g.num_nodes()
    g.ndata["degree"]=g.out_degrees(g.nodes()).float()
    ##################
    ########pagerank计算中心点
    num_node = g.num_nodes()#len(nx_graph.nodes)#

    max_iter = 5
    best_center_node = -1
    damping_value = (1.0-0.85) / num_node

    
    for _ in range(max_iter):
        g.ndata['pagerank'] = g.ndata['pagerank'] / g.ndata['degree']
        g.update_all(message_func=fn.copy_src(src='pagerank', out='m'),
                    reduce_func=fn.sum(msg='m',out='m_sum'))
        g.ndata['pagerank'] = damping_value + 0.85 * g.ndata['m_sum']
        ########选取pagerank最大的那个作为中心点
        best_center_node = int(torch.argmax(g.ndata['pagerank'][:num_net]).long())
    pred, _ = nx.dijkstra_predecessor_and_distance(nx_graph, best_center_node)
    father_list = list(range(-1, num_cell+num_net - 1))
    for v,father in pred.items():
        if father == []:
            father_list[v] = -1
            pass
        else:
            father_list[v] = father[0]
            
    for i in range(num_net):
        if father_list[i] == -1:
            continue
        father_list[i] = father_list[father_list[i]]
    father_list = father_list[:num_net]
    pass

    # TODO end

    return Tree(father_list)


def load_graph(dir_name: str) -> dgl.DGLHeteroGraph:
    pin_net_cell = np.load(f'{dir_name}/pin_net_cell.npy')
    cell_data = np.load(f'{dir_name}/cell_data.npy')
    net_data = np.load(f'{dir_name}/net_data.npy')
    n_cell, n_net = cell_data.shape[0], net_data.shape[0]
    cells = list(pin_net_cell[:, 1])
    nets = list(pin_net_cell[:, 0])

    graph = dgl.heterograph({
        ('cell', 'pins', 'net'): (cells, nets),
        ('net', 'pinned', 'cell'): (nets, cells),
    }, num_nodes_dict={'cell': n_cell, 'net': n_net})

    return graph

if __name__ == '__main__':
    # cells_ = [1, 2, 0, 0, 3, 5, 4]
    # nets_ = [0, 0, 0, 1, 1, 2, 2]
    # n_cell_, n_net_ = 6, 3
    # graph_ = dgl.heterograph({
    #     ('cell', 'pins', 'net'): (cells_, nets_),
    #     ('net', 'pinned', 'cell'): (nets_, cells_),
    # }, num_nodes_dict={'cell': n_cell_, 'net': n_net_})
    a = time.time()
    graph_ = load_graph('/home/xuyanyang/RL/Placement-datasets/ispd2015/mgc_superblue14')
    b = time.time()
    net_tree = generate_net_tree_from_netlist_graph(graph_)
    c = time.time()
    print(f"load time is {b-a} generate tree time is {c-b}")
    # print(net_tree.n_node)
    # print(net_tree.father_list)
    # print(net_tree.children_dict)
    # print(net_tree.path_dict)
    # assert net_tree.n_node == n_net_, f"{net_tree.n_node} != {n_net_}"
